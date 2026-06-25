// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
// ⚠️ DRAFT — compile on macOS 27 (CoreAI framework). The host autoregressive loop mirrors,
// op-for-op, the validated Python reference `conversion/verify_package.py`:
//   text → tokens → [voice-seeded backbone(stateful KV) → flow] × frames → mimi → PCM.
import CoreAI
import CoreAIShared
import Foundation

/// On-device Pocket-TTS synthesizer driving the three exported `.aimodel` bundles.
public final class PocketTTS: @unchecked Sendable {
    public let package: ModelPackage
    public let sampleRate: Int

    private let tokenizer: UnigramTokenizer

    // Bundles.
    private let backbone: Bundle
    private let flow: Bundle
    private let mimi: Bundle

    // Host-side glue (fp32).
    private let embed: [Float]          // [n_bins+1, D]
    private let inputLinear: [Float]    // [D, ldim]
    private let bosEmb: [Float]         // [ldim]
    private let eosW: [Float]           // [D]
    private let eosB: Float

    // Scalars.
    private let D: Int, ldim: Int, L: Int, H: Int, Dh: Int, cacheLen: Int
    private let temp: Float, eosThreshold: Float, frameSamples: Int

    public init(packageURL: URL) async throws {
        self.package = try ModelPackage(root: packageURL)
        let m = package.manifest
        self.sampleRate = m.sample_rate
        self.frameSamples = m.frame_samples
        let s = m.scalars
        (D, ldim, L, H, Dh, cacheLen) =
            (1024, s.ldim, s.num_layers, s.num_heads, s.dim_per_head, s.cache_len)
        self.temp = Float(s.temp)
        self.eosThreshold = Float(s.eos_threshold)

        self.tokenizer = try UnigramTokenizer(jsonURL: package.url(m.tokenizer_json))

        self.embed = try package.glue("embed")
        self.inputLinear = try package.glue("input_linear")
        self.bosEmb = try package.glue("bos_emb")
        self.eosW = try package.glue("out_eos_w")
        self.eosB = try package.glue("out_eos_b").first ?? 0

        func bundleURL(_ k: String) throws -> URL {
            guard let rel = m.bundles[k] else { throw PocketError.missing("bundle.\(k)") }
            return package.url(rel)
        }
        self.backbone = try await Bundle.load(at: bundleURL("backbone"))
        self.flow = try await Bundle.load(at: bundleURL("flow"))
        self.mimi = try await Bundle.load(at: bundleURL("mimi"))
    }

    public func validateTokenizer() -> String? { tokenizer.runSelfTest() }

    /// Synthesize speech for `text` with one of the packaged `voice` states. Returns mono PCM
    /// in [-1, 1] at `sampleRate`. `maxFrames` caps runaway generation (12.5 frames/s).
    public func synthesize(text: String, voice: String, seed: UInt64 = 0,
                           maxFrames: Int = 400) async throws -> [Float] {
        try await synthesize(tokens: tokenizer.encode(text), voice: voice, seed: seed,
                             maxFrames: maxFrames)
    }

    public func synthesize(tokens: [Int], voice: String, seed: UInt64 = 0,
                           maxFrames: Int = 400) async throws -> [Float] {
        guard let v = package.manifest.voices[voice] else { throw PocketError.missing("voice.\(voice)") }

        // ---- voice-seeded KV state (full [L,1,H,cacheLen,Dh], zero past `offset`) ----
        let off = v.offset
        let vkey = try package.floats(v.key)      // [L,1,H,off,Dh]
        let vval = try package.floats(v.value)
        var keyCache = backbone.makeState("keyCache", fullShape: [L, 1, H, cacheLen, Dh])
        var valCache = backbone.makeState("valueCache", fullShape: [L, 1, H, cacheLen, Dh])
        seedKV(&keyCache, vkey, off: off)
        seedKV(&valCache, vval, off: off)

        var pos = off

        // backbone step: inputs_embeds[1,1,D], pos[1] (+persistent KV) -> hidden[1,1,D]
        func step(_ emb: [Float]) async throws -> [Float] {
            var inEmb = backbone.makeInput("inputs_embeds", shape: [1, 1, D]); fill16(&inEmb, emb)
            var posA = backbone.makeInput("pos", shape: [1]); fill32(&posA, count: 1) { _ in Int32(pos) }
            var hidden = backbone.makeOutput("hidden", shape: [1, 1, D])
            var states = InferenceFunction.MutableViews()
            states.insert(&keyCache, for: "keyCache")
            states.insert(&valCache, for: "valueCache")
            var outs = InferenceFunction.MutableViews()
            outs.insert(&hidden, for: "hidden")
            _ = try await backbone.fn.run(inputs: ["inputs_embeds": inEmb, "pos": posA],
                                          states: consume states, outputViews: consume outs)
            pos += 1
            return read16(hidden, count: D)
        }

        // ---- text prefill (prefill-via-decode: one token embed at a time) ----
        for tid in tokens { _ = try await step(embedRow(tid)) }

        // ---- autoregressive decode ----
        var rng = SplitMix64(seed: seed)
        var latents: [[Float]] = []
        var prev: [Float]? = nil
        var eosStep: Int? = nil
        var i = 0
        while true {
            let seq = prev ?? bosEmb
            let hidden = try await step(matvec(inputLinear, seq, rows: D, cols: ldim))  // input_linear @ seq
            let eos = (dot(eosW, hidden) + eosB) > eosThreshold
            let latent = try await runFlow(cond: hidden, z: sampleNoise(&rng))
            if eos && eosStep == nil { eosStep = i }
            if let e = eosStep, i >= e + 2 { break }
            latents.append(latent); prev = latent; i += 1
            if i >= maxFrames { break }
        }
        if latents.isEmpty { return [] }

        return try await runMimi(latents)
    }

    // ----- flow bundle: cond[1,D], z[1,ldim] -> latent[1,ldim] -----
    private func runFlow(cond: [Float], z: [Float]) async throws -> [Float] {
        var condA = flow.makeInput("cond", shape: [1, D]); fill16(&condA, cond)
        var zA = flow.makeInput("z", shape: [1, ldim]); fill16(&zA, z)
        let outs = try await flow.fn.run(inputs: ["cond": condA, "z": zA])
        guard let nd = outs.remove("latent")?.ndArray else { throw PocketError.engine("flow: no latent") }
        return read16(nd, count: ldim)
    }

    // ----- mimi bundle: latents[1,T,ldim] -> wav[1,1,frameSamples*T] (dynamic T) -----
    private func runMimi(_ latents: [[Float]]) async throws -> [Float] {
        let T = latents.count
        var flat = [Float](); flat.reserveCapacity(T * ldim)
        for l in latents { flat.append(contentsOf: l) }
        var latA = mimi.makeInput("latents", shape: [1, T, ldim]); fill16(&latA, flat)
        let outs = try await mimi.fn.run(inputs: ["latents": latA])
        guard let nd = outs.remove("wav")?.ndArray else { throw PocketError.engine("mimi: no wav") }
        return read16(nd, count: frameSamples * T)
    }

    // ---- host math ----
    private func embedRow(_ tid: Int) -> [Float] { Array(embed[(tid * D)..<((tid + 1) * D)]) }

    /// y[r] = Σ_c W[r,c]·x[c], W row-major [rows, cols].
    private func matvec(_ W: [Float], _ x: [Float], rows: Int, cols: Int) -> [Float] {
        var y = [Float](repeating: 0, count: rows)
        W.withUnsafeBufferPointer { w in
            for r in 0..<rows {
                var acc: Float = 0; let base = r * cols
                for c in 0..<cols { acc += w[base + c] * x[c] }
                y[r] = acc
            }
        }
        return y
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        var s: Float = 0; for i in 0..<min(a.count, b.count) { s += a[i] * b[i] }; return s
    }

    /// z ~ N(0, temp) (Pocket-TTS scales the flow noise by sqrt(temp)).
    private func sampleNoise(_ rng: inout SplitMix64) -> [Float] {
        let std = temp.squareRoot()
        return (0..<ldim).map { _ in rng.nextGaussian() * std }
    }

    /// Write voice K/V (`[L,1,H,off,Dh]`) into the first `off` cache slots of a zeroed full state.
    private func seedKV(_ state: inout NDArray, _ src: [Float], off: Int) {
        let perLayerFull = H * cacheLen * Dh
        let perLayerSrc = H * off * Dh
        var view = state.mutableView(as: Float16.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for li in 0..<L {
                for h in 0..<H {
                    for t in 0..<off {
                        let dst = li * perLayerFull + h * cacheLen * Dh + t * Dh
                        let s = li * perLayerSrc + h * off * Dh + t * Dh
                        for d in 0..<Dh { ptr[dst + d] = Float16(src[s + d]) }
                    }
                }
            }
        }
    }
}

/// Deterministic Gaussian RNG (SplitMix64 + Box–Muller) for reproducible synthesis.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E3779B97F4A7C15 }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func nextUnit() -> Float { Float(next() >> 11) * Float(1.0 / 9007199254740992.0) }
    mutating func nextGaussian() -> Float {
        let u1 = max(nextUnit(), 1e-7), u2 = nextUnit()
        return (-2 * Foundation.log(u1)).squareRoot() * Foundation.cos(2 * .pi * u2)
    }
}
