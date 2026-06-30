// VibeVoiceEngine — the on-device streaming TTS pipeline.
//
// Mirrors `VibeVoiceStreamingForConditionalGenerationInference.generate`:
//   * text is fed in windows of 5 tokens through base_lm + tts_lm,
//   * after each window, 6 acoustic frames are produced by a CFG DPM-Solver loop
//     over the diffusion head, decoded to audio via the acoustic decoder,
//   * EOS is predicted per frame by the binary classifier.
// KV caches for base_lm / tts_lm / neg_tts_lm are seeded from the chosen voice's
// prefill and carried as Core AI state across steps.

import CoreAI
import Foundation
import Tokenizers

/// Filesystem layout of the converted model + voices + tokenizer.
public struct VibeVoiceAssets: Sendable {
    public let root: URL
    public var diffusionHead: URL { root.appending(path: "diffusion_head.aimodel") }
    public var baseLM: URL { root.appending(path: "base_lm.aimodel") }
    public var ttsLM: URL { root.appending(path: "tts_lm.aimodel") }
    public var connector: URL { root.appending(path: "acoustic_connector.aimodel") }
    public var eos: URL { root.appending(path: "eos_classifier.aimodel") }
    public var decoder: URL { root.appending(path: "acoustic_decoder.aimodel") }
    public var resources: URL { root.appending(path: "resources") }
    public var voicesDir: URL { root.appending(path: "voices") }
    public var tokenizerDir: URL { root.appending(path: "tokenizer") }

    public init(root: URL) { self.root = root }
}

public struct GenerationOptions: Sendable {
    public var cfgScale: Float = 1.5
    public var diffusionSteps: Int = 5
    public var maxContext: Int = 4096
    public var decoderFrames: Int = 256   // fixed block size of acoustic_decoder.aimodel
    public var decoderOverlap: Int = 8    // crossfade frames between decode blocks
    public var seed: UInt64?

    public init() {}
}

public struct AudioResult: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public var duration: Double { Double(samples.count) / Double(sampleRate) }
}

public actor VibeVoiceEngine {
    public let assets: VibeVoiceAssets
    public let catalog: VoiceCatalog

    private let constants = ModelConstants()
    private let debug = ProcessInfo.processInfo.environment["VV_DEBUG"] == "1"
    private let typeEmbedding: TypeEmbedding
    private let scaleBias: ScaleBias
    private let audio: AudioConfig
    private let tokenizer: any Tokenizer

    private let diffusionHead: StatelessComponent
    private let connector: StatelessComponent
    private let eosClassifier: StatelessComponent
    private let decoder: StatelessComponent

    public init(assets: VibeVoiceAssets) async throws {
        self.assets = assets
        self.catalog = try VoiceCatalog(directory: assets.voicesDir)
        self.typeEmbedding = try TypeEmbedding(contentsOf: assets.resources.appending(path: "type_embedding.json"))
        self.scaleBias = try ScaleBias(contentsOf: assets.resources.appending(path: "scale_bias.json"))
        self.audio = try AudioConfig(contentsOf: assets.resources.appending(path: "audio.json"))
        self.tokenizer = try await AutoTokenizer.from(modelFolder: assets.tokenizerDir)

        self.diffusionHead = StatelessComponent(url: assets.diffusionHead, preferredCompute: .gpu)
        self.connector = StatelessComponent(url: assets.connector, preferredCompute: .gpu)
        self.eosClassifier = StatelessComponent(url: assets.eos, preferredCompute: .gpu)
        self.decoder = StatelessComponent(url: assets.decoder, preferredCompute: .gpu)
    }

    public func availableVoices() -> [VoiceInfo] { catalog.voices }
    public func languages() -> [String] { catalog.languages }

    // MARK: - Synthesis

    /// Synthesize speech for `text` in the named `voice`.
    public func synthesize(
        text: String,
        voice voiceName: String,
        options: GenerationOptions = GenerationOptions(),
        onFrame: (@Sendable (Int) -> Void)? = nil
    ) async throws -> AudioResult {
        let voice = try catalog.load(named: voiceName)
        return try await synthesize(text: text, voice: voice, options: options, onFrame: onFrame)
    }

    public func synthesize(
        text: String,
        voice: Voice,
        options: GenerationOptions = GenerationOptions(),
        onFrame: (@Sendable (Int) -> Void)? = nil
    ) async throws -> AudioResult {
        let normalized = text
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
        let textIDs: [Int32] = tokenizer
            .encode(text: normalized.trimmingCharacters(in: .whitespacesAndNewlines) + "\n")
            .map { Int32($0) }
        if debug {
            FileHandle.standardError.write("textIDs (\(textIDs.count)): \(textIDs)\n".data(using: .utf8)!)
        }

        // Seed the three live KV stacks from the voice prefill.
        let baseLM = Qwen2Stack(
            url: assets.baseLM, kind: .tokenIDs,
            layers: constants.baseLayers, kvHeads: constants.kvHeads, headDim: constants.headDim,
            hiddenSize: constants.hiddenSize, maxContext: options.maxContext, preferredCompute: .gpu)
        let ttsLM = Qwen2Stack(
            url: assets.ttsLM, kind: .embeds,
            layers: constants.ttsLayers, kvHeads: constants.kvHeads, headDim: constants.headDim,
            hiddenSize: constants.hiddenSize, maxContext: options.maxContext, preferredCompute: .gpu)
        let negTTSLM = Qwen2Stack(
            url: assets.ttsLM, kind: .embeds,
            layers: constants.ttsLayers, kvHeads: constants.kvHeads, headDim: constants.headDim,
            hiddenSize: constants.hiddenSize, maxContext: options.maxContext, preferredCompute: .gpu)
        try await baseLM.seed(with: voice.lm)
        try await ttsLM.seed(with: voice.ttsLM)
        try await negTTSLM.seed(with: voice.negTTSLM)

        let typeText = typeEmbedding.row(1)
        let typeSpeech = typeEmbedding.row(0)
        var rng = GaussianRNG(seed: options.seed ?? UInt64.random(in: .min ... .max))

        var ttsHidden = voice.ttsLM.lastHidden          // condition for first frame
        var negHidden = voice.negTTSLM.lastHidden
        var scaledLatents: [[Float]] = []
        var finished = false
        var windowIndex = 0
        let maxLen = options.maxContext - 2

        while !finished {
            let start = windowIndex * constants.textWindow
            let end = min(start + constants.textWindow, textIDs.count)
            windowIndex += 1
            let window = (start < end) ? Array(textIDs[start..<end]) : []

            if !window.isEmpty {
                let baseHidden = try await baseLM.forward(tokenIDs: window)  // w × hidden
                if debug && windowIndex == 1 {
                    let last = lastRow(baseHidden, hidden: constants.hiddenSize)
                    let n = sqrt(last.reduce(0) { $0 + $1 * $1 })
                    let head = last.prefix(6).map { String(format: "%.4f", $0) }
                    FileHandle.standardError.write("  BASE lm out[-1] norm=\(String(format: "%.4f", n)) head=\(head)\n".data(using: .utf8)!)
                }
                let embeds = addType(baseHidden, type: typeText, rows: window.count)
                let ttsOut = try await ttsLM.forward(embeds: embeds)         // w × hidden
                ttsHidden = lastRow(ttsOut, hidden: constants.hiddenSize)
                if await ttsLM.processed > maxLen { break }
            } else if start >= textIDs.count {
                // Text exhausted: keep emitting speech until EOS or max length.
            }

            for _ in 0..<constants.speechWindow {
                if debug && scaledLatents.count < 3 {
                    func stat(_ v: [Float]) -> String {
                        let n = sqrt(v.reduce(0) { $0 + $1 * $1 })
                        let head = v.prefix(6).map { String(format: "%.4f", $0) }
                        return "norm=\(String(format: "%.4f", n)) head=\(head)"
                    }
                    FileHandle.standardError.write(
                        "  [f\(scaledLatents.count)] cond \(stat(ttsHidden))\n        neg  \(stat(negHidden))\n".data(using: .utf8)!)
                }
                let latent = try await sampleSpeech(
                    positive: ttsHidden, negative: negHidden,
                    cfgScale: options.cfgScale, steps: options.diffusionSteps, rng: &rng)

                // Scaled latent for the VAE decoder; raw latent for the connector.
                var scaled = [Float](repeating: 0, count: latent.count)
                for i in 0..<latent.count { scaled[i] = latent[i] / scaleBias.scale - scaleBias.bias }
                scaledLatents.append(scaled)
                onFrame?(scaledLatents.count)

                let acousticEmbed = try await connector.run([
                    "latent": (shape: [1, 1, constants.acousticVAEDim], data: latent)
                ])  // hidden

                let spTTS = add(acousticEmbed, typeSpeech)
                let ttsOut = try await ttsLM.forward(embeds: spTTS)
                ttsHidden = ttsOut

                let spNeg = add(acousticEmbed, typeSpeech)
                let negOut = try await negTTSLM.forward(embeds: spNeg)
                negHidden = negOut

                let eosLogit = try await eosClassifier.run([
                    "hidden": (shape: [1, constants.hiddenSize], data: ttsHidden)
                ])
                let eosProb = sigmoid(eosLogit.first ?? -10)
                if debug {
                    let textLeft = max(0, textIDs.count - windowIndex * constants.textWindow)
                    FileHandle.standardError.write(
                        "frame \(scaledLatents.count) win \(windowIndex) textLeft \(textLeft) eos \(String(format: "%.4f", eosProb))\n".data(using: .utf8)!)
                }
                if eosProb > 0.5 { finished = true; break }
                if await ttsLM.processed > maxLen { finished = true; break }
            }
        }

        let samples = try await decodeAll(scaledLatents, options: options)
        return AudioResult(samples: samples, sampleRate: audio.sampleRate)
    }

    // MARK: - Diffusion (CFG DPM-Solver loop)

    private func sampleSpeech(
        positive: [Float], negative: [Float],
        cfgScale: Float, steps: Int, rng: inout GaussianRNG
    ) async throws -> [Float] {
        let dim = constants.acousticVAEDim
        let solver = DPMSolverMultistep()
        solver.setTimesteps(steps)

        var x = (0..<dim).map { _ in Float(rng.next()) }  // randn, init_noise_sigma = 1.0
        let condition = positive + negative                // [2 × hidden]

        for t in solver.timesteps {
            let noisy = x + x                               // [2 × dim] (both halves identical)
            let ts: [Float] = [Float(t), Float(t)]
            let eps = try await diffusionHead.run([
                "noisy_latent": (shape: [2, dim], data: noisy),
                "timestep": (shape: [2], data: ts),
                "condition": (shape: [2, constants.hiddenSize], data: condition),
            ])
            var guided = [Float](repeating: 0, count: dim)
            for i in 0..<dim {
                let condEps = eps[i]
                let uncondEps = eps[dim + i]
                guided[i] = uncondEps + cfgScale * (condEps - uncondEps)
            }
            x = solver.step(modelOutput: guided, sample: x)
        }
        return x
    }

    // MARK: - Acoustic decode (fixed-block with crossfade)

    private func decodeAll(_ latents: [[Float]], options: GenerationOptions) async throws -> [Float] {
        let frames = latents.count
        guard frames > 0 else { return [] }
        let dim = constants.acousticVAEDim
        let B = options.decoderFrames
        let O = min(options.decoderOverlap, B / 2)
        let spf = audio.samplesPerFrame
        let step = max(1, B - O)

        var out: [Float] = []
        var startFrame = 0
        while startFrame < frames {
            let endFrame = min(startFrame + B, frames)
            let blockFrames = endFrame - startFrame

            // latents tensor [1, dim, B], channels-first, zero-padded to B.
            var data = [Float](repeating: 0, count: dim * B)
            for f in 0..<blockFrames {
                let frame = latents[startFrame + f]
                for c in 0..<dim { data[c * B + f] = frame[c] }
            }
            let audioBlock = try await decoder.run([
                "latents": (shape: [1, dim, B], data: data)
            ])
            let valid = blockFrames * spf

            if startFrame == 0 {
                out.append(contentsOf: audioBlock[0..<valid])
            } else {
                let cf = O * spf
                let base = out.count - cf
                if base >= 0 {
                    for i in 0..<cf {
                        let w = Float(i) / Float(cf)
                        out[base + i] = out[base + i] * (1 - w) + audioBlock[i] * w
                    }
                }
                if valid > cf { out.append(contentsOf: audioBlock[cf..<valid]) }
            }
            if endFrame == frames { break }
            startFrame += step
        }
        return out
    }

    // MARK: - Small vector helpers

    private func add(_ a: [Float], _ b: [Float]) -> [Float] {
        var out = a
        for i in 0..<min(a.count, b.count) { out[i] = a[i] + b[i] }
        return out
    }

    /// Add a single `type` row to each of `rows` rows of `flat` (rows × hidden).
    private func addType(_ flat: [Float], type: [Float], rows: Int) -> [Float] {
        let h = constants.hiddenSize
        var out = flat
        for r in 0..<rows {
            for i in 0..<h { out[r * h + i] += type[i] }
        }
        return out
    }

    private func lastRow(_ flat: [Float], hidden: Int) -> [Float] {
        let rows = flat.count / hidden
        let start = (rows - 1) * hidden
        return Array(flat[start..<(start + hidden)])
    }

    private func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + exp(-x)) }
}

/// Deterministic Gaussian noise source (SplitMix64 + Box–Muller).
struct GaussianRNG {
    private var state: UInt64
    private var spare: Float?

    init(seed: UInt64) { self.state = seed }

    private mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    private mutating func nextUniform() -> Double {
        Double(nextUInt64() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    mutating func next() -> Double {
        if let s = spare { spare = nil; return Double(s) }
        var u1 = nextUniform()
        let u2 = nextUniform()
        if u1 < 1e-12 { u1 = 1e-12 }
        let mag = (-2.0 * Foundation.log(u1)).squareRoot()
        let z0 = mag * Foundation.cos(2.0 * Double.pi * u2)
        let z1 = mag * Foundation.sin(2.0 * Double.pi * u2)
        spare = Float(z1)
        return z0
    }
}
