// Thin wrappers around the exported `.aimodel` graphs using the Core AI Swift
// runtime (`AIModel` / `InferenceFunction` / `NDArray`).
//
// Two flavours:
//   * `StatelessComponent` — diffusion head, acoustic connector, EOS classifier,
//     acoustic decoder. Plain inputs → outputs.
//   * `Qwen2Stack` — base_lm / tts_lm. Carries the KV cache as persistent Core AI
//     state, mirroring the offset/position convention baked into the export
//     (`offset = position_ids.length − query.length`).

import CoreAI
import Foundation

// MARK: - NDArray helpers

enum ND {
    /// Build a float NDArray for `name` with the given resolved `shape` and data.
    static func float(_ fn: InferenceFunction, _ name: String, shape: [Int], data: [Float]) -> NDArray {
        guard case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: name) else {
            fatalError("input '\(name)' is not an NDArray")
        }
        let resolved = desc.resolvingDynamicDimensions(shape)
        var array = NDArray(descriptor: resolved)
        write(&array, data)
        return array
    }

    /// Build an Int32 NDArray for `name`.
    static func int(_ fn: InferenceFunction, _ name: String, shape: [Int], data: [Int32]) -> NDArray {
        guard case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: name) else {
            fatalError("input '\(name)' is not an NDArray")
        }
        let resolved = desc.resolvingDynamicDimensions(shape)
        var array = NDArray(descriptor: resolved)
        var view = array.mutableView(as: Int32.self)
        view.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<data.count { ptr[i] = data[i] }
        }
        return array
    }

    static func write(_ array: inout NDArray, _ data: [Float]) {
        switch array.scalarType {
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for i in 0..<data.count { ptr[i] = data[i] }
            }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                for i in 0..<data.count { ptr[i] = Float16(data[i]) }
            }
        #endif
        default:
            fatalError("unsupported NDArray scalar type for float write: \(array.scalarType)")
        }
    }

    static func readFloats(_ array: NDArray) -> [Float] {
        let count = array.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        switch array.scalarType {
        case .float32:
            array.view(as: Float.self).withUnsafePointer { ptr, _, _ in
                for i in 0..<count { out[i] = ptr[i] }
            }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16:
            array.view(as: Float16.self).withUnsafePointer { ptr, _, _ in
                for i in 0..<count { out[i] = Float(ptr[i]) }
            }
        #endif
        default:
            fatalError("unsupported NDArray scalar type for read: \(array.scalarType)")
        }
        return out
    }
}

// MARK: - Stateless component

public actor StatelessComponent {
    private let url: URL
    private let preferredCompute: ComputeUnitKind
    private var model: AIModel?
    private var function: InferenceFunction?

    public init(url: URL, preferredCompute: ComputeUnitKind = .gpu) {
        self.url = url
        self.preferredCompute = preferredCompute
    }

    private func loaded() async throws -> InferenceFunction {
        if let fn = function { return fn }
        let model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: preferredCompute))
        guard let fn = try model.loadFunction(named: "main") else {
            throw VibeVoiceError.runtime("function 'main' not found in \(url.lastPathComponent)")
        }
        self.model = model
        self.function = fn
        return fn
    }

    /// Run with named float inputs; returns the first output flattened.
    public func run(_ floatInputs: [String: (shape: [Int], data: [Float])]) async throws -> [Float] {
        let fn = try await loaded()
        // Reload a fresh function per call for clean stateless buffers.
        guard let freshFn = try model?.loadFunction(named: "main") else {
            throw VibeVoiceError.runtime("could not reload 'main'")
        }
        var inputs: [String: NDArray] = [:]
        for (name, t) in floatInputs {
            inputs[name] = ND.float(freshFn, name, shape: t.shape, data: t.data)
        }
        var outputs = try await freshFn.run(inputs: inputs)
        guard let outName = freshFn.descriptor.outputNames.first,
              let nd = outputs.remove(outName)?.ndArray
        else { return [] }
        _ = fn  // keep primary function alive
        return ND.readFloats(nd)
    }
}

// MARK: - Stateless Qwen2 stack (base_lm / tts_lm) with a Swift-managed KV cache
//
// The exported graph is fully stateless: it takes `past_k`/`past_v` (the cache so
// far) plus a *single* query token/embed (q = 1) and returns the hidden state and
// the freshly computed `new_k`/`new_v`. This sidesteps a Core AI runtime bug that
// miscompiles the stateful, multi-token attention path. Multi-token windows are
// fed one token at a time here (autoregressive attention is order-invariant), and
// the cache is grown in Swift by appending `new_k`/`new_v`.

public actor Qwen2Stack {
    public enum InputKind { case tokenIDs, embeds }

    private let url: URL
    private let kind: InputKind
    private let layers: Int
    private let kvHeads: Int
    private let headDim: Int
    private let hiddenSize: Int
    private let maxContext: Int
    private let preferredCompute: ComputeUnitKind

    private var model: AIModel?
    private var function: InferenceFunction?
    private var inputName = ""
    private var positionName = ""
    private var pastKeyName = ""
    private var pastValueName = ""
    private var hiddenName = ""
    private var newKeyName = ""
    private var newValueName = ""

    // Cache laid out as (layers, kvHeads, length, headDim), flat and contiguous.
    private var keyData: [Float] = []
    private var valData: [Float] = []

    /// Number of positions currently committed to the cache.
    public private(set) var processed = 0

    public init(
        url: URL, kind: InputKind,
        layers: Int, kvHeads: Int, headDim: Int, hiddenSize: Int, maxContext: Int,
        preferredCompute: ComputeUnitKind = .gpu
    ) {
        self.url = url
        self.kind = kind
        self.layers = layers
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.maxContext = maxContext
        self.preferredCompute = preferredCompute
    }

    private func ensureLoaded() async throws -> InferenceFunction {
        if let fn = function { return fn }
        let model = try await AIModel(
            contentsOf: url,
            options: SpecializationOptions(preferredComputeUnitKind: preferredCompute)
        )
        guard let fn = try model.loadFunction(named: "main") else {
            throw VibeVoiceError.runtime("function 'main' not found in \(url.lastPathComponent)")
        }
        let d = fn.descriptor
        // Inputs: (input_ids|inputs_embeds, position_ids, past_k, past_v)
        self.inputName = d.inputNames[0]
        self.positionName = d.inputNames[1]
        self.pastKeyName = d.inputNames[2]
        self.pastValueName = d.inputNames[3]
        // Outputs: (hidden, new_k, new_v)
        self.hiddenName = d.outputNames[0]
        self.newKeyName = d.outputNames[1]
        self.newValueName = d.outputNames[2]
        self.model = model
        self.function = fn
        return fn
    }

    /// Seed the KV cache with a voice prefill (layout layers,1,kv,L,headDim).
    public func seed(with prefill: StackPrefill) async throws {
        _ = try await ensureLoaded()
        // Prefill is already (layer,head,length,headDim) contiguous — adopt it.
        self.keyData = prefill.key
        self.valData = prefill.value
        self.processed = prefill.length
    }

    /// Forward `queryLen` new tokens/embeds one at a time; returns hidden states
    /// (queryLen × hidden, row-major). `tokenIDs` is used when `kind == .tokenIDs`,
    /// `embeds` (row-major queryLen × hidden) when `kind == .embeds`.
    public func forward(tokenIDs: [Int32]? = nil, embeds: [Float]? = nil) async throws -> [Float] {
        _ = try await ensureLoaded()
        guard !keyData.isEmpty else {
            throw VibeVoiceError.runtime("KV cache not seeded; call seed(with:) first")
        }
        let queryLen: Int
        switch kind {
        case .tokenIDs: queryLen = tokenIDs?.count ?? 0
        case .embeds: queryLen = (embeds?.count ?? 0) / hiddenSize
        }
        guard queryLen > 0 else { return [] }

        guard let fn = function else { throw VibeVoiceError.runtime("stack not loaded") }
        var hidden = [Float](repeating: 0, count: queryLen * hiddenSize)
        for t in 0..<queryLen {
            let single: NDArray
            switch kind {
            case .tokenIDs:
                let id = tokenIDs![t]
                single = ND.int(fn, inputName, shape: [1, 1], data: [id])
            case .embeds:
                let row = Array(embeds![(t * hiddenSize)..<((t + 1) * hiddenSize)])
                single = ND.float(fn, inputName, shape: [1, 1, hiddenSize], data: row)
            }
            let h = try await step(input: single, position: Int32(processed))
            for d in 0..<hiddenSize { hidden[t * hiddenSize + d] = h[d] }
        }
        return hidden
    }

    /// Run a single q=1 step: build past_k/past_v from the current cache, execute
    /// the graph, append new_k/new_v, and return the hidden row.
    private func step(input: NDArray, position: Int32) async throws -> [Float] {
        guard let model = model, let fn = function else {
            throw VibeVoiceError.runtime("stack not loaded")
        }
        // Fresh function per call keeps the stateless graph's buffers clean.
        guard let freshFn = try model.loadFunction(named: "main") else {
            throw VibeVoiceError.runtime("could not reload 'main'")
        }
        let L = processed
        let pastShape = [layers, 1, kvHeads, L, headDim]
        let pastK = ND.float(freshFn, pastKeyName, shape: pastShape, data: keyData)
        let pastV = ND.float(freshFn, pastValueName, shape: pastShape, data: valData)
        let posArray = ND.int(freshFn, positionName, shape: [1, 1], data: [position])

        var outputs = try await freshFn.run(inputs: [
            inputName: input,
            positionName: posArray,
            pastKeyName: pastK,
            pastValueName: pastV,
        ])
        guard let hiddenND = outputs.remove(hiddenName)?.ndArray,
              let newKND = outputs.remove(newKeyName)?.ndArray,
              let newVND = outputs.remove(newValueName)?.ndArray
        else { throw VibeVoiceError.runtime("missing stack outputs") }

        appendCache(newKey: ND.readFloats(newKND), newValue: ND.readFloats(newVND))
        processed += 1
        _ = fn  // keep primary function alive
        return ND.readFloats(hiddenND)
    }

    /// Append one position to the (layers,kvHeads,length,headDim) cache. `newKey`
    /// and `newValue` are (layers,kvHeads,headDim) flat for the single new position.
    private func appendCache(newKey: [Float], newValue: [Float]) {
        let L = processed
        let newLen = L + 1
        let total = layers * kvHeads * newLen * headDim
        var nk = [Float](repeating: 0, count: total)
        var nv = [Float](repeating: 0, count: total)
        for lh in 0..<(layers * kvHeads) {
            let oldBase = lh * L * headDim
            let newBase = lh * newLen * headDim
            if L > 0 {
                for i in 0..<(L * headDim) {
                    nk[newBase + i] = keyData[oldBase + i]
                    nv[newBase + i] = valData[oldBase + i]
                }
            }
            let tailBase = newBase + L * headDim
            let srcBase = lh * headDim
            for d in 0..<headDim {
                nk[tailBase + d] = newKey[srcBase + d]
                nv[tailBase + d] = newValue[srcBase + d]
            }
        }
        keyData = nk
        valData = nv
    }
}
