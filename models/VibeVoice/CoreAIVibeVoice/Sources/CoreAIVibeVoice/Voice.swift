// Voice catalog + voice prefill bundles.
//
// A "voice" is a cached prompt: the prefilled KV caches and final hidden states
// of the four Qwen2 stacks (lm / tts_lm / neg_lm / neg_tts_lm) for a speaker's
// reference prompt. The same bundle is reused for any text; the text is streamed
// in afterwards. Bundles are produced by `export/convert_voices.py` and stored as
// `<name>.safetensors` (fp16) alongside a `voices.json` catalog.

import Foundation

/// Catalog entry describing an available voice without loading its weights.
public struct VoiceInfo: Codable, Sendable, Identifiable, Hashable {
    public let name: String
    public let language: String
    public let gender: String
    public let file: String

    public var id: String { name }
}

/// The prefill state for one Qwen2 stack: stacked KV cache + final hidden states.
public struct StackPrefill: Sendable {
    /// Flattened key cache, layout (layers, 1, kvHeads, length, headDim).
    public let key: [Float]
    /// Flattened value cache, same layout as `key`.
    public let value: [Float]
    /// Final hidden states, layout (1, length, hidden).
    public let hidden: [Float]
    public let layers: Int
    public let kvHeads: Int
    public let length: Int
    public let headDim: Int
    public let hiddenSize: Int

    /// The last row of `hidden` (used as a diffusion condition), length `hiddenSize`.
    public var lastHidden: [Float] {
        let start = (length - 1) * hiddenSize
        return Array(hidden[start..<(start + hiddenSize)])
    }
}

/// A fully-loaded voice: prefills for all four stacks plus metadata.
public struct Voice: Sendable {
    public let info: VoiceInfo
    public let lm: StackPrefill
    public let ttsLM: StackPrefill
    public let negLM: StackPrefill
    public let negTTSLM: StackPrefill

    static func loadStack(_ st: Safetensors, _ prefix: String) throws -> StackPrefill {
        let k = try st.require("\(prefix).k")   // (layers,1,kv,L,headDim)
        let v = try st.require("\(prefix).v")
        let h = try st.require("\(prefix).h")   // (1,L,hidden)
        let layers = k.shape[0]
        let kvHeads = k.shape[2]
        let length = k.shape[3]
        let headDim = k.shape[4]
        let hiddenSize = h.shape[2]
        return StackPrefill(
            key: k.data, value: v.data, hidden: h.data,
            layers: layers, kvHeads: kvHeads, length: length,
            headDim: headDim, hiddenSize: hiddenSize
        )
    }

    public init(info: VoiceInfo, fileURL: URL) throws {
        let st = try Safetensors(contentsOf: fileURL)
        self.info = info
        self.lm = try Voice.loadStack(st, "lm")
        self.ttsLM = try Voice.loadStack(st, "tts_lm")
        self.negLM = try Voice.loadStack(st, "neg_lm")
        self.negTTSLM = try Voice.loadStack(st, "neg_tts_lm")
    }
}

/// Discovers and loads voices from a directory containing `voices.json` + `*.safetensors`.
public struct VoiceCatalog: Sendable {
    public let directory: URL
    public let voices: [VoiceInfo]

    public init(directory: URL) throws {
        self.directory = directory
        let catalogURL = directory.appending(path: "voices.json")
        if FileManager.default.fileExists(atPath: catalogURL.path) {
            let data = try Data(contentsOf: catalogURL)
            self.voices = try JSONDecoder().decode([VoiceInfo].self, from: data)
        } else {
            // Fall back to scanning *.safetensors and inferring metadata from names.
            let files = (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
            self.voices = files
                .filter { $0.pathExtension == "safetensors" }
                .map { url in
                    let name = url.deletingPathExtension().lastPathComponent
                    return VoiceInfo(name: name, language: "unknown", gender: "unknown", file: url.lastPathComponent)
                }
                .sorted { $0.name < $1.name }
        }
    }

    /// All languages represented in the catalog, sorted.
    public var languages: [String] {
        Array(Set(voices.map(\.language))).sorted()
    }

    public func info(named name: String) -> VoiceInfo? {
        let lower = name.lowercased()
        if let exact = voices.first(where: { $0.name.lowercased() == lower }) { return exact }
        return voices.first(where: { $0.name.lowercased().contains(lower) })
    }

    public func load(_ info: VoiceInfo) throws -> Voice {
        try Voice(info: info, fileURL: directory.appending(path: info.file))
    }

    public func load(named name: String) throws -> Voice {
        guard let info = info(named: name) else {
            throw VibeVoiceError.voiceNotFound(name)
        }
        return try load(info)
    }
}
