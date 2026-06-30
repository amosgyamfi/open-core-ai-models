// Small runtime resources dumped by the exporter (type embeddings, scale/bias,
// audio config) plus model dimensions and special token ids.

import Foundation

/// Errors thrown across the runtime.
public enum VibeVoiceError: Error, CustomStringConvertible {
    case assetMissing(String)
    case voiceNotFound(String)
    case shapeMismatch(String)
    case runtime(String)

    public var description: String {
        switch self {
        case .assetMissing(let s): return "required asset missing: \(s)"
        case .voiceNotFound(let s): return "voice not found: \(s)"
        case .shapeMismatch(let s): return "shape mismatch: \(s)"
        case .runtime(let s): return "runtime error: \(s)"
        }
    }
}

/// Fixed VibeVoice-Realtime-0.5B dimensions and special token ids.
///
/// The special ids reuse Qwen2.5 vision tokens (see
/// `modular_vibevoice_text_tokenizer.py`): they are constant across the vocab.
public struct ModelConstants: Sendable {
    public var hiddenSize = 896
    public var acousticVAEDim = 64
    public var baseLayers = 4
    public var ttsLayers = 20
    public var kvHeads = 2
    public var headDim = 64

    public var speechStartID = 151_652   // <|vision_start|>
    public var speechEndID = 151_653     // <|vision_end|>
    public var speechDiffusionID = 151_654 // <|vision_pad|>
    public var padID = 151_655           // <|image_pad|>
    public var negTextID = 151_655       // negative prompt token (<|image_pad|>)
    public var eosID = 151_643           // <|endoftext|>

    public var textWindow = 5            // TTS_TEXT_WINDOW_SIZE
    public var speechWindow = 6          // TTS_SPEECH_WINDOW_SIZE

    public init() {}
}

/// `tts_input_types` embedding: 2 rows of `hiddenSize` (row 1 = text, row 0 = speech).
public struct TypeEmbedding: Sendable {
    public let rows: [[Float]]   // [2][hidden]

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        struct Blob: Decodable { let shape: [Int]; let data: [Float] }
        let blob = try JSONDecoder().decode(Blob.self, from: data)
        let h = blob.shape[1]
        self.rows = [Array(blob.data[0..<h]), Array(blob.data[h..<(2 * h)])]
    }

    public func row(_ i: Int) -> [Float] { rows[i] }
}

/// Acoustic latent scaling / bias factors used before VAE decoding.
public struct ScaleBias: Sendable {
    public let scale: Float
    public let bias: Float

    public init(contentsOf url: URL) throws {
        struct Blob: Decodable {
            let speech_scaling_factor: Float
            let speech_bias_factor: Float
        }
        let blob = try JSONDecoder().decode(Blob.self, from: try Data(contentsOf: url))
        self.scale = blob.speech_scaling_factor
        self.bias = blob.speech_bias_factor
    }
}

/// Audio output configuration.
public struct AudioConfig: Sendable {
    public let sampleRate: Int
    public let samplesPerFrame: Int
    public let acousticVAEDim: Int

    public init(contentsOf url: URL) throws {
        struct Blob: Decodable {
            let sample_rate: Int
            let samples_per_frame: Int
            let acoustic_vae_dim: Int
        }
        let blob = try JSONDecoder().decode(Blob.self, from: try Data(contentsOf: url))
        self.sampleRate = blob.sample_rate
        self.samplesPerFrame = blob.samples_per_frame
        self.acousticVAEDim = blob.acoustic_vae_dim
    }

    public init(sampleRate: Int = 24_000, samplesPerFrame: Int = 3200, acousticVAEDim: Int = 64) {
        self.sampleRate = sampleRate
        self.samplesPerFrame = samplesPerFrame
        self.acousticVAEDim = acousticVAEDim
    }
}
