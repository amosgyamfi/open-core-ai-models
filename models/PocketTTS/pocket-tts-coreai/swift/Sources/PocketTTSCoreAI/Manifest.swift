// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
import Foundation

/// Decoded `manifest.json` from the `PocketTTSCoreAI` package produced by `package.py`.
public struct Manifest: Codable, Sendable {
    public struct TensorRef: Codable, Sendable { public let file: String; public let shape: [Int] }
    public struct Scalars: Codable, Sendable {
        public let ldim, num_layers, num_heads, dim_per_head, cache_len, n_bins: Int
        public let temp, eos_threshold: Double
    }
    public struct Voice: Codable, Sendable {
        public let offset: Int; public let key: TensorRef; public let value: TensorRef
    }

    public let name: String
    public let dtype: String
    public let sample_rate: Int
    public let frame_samples: Int
    public let scalars: Scalars
    public let bundles: [String: String]
    public let glue: [String: TensorRef]
    public let voices: [String: Voice]
    public let tokenizer: String
    public let tokenizer_json: String
}

/// Loads the package directory: manifest + flat float32 `.bin` tensors.
public struct ModelPackage: Sendable {
    public let root: URL
    public let manifest: Manifest

    public init(root: URL) throws {
        self.root = root
        let data = try Data(contentsOf: root.appendingPathComponent("manifest.json"))
        self.manifest = try JSONDecoder().decode(Manifest.self, from: data)
    }

    public func url(_ rel: String) -> URL { root.appendingPathComponent(rel) }

    /// Read a `.bin` (raw little-endian float32, row-major) into a flat `[Float]`.
    public func floats(_ ref: Manifest.TensorRef) throws -> [Float] {
        let data = try Data(contentsOf: url(ref.file))
        let count = ref.shape.reduce(1, *)
        precondition(data.count == count * MemoryLayout<Float>.size,
                     "size mismatch for \(ref.file): \(data.count) bytes vs \(count) floats")
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self).prefix(count)) }
    }

    public func glue(_ key: String) throws -> [Float] {
        guard let ref = manifest.glue[key] else { throw PocketError.missing("glue.\(key)") }
        return try floats(ref)
    }
}

public enum PocketError: Error, CustomStringConvertible {
    case missing(String), badShape(String), engine(String)
    public var description: String {
        switch self {
        case .missing(let s): return "missing resource: \(s)"
        case .badShape(let s): return "bad shape: \(s)"
        case .engine(let s): return "engine error: \(s)"
        }
    }
}
