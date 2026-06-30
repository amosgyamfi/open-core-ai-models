// Minimal .safetensors reader.
//
// The format is intentionally simple:
//   [8 bytes little-endian header length N]
//   [N bytes UTF-8 JSON header: { name: {dtype, shape, data_offsets:[a,b]}, ... , "__metadata__": {...} }]
//   [raw tensor bytes]
//
// We only need to read F16 / F32 tensors (voice prefills are stored as F16) and
// the string metadata block, so this loader supports exactly those.

import Foundation

/// A single tensor view into a memory-mapped safetensors file, decoded to `[Float]`.
public struct STTensor: Sendable {
    public let shape: [Int]
    public let data: [Float]

    public var count: Int { data.count }
}

public struct Safetensors: Sendable {
    public let tensors: [String: STTensor]
    public let metadata: [String: String]

    enum STError: Error, CustomStringConvertible {
        case tooSmall
        case badHeader
        case unsupportedDType(String)
        case missing(String)

        var description: String {
            switch self {
            case .tooSmall: return "safetensors file too small"
            case .badHeader: return "safetensors header is not valid JSON"
            case .unsupportedDType(let d): return "unsupported safetensors dtype '\(d)' (expected F16/F32)"
            case .missing(let n): return "tensor '\(n)' not found in safetensors file"
            }
        }
    }

    public init(contentsOf url: URL) throws {
        let raw = try Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(data: raw)
    }

    public init(data raw: Data) throws {
        guard raw.count >= 8 else { throw STError.tooSmall }
        let headerLen = raw.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerLen)
        guard raw.count >= headerEnd else { throw STError.tooSmall }

        let headerData = raw.subdata(in: headerStart..<headerEnd)
        guard let obj = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw STError.badHeader
        }

        var meta: [String: String] = [:]
        var parsed: [String: STTensor] = [:]
        let base = headerEnd

        for (name, value) in obj {
            if name == "__metadata__" {
                if let m = value as? [String: String] { meta = m }
                continue
            }
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shapeAny = entry["shape"] as? [Any],
                  let offsets = entry["data_offsets"] as? [Any],
                  offsets.count == 2
            else { throw STError.badHeader }

            let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            let a = (offsets[0] as? NSNumber)?.intValue ?? 0
            let b = (offsets[1] as? NSNumber)?.intValue ?? 0
            let count = shape.reduce(1, *)
            let slice = raw.subdata(in: (base + a)..<(base + b))

            let floats: [Float]
            switch dtype {
            case "F32":
                floats = slice.withUnsafeBytes { buf in
                    let p = buf.bindMemory(to: Float32.self)
                    return Array(p.prefix(count)).map { Float($0) }
                }
            case "F16":
                floats = slice.withUnsafeBytes { buf -> [Float] in
                    let p = buf.bindMemory(to: UInt16.self)
                    var out = [Float](repeating: 0, count: count)
                    for i in 0..<count { out[i] = Float(Float16(bitPattern: p[i])) }
                    return out
                }
            case "BF16":
                floats = slice.withUnsafeBytes { buf -> [Float] in
                    let p = buf.bindMemory(to: UInt16.self)
                    var out = [Float](repeating: 0, count: count)
                    for i in 0..<count {
                        // bfloat16 = top 16 bits of a float32
                        out[i] = Float(bitPattern: UInt32(p[i]) << 16)
                    }
                    return out
                }
            default:
                throw STError.unsupportedDType(dtype)
            }
            parsed[name] = STTensor(shape: shape, data: floats)
        }

        self.tensors = parsed
        self.metadata = meta
    }

    public func require(_ name: String) throws -> STTensor {
        guard let t = tensors[name] else { throw STError.missing(name) }
        return t
    }
}
