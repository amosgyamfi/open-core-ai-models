// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
import Foundation

/// Minimal 16-bit PCM mono WAV writer (no AVFoundation dependency).
public enum WavWriter {
    public static func write(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        let n = samples.count
        var data = Data(capacity: 44 + n * 2)
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        data.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + n * 2))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(UInt32(n * 2))
        for s in samples {
            let c = max(-1, min(1, s))
            u16(UInt16(bitPattern: Int16(c * 32767)))
        }
        try data.write(to: url)
    }
}
