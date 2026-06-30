// Audio helpers: 16-bit PCM WAV writer and a Float→AVAudioPCMBuffer bridge.

import Foundation
#if canImport(AVFoundation)
import AVFoundation
#endif

public enum AudioUtilities {
    /// Write mono float samples (range ~[-1, 1]) as a 16-bit PCM WAV file.
    public static func writeWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
        let numSamples = samples.count
        let bytesPerSample = 2
        let dataBytes = numSamples * bytesPerSample
        let byteRate = sampleRate * bytesPerSample

        var data = Data(capacity: 44 + dataBytes)
        func appendU32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func appendU16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }

        data.append(contentsOf: Array("RIFF".utf8))
        appendU32(UInt32(36 + dataBytes))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        appendU32(16)                       // PCM fmt chunk size
        appendU16(1)                        // PCM
        appendU16(1)                        // mono
        appendU32(UInt32(sampleRate))
        appendU32(UInt32(byteRate))
        appendU16(UInt16(bytesPerSample))   // block align
        appendU16(16)                       // bits per sample
        data.append(contentsOf: Array("data".utf8))
        appendU32(UInt32(dataBytes))

        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            appendU16(UInt16(bitPattern: Int16(clamped * 32767.0)))
        }
        try data.write(to: url)
    }

    #if canImport(AVFoundation)
    /// Wrap mono float samples in an AVAudioPCMBuffer for live playback.
    public static func pcmBuffer(samples: [Float], sampleRate: Double) -> AVAudioPCMBuffer? {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
        else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let ch = buffer.floatChannelData {
            for i in 0..<samples.count { ch[0][i] = samples[i] }
        }
        return buffer
    }
    #endif
}
