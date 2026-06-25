// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
import ArgumentParser
import Foundation
import PocketTTSCoreAI

@main
struct PocketTTSCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pocket-tts",
        abstract: "On-device Pocket-TTS synthesis on Apple Core AI (community port)."
    )

    @Option(name: .shortAndLong, help: "Path to the PocketTTSCoreAI package directory.")
    var package: String

    @Option(name: .shortAndLong, help: "Text to synthesize.")
    var text: String = "Hello world, this is a test of pocket text to speech."

    @Option(name: .shortAndLong, help: "Voice name (alba, marius, javert, jean, fantine, cosette, eponine, azelma).")
    var voice: String = "alba"

    @Option(name: .shortAndLong, help: "Output WAV path.")
    var output: String = "pocket_tts_out.wav"

    @Option(help: "RNG seed for the flow noise.")
    var seed: UInt64 = 0

    mutating func run() async throws {
        let pkgURL = URL(fileURLWithPath: package)
        let tts = try await PocketTTS(packageURL: pkgURL)

        if let err = tts.validateTokenizer() {
            FileHandle.standardError.write(Data(("warning: " + err + "\n").utf8))
        }

        let t0 = Date()
        let pcm = try await tts.synthesize(text: text, voice: voice, seed: seed)
        let dt = Date().timeIntervalSince(t0)
        let secs = Double(pcm.count) / Double(tts.sampleRate)

        let outURL = URL(fileURLWithPath: output)
        try WavWriter.write(pcm, sampleRate: tts.sampleRate, to: outURL)
        let rtf = dt > 0 ? secs / dt : 0
        print(String(format: "wrote %@  (%.2fs audio, voice=%@, %.0f samples, %.1f× realtime)",
                     outURL.path, secs, voice, Double(pcm.count), rtf))
    }
}
