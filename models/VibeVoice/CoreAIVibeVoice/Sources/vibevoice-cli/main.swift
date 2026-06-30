// Command-line runner for the Core AI VibeVoice pipeline (macOS 27+).
//
//   swift run vibevoice-cli --assets <bundle> --voice en-Carter_man \
//       --text "Hello from on-device Core AI." --output out.wav
//
//   swift run vibevoice-cli --assets <bundle> --list-voices

import ArgumentParser
import CoreAIVibeVoice
import Foundation

@main
struct VibeVoiceCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vibevoice-cli",
        abstract: "On-device text-to-speech with the Core AI VibeVoice model."
    )

    @Option(name: .long, help: "Path to the runtime asset bundle (the exports/ directory).")
    var assets: String

    @Option(name: .long, help: "Voice name, e.g. en-Carter_man.")
    var voice: String = "en-Carter_man"

    @Option(name: .long, help: "Text to speak (or use --text-file).")
    var text: String?

    @Option(name: .long, help: "Read the script from a text file.")
    var textFile: String?

    @Option(name: .long, help: "Output WAV path.")
    var output: String = "vibevoice_out.wav"

    @Option(name: .long, help: "Classifier-free guidance scale.")
    var cfgScale: Float = 1.5

    @Option(name: .long, help: "Diffusion (DPM-Solver) steps per frame.")
    var steps: Int = 5

    @Option(name: .long, help: "Random seed for reproducible output.")
    var seed: UInt64?

    @Flag(name: .long, help: "List available voices and languages, then exit.")
    var listVoices = false

    mutating func run() async throws {
        let assetsURL = URL(fileURLWithPath: (assets as NSString).expandingTildeInPath)
        let engine = try await VibeVoiceEngine(assets: VibeVoiceAssets(root: assetsURL))

        if listVoices {
            let voices = await engine.availableVoices()
            let langs = await engine.languages()
            print("Languages (\(langs.count)): \(langs.joined(separator: ", "))")
            print("Voices (\(voices.count)):")
            for v in voices.sorted(by: { ($0.language, $0.name) < ($1.language, $1.name) }) {
                print("  \(v.name)  [\(v.language), \(v.gender)]")
            }
            return
        }

        var script = text ?? ""
        if let file = textFile {
            script = try String(contentsOf: URL(fileURLWithPath: file), encoding: .utf8)
        }
        guard !script.isEmpty else {
            throw ValidationError("Provide --text or --text-file.")
        }

        var options = GenerationOptions()
        options.cfgScale = cfgScale
        options.diffusionSteps = steps
        options.seed = seed

        print("Synthesizing with voice '\(voice)' (cfg=\(cfgScale), steps=\(steps))…")
        let started = Date()
        let result = try await engine.synthesize(text: script, voice: voice, options: options) { frames in
            if frames % 30 == 0 { FileHandle.standardError.write(Data("  \(frames) frames\n".utf8)) }
        }
        let elapsed = Date().timeIntervalSince(started)

        let outURL = URL(fileURLWithPath: (output as NSString).expandingTildeInPath)
        try AudioUtilities.writeWAV(samples: result.samples, sampleRate: result.sampleRate, to: outURL)

        let rtf = elapsed / max(result.duration, 0.0001)
        print(String(format: "Done: %.2fs audio in %.2fs (RTF %.2fx) → %@",
                     result.duration, elapsed, rtf, outURL.path))
    }
}
