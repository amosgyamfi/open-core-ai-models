import AVFoundation
import CoreAIVibeVoice
import SwiftUI

@MainActor
final class TTSModel: ObservableObject {
    @Published var voices: [VoiceInfo] = []
    @Published var languages: [String] = []
    @Published var selectedLanguage = "English"
    @Published var selectedVoice = "en-Carter_man"
    @Published var text = "Hello! This speech is generated entirely on device with Core AI."
    @Published var cfgScale: Double = 1.5
    @Published var steps: Double = 5
    @Published var isLoading = false
    @Published var isGenerating = false
    @Published var status = "Loading model…"
    @Published var lastDuration: Double = 0
    @Published var lastRTF: Double = 0

    private var engine: VibeVoiceEngine?
    private let player = AVAudioEnginePlayer()

    var voicesForSelectedLanguage: [VoiceInfo] {
        voices.filter { $0.language == selectedLanguage }.sorted { $0.name < $1.name }
    }

    /// Locate the asset bundle "VibeVoiceAssets" embedded as a folder reference.
    private func assetsURL() -> URL? {
        if let url = Bundle.main.url(forResource: "VibeVoiceAssets", withExtension: nil) {
            return url
        }
        return nil
    }

    func load() async {
        guard let root = assetsURL() else {
            status = "VibeVoiceAssets not found in app bundle."
            return
        }
        do {
            isLoading = true
            let engine = try await VibeVoiceEngine(assets: VibeVoiceAssets(root: root))
            self.engine = engine
            self.voices = await engine.availableVoices()
            self.languages = await engine.languages()
            if !languages.contains(selectedLanguage) { selectedLanguage = languages.first ?? "English" }
            if let first = voicesForSelectedLanguage.first { selectedVoice = first.name }
            status = "Ready — \(voices.count) voices, \(languages.count) languages."
        } catch {
            status = "Load failed: \(error)"
        }
        isLoading = false
    }

    func generate() async {
        guard let engine else { return }
        isGenerating = true
        status = "Generating…"
        do {
            var options = GenerationOptions()
            options.cfgScale = Float(cfgScale)
            options.diffusionSteps = Int(steps)
            let start = Date()
            let result = try await engine.synthesize(text: text, voice: selectedVoice, options: options)
            let elapsed = Date().timeIntervalSince(start)
            lastDuration = result.duration
            lastRTF = elapsed / max(result.duration, 0.0001)
            status = String(format: "Done — %.1fs audio (RTF %.2fx)", result.duration, lastRTF)
            player.play(samples: result.samples, sampleRate: Double(result.sampleRate))
        } catch {
            status = "Generation failed: \(error)"
        }
        isGenerating = false
    }
}

struct ContentView: View {
    @StateObject private var model = TTSModel()

    var body: some View {
        Form {
            Section("Voice") {
                Picker("Language", selection: $model.selectedLanguage) {
                    ForEach(model.languages, id: \.self) { Text($0).tag($0) }
                }
                .onChange(of: model.selectedLanguage) { _, _ in
                    if let first = model.voicesForSelectedLanguage.first {
                        model.selectedVoice = first.name
                    }
                }
                Picker("Speaker", selection: $model.selectedVoice) {
                    ForEach(model.voicesForSelectedLanguage) { v in
                        Text("\(v.name) · \(v.gender)").tag(v.name)
                    }
                }
            }

            Section("Text") {
                TextEditor(text: $model.text)
                    .frame(minHeight: 120)
                    .font(.body)
            }

            Section("Settings") {
                VStack(alignment: .leading) {
                    Text("CFG scale: \(model.cfgScale, specifier: "%.1f")")
                    Slider(value: $model.cfgScale, in: 1.0...3.0, step: 0.1)
                }
                VStack(alignment: .leading) {
                    Text("Diffusion steps: \(Int(model.steps))")
                    Slider(value: $model.steps, in: 2...20, step: 1)
                }
            }

            Section {
                Button {
                    Task { await model.generate() }
                } label: {
                    HStack {
                        if model.isGenerating { ProgressView() }
                        Text(model.isGenerating ? "Generating…" : "Generate & Play")
                    }
                    .frame(maxWidth: .infinity)
                }
                .disabled(model.isLoading || model.isGenerating || model.text.isEmpty)
            } footer: {
                Text(model.status).font(.footnote).foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Core AI VibeVoice")
        .task { await model.load() }
    }
}

/// Tiny AVAudioEngine-based player for float PCM.
final class AVAudioEnginePlayer {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()

    init() {
        engine.attach(node)
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)!
        engine.connect(node, to: engine.mainMixerNode, format: format)
    }

    func play(samples: [Float], sampleRate: Double) {
        guard let buffer = AudioUtilities.pcmBuffer(samples: samples, sampleRate: sampleRate) else { return }
        do {
            if !engine.isRunning { try engine.start() }
            node.stop()
            node.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
            node.play()
        } catch {
            print("playback error: \(error)")
        }
    }
}
