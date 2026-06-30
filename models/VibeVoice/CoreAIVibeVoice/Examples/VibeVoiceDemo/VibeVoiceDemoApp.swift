// VibeVoiceDemo — a minimal SwiftUI app showcasing the Core AI VibeVoice TTS
// engine. Drop these two files into an iOS/macOS App target (Xcode 27+,
// deployment target 27.0) that depends on the local `CoreAIVibeVoice` package,
// add the runtime asset bundle (the exports/ directory) to the app bundle as a
// folder reference named "VibeVoiceAssets", and run.

import SwiftUI

@main
struct VibeVoiceDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .defaultSize(width: 560, height: 640)
        #endif
    }
}
