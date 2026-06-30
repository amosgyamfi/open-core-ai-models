// swift-tools-version: 6.0
//
// CoreAIVibeVoice — an on-device, Core AI–powered text-to-speech runtime for the
// converted microsoft/VibeVoice-Realtime-0.5B model. Targets Apple silicon on
// iOS 27+ / macOS 27+ where the system `CoreAI` framework is available.

import PackageDescription

let package = Package(
    name: "CoreAIVibeVoice",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "CoreAIVibeVoice", targets: ["CoreAIVibeVoice"]),
        .executable(name: "vibevoice-cli", targets: ["vibevoice-cli"]),
    ],
    dependencies: [
        // Qwen2 BPE tokenizer (reads tokenizer.json) + argument parsing.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "CoreAIVibeVoice",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            path: "Sources/CoreAIVibeVoice",
            swiftSettings: [
                .enableUpcomingFeature("MemberImportVisibility")
            ]
        ),
        .executableTarget(
            name: "vibevoice-cli",
            dependencies: [
                "CoreAIVibeVoice",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/vibevoice-cli"
        ),
    ]
)
