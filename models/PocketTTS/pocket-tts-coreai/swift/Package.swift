// swift-tools-version: 6.0
// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
import PackageDescription

let package = Package(
    name: "PocketTTSCoreAI",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "PocketTTSCoreAI", targets: ["PocketTTSCoreAI"]),
        .executable(name: "pocket-tts", targets: ["pocket-tts-cli"]),
    ],
    dependencies: [
        // CoreAIShared gives us `PreparedModel.prepare(at:)` (single-graph GPU specialization,
        // the path the SAM3/Whisper sample apps use) without the heavy CoreAILM stack.
        .package(url: "https://github.com/apple/coreai-models", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "PocketTTSCoreAI",
            dependencies: [
                .product(name: "CoreAISegmentation", package: "coreai-models"),  // pulls CoreAIShared + CoreAI
            ],
            swiftSettings: [.enableUpcomingFeature("MemberImportVisibility")]
        ),
        .executableTarget(
            name: "pocket-tts-cli",
            dependencies: [
                "PocketTTSCoreAI",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [.enableUpcomingFeature("MemberImportVisibility")]
        ),
        .testTarget(name: "PocketTTSCoreAITests", dependencies: ["PocketTTSCoreAI"]),
    ],
    swiftLanguageModes: [.v6]
)
