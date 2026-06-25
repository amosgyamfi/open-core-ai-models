// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
import Foundation
import Testing
@testable import PocketTTSCoreAI

/// Point this at a built package dir to run on-device parity checks:
///   POCKETTTS_PACKAGE=/path/to/dist/PocketTTSCoreAI swift test
private func packageURL() -> URL? {
    ProcessInfo.processInfo.environment["POCKETTTS_PACKAGE"].map { URL(fileURLWithPath: $0) }
}

@Test func manifestDecodes() throws {
    guard let root = packageURL() else { return }
    let pkg = try ModelPackage(root: root)
    #expect(pkg.manifest.sample_rate == 24000)
    #expect(pkg.manifest.frame_samples == 1920)
    #expect(pkg.manifest.scalars.ldim == 32)
    #expect(pkg.manifest.bundles["backbone"] != nil)
    #expect(!pkg.manifest.voices.isEmpty)
}

@Test func tokenizerMatchesSentencePiece() throws {
    guard let root = packageURL() else { return }
    let pkg = try ModelPackage(root: root)
    let tok = try UnigramTokenizer(jsonURL: pkg.url(pkg.manifest.tokenizer_json))
    // The embedded self-test vectors were produced by the reference SentencePiece encoder.
    #expect(tok.runSelfTest() == nil)
}
