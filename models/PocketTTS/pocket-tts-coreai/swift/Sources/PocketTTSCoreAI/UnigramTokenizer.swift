// Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
import Foundation

/// SentencePiece **Unigram** tokenizer with byte-fallback, reimplemented in pure Swift from the
/// exported `tokenizer.json` (pieces + log-prob scores). Mirrors the pocket-tts default config:
/// identity normalization, `add_dummy_prefix`, `escape_whitespaces` (space → ▁ U+2581),
/// `remove_extra_whitespaces = false`. Validated at runtime against the embedded `selftest` vectors.
public struct UnigramTokenizer: Sendable {
    public struct TokJSON: Codable, Sendable {
        public struct Piece: Codable, Sendable { let piece: String; let score: Double; let type: Int }
        public struct Test: Codable, Sendable { let text: String; let ids: [Int] }
        let type: String
        let unk_id: Int
        let byte_fallback: Bool
        let add_dummy_prefix: Bool
        let remove_extra_whitespaces: Bool
        let space: String
        let pieces: [Piece]
        let selftest: [Test]
    }

    private let unkId: Int
    private let byteFallback: Bool
    private let addDummyPrefix: Bool
    private let space: Character
    private let vocab: [String: (id: Int, score: Float)]   // NORMAL/USER pieces only
    private let byteId: [Int]                               // byte value 0..255 -> piece id (or -1)
    private let unkScore: Float
    private let maxPieceScalars: Int
    public let selftest: [TokJSON.Test]

    public init(jsonURL: URL) throws {
        let cfg = try JSONDecoder().decode(TokJSON.self, from: Data(contentsOf: jsonURL))
        self.unkId = cfg.unk_id
        self.byteFallback = cfg.byte_fallback
        self.addDummyPrefix = cfg.add_dummy_prefix
        self.space = cfg.space.first ?? "\u{2581}"
        self.selftest = cfg.selftest

        var v = [String: (Int, Float)]()
        var bytes = [Int](repeating: -1, count: 256)
        var minScore = Float.greatestFiniteMagnitude
        var maxLen = 1
        for (id, p) in cfg.pieces.enumerated() {
            let s = Float(p.score)
            switch p.type {
            case 1, 4:   // NORMAL, USER_DEFINED — match literally in the lattice
                v[p.piece] = (id, s)
                maxLen = max(maxLen, p.piece.unicodeScalars.count)
                minScore = min(minScore, s)
            case 6:      // BYTE piece "<0xHH>"
                if let b = Self.byteValue(p.piece) { bytes[b] = id }
            default: break   // UNKNOWN / CONTROL / UNUSED
            }
        }
        self.vocab = v
        self.byteId = bytes
        self.unkScore = minScore - 10.0   // SentencePiece unk penalty
        self.maxPieceScalars = maxLen
    }

    /// Encode text to token ids (SentencePiece Viterbi). Mirrors `sp.encode(text, out_type=int)`.
    public func encode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        // ---- normalize: prepend dummy prefix space (unconditional), then escape whitespace ----
        let raw = addDummyPrefix ? " " + text : text
        let norm = String(raw.map { $0 == " " ? space : $0 })
        let scalars = Array(norm.unicodeScalars)
        let n = scalars.count
        if n == 0 { return [] }

        // ---- Viterbi over the unigram lattice ----
        var best = [Float](repeating: -Float.greatestFiniteMagnitude, count: n + 1)
        var backId = [Int](repeating: -1, count: n + 1)     // piece id (>=0) or -1 for unk edge
        var backStart = [Int](repeating: -1, count: n + 1)
        best[0] = 0

        for i in 0..<n where best[i] > -Float.greatestFiniteMagnitude {
            let maxJ = min(n, i + maxPieceScalars)
            var matchedAny = false
            var j = i + 1
            while j <= maxJ {
                let sub = String(String.UnicodeScalarView(scalars[i..<j]))
                if let (id, score) = vocab[sub] {
                    matchedAny = true
                    let cand = best[i] + score
                    if cand > best[j] { best[j] = cand; backId[j] = id; backStart[j] = i }
                }
                j += 1
            }
            // single-scalar unk edge (always available so the path is complete)
            _ = matchedAny
            let cand = best[i] + unkScore
            if cand > best[i + 1] { best[i + 1] = cand; backId[i + 1] = -1; backStart[i + 1] = i }
        }

        // ---- backtrack ----
        var pieces: [(id: Int, start: Int, end: Int)] = []
        var pos = n
        while pos > 0 {
            let s = backStart[pos]
            pieces.append((backId[pos], s, pos))
            pos = s
        }
        pieces.reverse()

        // ---- emit, expanding unk edges via byte fallback ----
        var ids = [Int]()
        for p in pieces {
            if p.id >= 0 { ids.append(p.id); continue }
            let sub = String(String.UnicodeScalarView(scalars[p.start..<p.end]))
            if byteFallback {
                for b in Array(sub.utf8) {
                    ids.append(byteId[Int(b)] >= 0 ? byteId[Int(b)] : unkId)
                }
            } else {
                ids.append(unkId)
            }
        }
        return ids
    }

    /// Returns nil unless every embedded self-test vector reproduces exactly.
    public func runSelfTest() -> String? {
        for t in selftest {
            let got = encode(t.text)
            if got != t.ids { return "tokenizer mismatch on \"\(t.text)\": got \(got) want \(t.ids)" }
        }
        return nil
    }

    private static func byteValue(_ piece: String) -> Int? {
        guard piece.hasPrefix("<0x"), piece.hasSuffix(">") else { return nil }
        let hex = piece.dropFirst(3).dropLast()
        return Int(hex, radix: 16)
    }
}
