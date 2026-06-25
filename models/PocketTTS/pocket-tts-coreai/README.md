# Pocket-TTS → Apple Core AI

A community port of [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts) (a ~100M-parameter,
CPU-friendly streaming TTS model) to **Apple Core AI** (`.aimodel`), with a Swift runtime package and
CLI so you can synthesize speech fully on-device on iOS/macOS 27.

> **Not an Apple model.** Pocket-TTS is © Kyutai, released under **CC-BY-4.0**. This repository only
> contains the *conversion recipe*, host glue, and Swift runtime — no upstream weights are committed.
> You build the `.aimodel` bundles yourself from the public checkpoint. Respect the upstream
> [prohibited-use policy](https://huggingface.co/kyutai/pocket-tts) (no voice impersonation/cloning
> without consent, no deception, etc.).

## What's here

```
pocket-tts-coreai/
├── conversion/                 # PyTorch → Core AI recipe (Python)
│   ├── pockettts_coreai/       #   export-safe re-authoring of the 3 sub-networks
│   ├── capture_oracle.py       #   record reference tensors from upstream pocket-tts
│   ├── verify_eager.py         #   eager parity of the re-authoring vs oracle
│   ├── export.py               #   torch.export → .aimodel + engine parity gate
│   ├── generate.py             #   full synthesis on the engine (oracle-gated + free)
│   ├── package.py              #   assemble the distributable package
│   └── verify_package.py       #   package-only synthesis (the Swift contract, in Python)
└── swift/                      # On-device runtime
    ├── Sources/PocketTTSCoreAI #   engine + Unigram tokenizer + WAV writer
    └── Sources/pocket-tts-cli  #   `pocket-tts` CLI
```

## The model, split into three Core AI graphs

Pocket-TTS is a continuous (flow-matching) audio LM. It does not fit Apple's standard
`input_ids → logits` `CoreAILM` pipeline, so it is exported as **three single-graph bundles** driven by
a thin host autoregressive loop (the same shape as the VoxCPM recipe in the community zoo):

| Bundle | Inputs | Output | State | Role |
|---|---|---|---|---|
| `backbone.aimodel` | `inputs_embeds [1,1,1024]`, `pos [1]` | `hidden [1,1,1024]` | `keyCache`, `valueCache` `[6,1,16,2048,64]` | one stateful transformer step (RoPE + causal KV) |
| `flow.aimodel` | `cond [1,1024]`, `z [1,32]` | `latent [1,32]` | — | Lagrangian self-distillation flow step |
| `mimi.aimodel` | `latents [1,T,32]` (dynamic `T`) | `wav [1,1,1920·T]` | — | Mimi neural codec decode → 24 kHz PCM |

Everything else is tiny **host glue** (token embedding lookup, latent→model-dim projection, BOS, EOS
head, Gaussian noise, voice-state seeding) — pure Accelerate-friendly vector math, shipped as raw
float32 `.bin` tensors in the package.

## Parity (engine vs upstream)

Every bundle is gated on the Core AI runtime against a recorded upstream oracle:

| Bundle | fp32 | fp16 |
|---|---|---|
| backbone (52 stateful steps) | `min_cos 1.000000` | `min_cos 0.999971` |
| flow (38 steps) | `min_cos 1.000000` | `min_cos 0.999165` |
| mimi (69 120 samples) | `cos 1.000011` | `cos 0.999971` |
| **end-to-end** (host loop on engine) | raw-waveform `cos 1.0000` | magnitude-spectrum `cos 0.983`¹ |

¹ fp16 accumulates tiny per-step phase drift over the autoregressive chain, so sample-aligned
waveform cosine is a pessimistic metric; the magnitude spectrum (what you hear) stays ~identical.
Use **fp32** when you need bit-exact reproduction, **fp16** to halve size for shipping.

## Quickstart — build the bundles

Requires macOS 27 + the Apple Core AI Python tools (`coreai-core`, `coreai-torch`, `coreai-opt`) and
PyTorch 2.5+. From `conversion/` (a `.venv` is expected):

```bash
.venv/bin/python capture_oracle.py            # record upstream reference (downloads the checkpoint)
.venv/bin/python verify_eager.py              # eager re-authoring parity
.venv/bin/python export.py all --dtype fp16   # → artifacts/bundles/*.aimodel  (+ engine gate)
.venv/bin/python generate.py --reproduce-oracle --dtype fp32   # full-pipeline gate
.venv/bin/python package.py --dtype fp16      # → dist/PocketTTSCoreAI/  (ship this)
```

See [`conversion/CONVERSION.md`](conversion/CONVERSION.md) for the full recipe and the export-safety
rewrites that made it work.

## Quickstart — run on-device (Swift)

The `dist/PocketTTSCoreAI/` package (bundles + `manifest.json` + tokenizer + glue + voices) is fully
self-contained. From `swift/` on macOS 27 / Xcode 27:

```bash
swift run pocket-tts \
  --package /path/to/dist/PocketTTSCoreAI \
  --text "Hello world, this is a test of pocket text to speech." \
  --voice alba --output out.wav
```

Or embed the library:

```swift
import PocketTTSCoreAI

let tts = try await PocketTTS(packageURL: packageURL)
let pcm = try await tts.synthesize(text: "Hello from Core AI.", voice: "alba")  // [Float] @ 24 kHz
try WavWriter.write(pcm, sampleRate: tts.sampleRate, to: outURL)
```

Voices included: `alba, marius, javert, jean, fantine, cosette, eponine, azelma`.

> The Swift sources are a faithful **draft against the verified Core AI Swift API** (mirroring Apple's
> `CoreAISequentialEngine` and the zoo's `HybridCoreAIEngine`); compile and shake out on macOS 27 since
> the `CoreAI` framework is absent from earlier SDKs. The pure-Swift Unigram tokenizer is validated to
> match upstream SentencePiece exactly on 311/312 fuzz cases (the package embeds self-test vectors the
> runtime checks on load).

## License

Recipe/runtime code: same license as this repository. Upstream Pocket-TTS weights and voices:
CC-BY-4.0 © Kyutai — see the [model card](MODEL_CARD.md).
