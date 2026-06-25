# Model card — Pocket-TTS for Core AI

## Summary

On-device text-to-speech for Apple platforms, ported from [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts)
to **Apple Core AI** (`.aimodel`). English, ~100M parameters, 24 kHz output, voice selection from a
fixed catalog of packaged speakers.

- **Upstream model:** Kyutai Pocket-TTS (Continuous Audio Language Model). Paper: [arXiv:2509.06926](https://arxiv.org/abs/2509.06926).
- **Upstream license:** CC-BY-4.0.
- **This artifact:** a community conversion recipe + Swift runtime. **Not affiliated with or endorsed by Apple or Kyutai.**
- **Target runtime:** Core AI on iOS 27 / macOS 27 (Apple Silicon).

## Architecture (as exported)

Three Core AI graphs + a host autoregressive loop:

1. **Backbone** — 6-layer / 1024-dim / 16-head streaming transformer, one token step per call, with
   baked Rotary Positional Embeddings, an explicit causal mask, and a fixed-capacity (`cache_len=2048`)
   key/value cache exposed as two mutable Core AI states. KV slots are updated with an export-safe
   one-hot masked write at the current position.
2. **Flow decoder** — `SimpleMLPAdaLN` flow network performing one Lagrangian self-distillation step,
   mapping `(hidden condition, Gaussian noise) → continuous audio latent [32]`.
3. **Mimi decoder** — the Mimi neural codec (SEANet decoder + projected transformer), run statelessly
   over the full latent sequence with a dynamic length `T`, producing 1920 PCM samples per frame.

Host glue (float32, shipped as raw `.bin`): SentencePiece (Unigram) tokenizer, token embedding table
`[4001,1024]`, latent→model projection, BOS embedding, EOS head, noise sampling, and precomputed
per-voice KV state.

## Inputs / outputs

- **Input:** UTF-8 English text + a voice name from the packaged catalog.
- **Output:** mono 16-bit PCM at 24 kHz.

## Intended use

On-device narration, accessibility/screen-reading, voice for assistants, prototyping. Designed for
low-latency CPU/GPU inference without network access.

## Out-of-scope / prohibited use

Per the upstream policy: **no** voice impersonation or cloning of real people without explicit lawful
consent; **no** misinformation, deception, or presenting generated audio as genuine recordings; **no**
unlawful, harassing, discriminatory, or privacy-invasive content. This port ships only fixed catalog
voices and intentionally omits the voice-cloning front-end.

## Precision & size

| Variant | Bundles total | Notes |
|---|---|---|
| fp16 (ship) | ~183 MB (backbone 144 / flow 19 / mimi 20) | engine parity cos ≥ 0.999 per stage |
| fp32 (reference) | ~364 MB | bit-exact reproduction of upstream |

## Limitations

- **English only** (upstream limitation).
- fp16 introduces small autoregressive drift — perceptually transparent but not sample-identical to upstream.
- Fixed catalog voices only; no runtime voice cloning.
- Backbone CPU-only specialization can fail to compile the large masked-write state graph on some
  configurations; it loads on the default/GPU specialization (handled automatically by the loader).

## Validation

Numeric parity is gated at every layer against a recorded upstream oracle: per-bundle on the Core AI
runtime, and end-to-end through the full host loop. fp32 reproduces the reference waveform at cosine
≈ 1.0. See [README](README.md) for the parity table and [CONVERSION.md](conversion/CONVERSION.md) for
methodology.

## Attribution

Pocket-TTS by Manu Orsini, Simon Rouard, Gabriel de Marmiesse, Václav Volhejn, Neil Zeghidour,
Alexandre Défossez (Kyutai). If you use this port, credit Kyutai and cite arXiv:2509.06926.
