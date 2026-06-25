# Pocket TTS Core AI Conversion Workspace

This repository prepares Kyutai Pocket TTS for Apple Core AI by converting the model's tensor submodules into `.aimodel` assets with the Core AI PyTorch Extensions (`coreai-torch`). The goal is an on-device SwiftUI TTS pipeline where Swift owns text preprocessing, autoregressive generation, audio scheduling, and playback, while Core AI runs the heavy neural network pieces on Apple hardware.

Pocket TTS is a compact multilingual text-to-speech system from Kyutai. The upstream project describes it as a CPU-friendly model with about 100M parameters, streaming generation, low first-chunk latency, voice conditioning, and support for English, French, German, Portuguese, Italian, and Spanish. The Hugging Face model is gated, so you must accept the model terms before downloading the voice-cloning weights.

## What Gets Converted

Pocket TTS is not a single static `text -> waveform` graph. It combines SentencePiece tokenization, text embedding, a FlowLM autoregressive latent generator, a Mimi audio codec/decoder, random sampling, streaming state, and Python-side control flow.

This workspace therefore exports three Core AI model boundaries:

1. `text_conditioner.aimodel`: token ids to FlowLM text embeddings.
2. `flow_lm_context_step.aimodel`: full latent context plus text embeddings plus explicit noise to the next normalized audio latent and EOS logit.
3. `mimi_decode_chunk.aimodel`: normalized FlowLM latents to PCM audio samples.

The split keeps random sampling and autoregressive orchestration outside the model. That is intentional: Swift can seed noise, apply EOS thresholds, append latents, and stream decoded audio deterministically.

The English conversion in this workspace has been generated at:

```text
models/coreai/pocket-tts-english/
  text_conditioner.aimodel/          # 16 MB
  flow_lm_context_step.aimodel/      # 336 MB
  mimi_decode_chunk.aimodel/         # 39 MB
  metadata.json
```

## Current Limitations

- `coreai-torch==0.4.0` depends on `coreai-core==1.0.0b1`. On macOS arm64, `uv` reports no matching wheel. This repo therefore includes `docker/Dockerfile.coreai`, which runs the conversion in Linux x86_64 via Colima/Docker where Apple's `manylinux_2_34_x86_64` wheel is available.
- The Hugging Face `kyutai/pocket-tts` repository is gated. Run `hf auth login` with an account that accepted the model terms before converting the gated voice-cloning model.
- The exported FlowLM step uses a full latent context instead of Pocket TTS's mutable KV cache. This is simpler to compile and integrate, but it is less efficient than the original Python streaming path.
- The exported FlowLM step prepends the BOS embedding internally. Do not pass `NaN` BOS sentinels to the Core AI model.
- SentencePiece tokenization is not converted. A Swift app needs a tokenizer implementation or a preprocessing layer that supplies token ids.
- The Swift app must provide standard-normal noise to `flow_lm_context_step.aimodel`. Use a deterministic random generator if repeatable speech is needed.
- Voice cloning from arbitrary audio is not included in the exported Core AI boundary. Prefer precomputed voice states or a separate conversion path for audio prompt encoding.
- The converter applies export-only rewrites for Core AI compatibility: Pocket TTS custom `var`-based normalizations are rewritten to mean-square formulas, and Mimi `ELU(alpha=1)` is rewritten to an equivalent lower-level expression.

## Setup

Create a local environment:

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -e .
```

Install Apple's Core AI PyTorch Extensions on a supported platform:

```bash
uv pip install '.[conversion]' --prerelease=allow
```

On this Mac, direct installation is not supported by the currently published Core AI wheel. The working local route is:

```bash
brew install colima docker qemu lima-additional-guestagents
colima start --arch x86_64 --cpu 4 --memory 8 --disk 80
docker build --platform linux/amd64 \
  -f docker/Dockerfile.coreai \
  -t pocket-tts-coreai:amd64 .
```

Check the environment:

```bash
python scripts/check_environment.py
```

Authenticate with Hugging Face if you want the gated model:

```bash
hf auth login
```

Optional offline asset download:

```bash
python scripts/download_hf_assets.py --language english --local-dir models/hf/pocket-tts
```

## Convert To Core AI

Run:

```bash
python scripts/convert_pocket_tts_to_coreai.py \
  --language english \
  --output-dir models/coreai/pocket-tts-english \
  --max-text-tokens 96 \
  --max-latent-context 64 \
  --decode-latent-frames 1 \
  --lsd-decode-steps 4 \
  --temperature 0.7
```

On macOS arm64, run the same converter inside the Linux x86_64 image:

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  pocket-tts-coreai:amd64 \
  python scripts/convert_pocket_tts_to_coreai.py \
    --language english \
    --output-dir models/coreai/pocket-tts-english
```

Expected output:

```text
models/coreai/pocket-tts-english/
  text_conditioner.aimodel/
  flow_lm_context_step.aimodel/
  mimi_decode_chunk.aimodel/
  metadata.json
```

## SwiftUI Integration Guide

A Core AI SwiftUI app should wrap the three assets in a small TTS engine:

1. Load the `.aimodel` assets with Core AI runtime APIs.
2. Tokenize input text with the same Pocket TTS SentencePiece tokenizer.
3. Pad or split text to the exported `max_text_tokens`.
4. Run `text_conditioner` to get `text_embeddings`.
5. Initialize a latent context of shape `[1, max_latent_context, latent_dim]` with previous normalized latents. The model prepends BOS internally.
6. For each generation step, sample a standard-normal `noise` tensor of shape `[1, latent_dim]`.
7. Run `flow_lm_context_step`.
8. Stop when `eos_logit` crosses your threshold and you have emitted the desired trailing frames.
9. Append `next_latent` into the rolling latent context.
10. Run `mimi_decode_chunk` on one or more generated latents and stream the PCM output through `AVAudioEngine`.

The `metadata.json` written by the converter records the sample rate, latent dimensions, context sizes, and asset paths that the Swift side should mirror.

## References

- Pocket TTS GitHub: https://github.com/kyutai-labs/pocket-tts
- Pocket TTS Hugging Face: https://huggingface.co/kyutai/pocket-tts
- Apple Core AI: https://developer.apple.com/documentation/coreai
- Core AI PyTorch Extensions: https://apple.github.io/coreai-torch/main/
- Apple Core AI model examples: https://github.com/apple/coreai-models/tree/main/models
- Community Core AI model zoo: https://github.com/john-rocky/coreai-model-zoo
