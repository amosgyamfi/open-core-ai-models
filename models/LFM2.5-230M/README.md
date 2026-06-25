# LFM2.5-230M → Apple Core AI (`.aimodel`)

Converts [`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M) — a
230M-parameter conv + GQA **hybrid** decoder — into Apple **Core AI** `.aimodel`
bundles you can run on macOS / iOS 27 with the [Core AI framework](https://developer.apple.com/core-ai/).

Both bundles are validated end-to-end on the Core AI GPU runtime against the fp32
HuggingFace reference (top-1 parity at every confident position).

## What's in this folder

| Path | What |
|---|---|
| `exports/lfm2_5_230m_decode_int8lin/` | **Ship bundle** — int8 (per-block-32 linear) MLP + conv mixer, fp32 attention, fp16 head. ~294 MB. |
| `exports/lfm2_5_230m_decode_fp16/` | Full-precision baseline. ~474 MB. |
| `coreai-models/` | Apple's [`apple/coreai-models`](https://github.com/apple/coreai-models) checkout + the **LFM2 overlay** that makes this conversion possible. |
| `coreai-model-zoo/` | Community [`john-rocky/coreai-model-zoo`](https://github.com/john-rocky/coreai-model-zoo) checkout (export-script reference). |
| `coreai-models/python/src/coreai_models/models/macos/lfm2.py` | **The re-authored LFM2.5 decoder** (the missing piece — see below). |
| `export_lfm2_230m.py` | Export driver (PyTorch → `.aimodel`). |
| `verify_lfm2_230m.py` | Oracle parity gate (re-authored torch model vs fp32 HF). |
| `run_aimodel_check.py` | Engine gate (runs the exported `.aimodel` through the Core AI runtime). |
| `generate_demo.py` | Greedy chat generation through the Core AI runtime. |

Each bundle is a Core AI **LanguageBundle**: a `.aimodel` (the `main.mlirb`
program + metadata), a `metadata.json`, a `tokenizer/` folder, and the upstream
`LICENSE`.

## Why a custom overlay was needed

LFM2.5 is **not** in Apple's `coreai-models` catalog (which ships Qwen3, Gemma 3,
Mistral, etc.), and the community zoo's `export_lfm2_decode_pipelined.py` imports
`coreai_models.models.macos.lfm2` — a re-authored decoder the zoo author keeps as
private working-tree edits and never published. So the model had to be
**re-authored from scratch** against Apple's `coreai_models` primitives
(`KVCache`, `RMSNorm`, `RoPE`, `SDPA`, `MLP`), faithfully matching the HF
`transformers` LFM2 implementation:

- 14 layers = **8 short-conv mixers + 6 GQA attention layers** (`layer_types`),
  hidden 1024, 16 q / 8 kv heads (head_dim 64), depthwise causal conv kernel 3,
  per-head q/k RMSNorm, full-dim RoPE θ=1e6, SwiGLU MLP (ff 2560), tied 65 536 head.
- **Decode-only, S=1, loop-free** (the conv mixer is a 3-tap depthwise conv — no
  recurrent scan), so prefill runs as pipelined S=1 steps.

Two macOS-27 GPU-delegate workarounds (documented by the zoo) are baked into the
overlay:

1. **One fused conv-state write per step.** Each conv layer returns its new
   history columns; the model issues a single full-state `mutable_slice_update`
   (per-layer chained writes are silently dropped by the GPU delegate).
2. **fp32 attention projections.** q/k/v/out keep fp32 weights (LFM2.5's large
   q/k-norm gains amplify fp16 matmul error into garbage logits). The conv-mixer
   and MLP matmuls stay fp16/int8.

## Validation (this machine: M-series, macOS 27.0 / Xcode 27.0)

Prompt: *"The capital of France is Paris. The largest planet in the solar system is"*

| Gate | What it checks | Result |
|---|---|---|
| Oracle (torch, fp16) | re-authored model vs fp32 HF | **15/15** confident top-1, cos 0.999996 |
| Oracle (torch, int8lin) | quantized model vs fp32 HF | **15/15** confident top-1, cos 0.999862 |
| Engine (`.aimodel`, fp16, GPU) | exported bundle vs fp32 HF | **15/15** confident top-1 |
| Engine (`.aimodel`, int8lin, GPU) | exported bundle vs fp32 HF | **15/15** confident top-1 |

One position (margin 0.094 < 0.1) is an fp32 statistical tie and is excluded from
the gate per the zoo's instrument-selection rule; every confident position matches.

Greedy chat sample (int8 bundle, through the Core AI runtime):

> **In one sentence, what is C. elegans?** → *C. elegans is a well-studied
> nematode worm … widely used in research due to its simple biology and ease of study.*

## Reproduce

Requires macOS 27+/Xcode 27+ and `uv` (≥ 0.9).

```bash
cd coreai-models
uv sync                       # installs coreai-core / coreai-torch / coreai-opt + torch 2.9

# export (ship config)
uv run python ../export_lfm2_230m.py int8lin --hf-id LiquidAI/LFM2.5-230M --out-dir ../exports
# or full precision
uv run python ../export_lfm2_230m.py fp16    --hf-id LiquidAI/LFM2.5-230M --out-dir ../exports

# parity gates
uv run python ../verify_lfm2_230m.py --mode int8lin
uv run python ../run_aimodel_check.py --bundle ../exports/lfm2_5_230m_decode_int8lin/lfm2_5_230m_decode_int8lin.aimodel --unit gpu
uv run python ../generate_demo.py     --bundle ../exports/lfm2_5_230m_decode_int8lin/lfm2_5_230m_decode_int8lin.aimodel
```

## Using it in a Core AI project

The bundle exposes one inference function, `main`:

- **inputs**: `input_ids [1,1] i32`, `position_ids [1,S] i32`
- **states**: `keyCache`, `valueCache` (6 attention layers), `convState`
  (`[8,1,1024,2]` fixed-shape conv history)
- **output**: `logits [1,1,65536]`

`input_ids` is static `[1,1]`; `position_ids` and the KV seq length are dynamic, so
the bundle is decode-only. Drive it by feeding the prompt one token at a time, then
generating. Swift sketch (Core AI runtime):

```swift
import CoreAI

let model = try await AIModel.load(at: bundleURL,
    specialization: .init(preferredComputeUnit: .gpu))
let fn = try model.loadFunction(named: "main")

// allocate the three state buffers once and reuse them across decode steps:
//   keyCache/valueCache: [6, 1, 8, maxSeq, 64] fp16
//   convState:           [8, 1, 1024, 2] fp16
let state = makeState()  // NDArrays you keep for the whole generation

for pos in 0..<promptTokens.count + maxNew {
    let inputs = ["input_ids": ids1x1, "position_ids": positions1xS]
    let out = try await fn(inputs: inputs, state: state)   // state mutated in place
    let next = argmax(out["logits"]!)
    // append next, grow position_ids, repeat
}
```

> **On-device throughput note.** For the high-throughput pipelined GPU engine
> (the zoo reports ~200+ tok/s for LFM2.5 on M4 Max / ~40+ on iPhone), set
> `COREAI_CHUNK_THRESHOLD=1` before engine creation, and because `convState` is an
> extra fixed-shape state, the zoo's `coreai-pipelined-extra-states` Swift patch is
> needed for that engine. The plain `coreai.runtime` / `AIModel` path used by the
> gates here drives the three states directly with no patch (verified on GPU).

Use this model for data extraction and lightweight on-device agentic / tool-use
pipelines — not reasoning-heavy math, code, or creative writing (per LiquidAI).

## Licensing

- **Model weights**: LFM Open License v1.0 (`lfm1.0`) — see `LICENSE` shipped inside
  each bundle. Commercial use is licensed only below a US $10 M annual-revenue
  threshold; redistribution must retain notices and include the license.
- **Conversion code** (the `lfm2.py` overlay + scripts): BSD-3-Clause, derived from
  Apple's `coreai_models`.
```
