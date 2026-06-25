# Pocket-TTS → Core AI conversion recipe

This is the end-to-end recipe for converting [`kyutai/pocket-tts`](https://huggingface.co/kyutai/pocket-tts)
into three Apple Core AI `.aimodel` bundles plus host glue, with numeric-parity gates at every step. It
follows the pattern established by the community zoo's VoxCPM recipe (continuous/flow TTS that doesn't
fit Apple's standard `CoreAILM` pipeline).

## 0. Pipeline overview

```
capture_oracle.py  →  verify_eager.py  →  export.py  →  generate.py  →  package.py  →  verify_package.py
   (reference)         (re-author OK)      (.aimodel)     (e2e gate)      (ship dir)     (Swift contract)
```

The guiding principle is **oracle-driven re-authoring**: record every intermediate tensor from the
unmodified upstream model once, then prove each re-authored / exported stage reproduces it (cosine ≥
0.99) before moving on. This isolates bugs to a single stage instead of debugging the whole TTS chain.

## 1. Capture the oracle (`capture_oracle.py`)

Instrument the upstream `TTSModel` during one `generate_audio(...)` call and dump, to
`artifacts/oracle.npz`: text tokens, every backbone input/output (with write offsets), every flow
`(cond, noise, latent)` triple, the queued Mimi latents, and the final waveform.

Gotcha worth burning in: **`lsd_decode` mutates its noise input in place.** Snapshot `noise0` *before*
the call, or the recorded "noise" is actually the output latent and the flow gate silently "passes" on
garbage.

## 2. Re-author the three sub-networks (`pockettts_coreai/model.py`)

`torch.export` + `TorchConverter` reject data-dependent control flow, in-place state growth, and a
handful of ATen ops. The re-authoring keeps the upstream weights but makes the graph static and
export-safe:

### Backbone (`Backbone`)
- **One step per call.** Query length is fixed at 1; prefill is done by the host as a sequence of
  single-token steps (causally identical to a batched prefill).
- **Baked RoPE.** Rotary angles are computed from a `pos [1]` input (kept as a 1-element tensor — a
  0-D scalar trips `coreai.reshape: operand must be 1D … Index type`).
- **Explicit causal mask** over the `cache_len` window from `pos`.
- **One-hot masked KV write** instead of dynamic indexing:
  `k_all = k_cache·(1−onehot) + k·onehot`, with `onehot = (arange(CL) == pos)`. This is the export-safe
  way to mutate a fixed-capacity cache; `keyCache`/`valueCache` are declared as Core AI **states** and
  persist (mutated in place) across calls.

### Flow decoder (`FlowDecoder`)
- Wraps the upstream `SimpleMLPAdaLN` for a single LSD step `(cond, z) → latent`.
- **`aten.var.correction` is unsupported.** The custom `LayerNorm`/`RMSNorm` use `torch.var(...,
  correction=…)`; their `forward` is patched to compute variance manually (`x.pow(2).mean − mean²`).

### Mimi decoder (`MimiDecoder`)
- Stateless, full-sequence: `latents [1,T,32] → wav [1,1,1920·T]`, with `T` a `torch.export.Dim`.
- **`aten.elu` is unsupported.** SEANet's `nn.ELU` is patched to its closed form
  `where(x>0, x, α·(exp(x)−1))` (α = 1).
- Streaming convolution state is folded away — the decode runs over the whole sequence at once.

Run `verify_eager.py` to confirm all three reproduce the oracle in eager PyTorch (the voice KV state
must be seeded first; see `seed_caches_from_voice`).

## 3. Export + engine gate (`export.py`)

```python
ep = torch.export.export(module, kwargs=sample, dynamic_shapes=dyn).run_decompositions(get_decomp_table())
prog = TorchConverter().add_exported_program(ep, input_names=…, output_names=…, state_names=…).to_coreai()
prog.optimize()
prog.save_asset(out/"name.aimodel", rt.AIModelAssetMetadata())
```

Then each bundle is **loaded and run on the Core AI runtime** and compared to the oracle. Notes:

- The runtime API is **async** — `await rt.AIModel.load(...)`, `await fn(...)`.
- **Specialization fallback:** try `SpecializationOptions.cpu_only()` (best for numeric parity), and
  fall back to `default()` (GPU). The backbone's ~100 MB masked-write state graph does not compile on
  the CPU backend but is fine on GPU.
- State names `keyCache`/`valueCache` are passed via the `state=` argument and reused between calls.

Gate results (fp32 cos ≈ 1.0, fp16 cos ≥ 0.999) are in the top-level README.

## 4. End-to-end gate (`generate.py`)

Wires the three bundles + host glue into the real autoregressive loop on the engine. With
`--reproduce-oracle` it feeds the oracle's noise and text so the run is deterministic and comparable to
the oracle waveform (fp32 raw cosine 1.0). Without it, it samples fresh noise and writes a normal WAV.

The decode loop per frame:
```
seq        = prev_latent ?? bos_emb
input_emb  = input_linear @ seq                 # [1024]
hidden     = backbone(input_emb, pos);  pos++    # stateful
eos        = (out_eos · hidden + b) > threshold
latent     = flow(hidden, z),  z ~ N(0, temp)
… append latent, stop 2 frames after first EOS …
wav        = mimi(stack(latents))
```

## 5. Package (`package.py`) + Swift contract (`verify_package.py`)

`package.py` assembles `dist/PocketTTSCoreAI/`: the three bundles, `manifest.json` (shapes + scalars +
voice offsets), the SentencePiece `.model`, a Swift-friendly `tokenizer.json` (Unigram pieces/scores +
self-test vectors), the host-glue tensors as raw float32 `.bin`, and precomputed KV state for each
catalog voice (cropped to the written length).

`verify_package.py` then re-runs synthesis using **only** the package (no `pocket_tts` import) — this
is exactly the contract the Swift runtime implements, so it doubles as the reference for the Swift port
(and reproduces the oracle: fp16 magnitude-spectrum cosine 0.983).

## Environment

macOS 27, Python 3.12, PyTorch ≥ 2.5, and Apple's Core AI tools `coreai-core` / `coreai-torch` /
`coreai-opt`, plus upstream `pocket-tts` (pulls `sentencepiece`). A `.venv` under `conversion/` is
assumed by the commands in the README.
