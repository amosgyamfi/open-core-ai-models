#!/usr/bin/env python
"""Export the two VibeVoice Qwen2 stacks to *stateless* Core AI .aimodel graphs.

VibeVoice-Realtime-0.5B splits a Qwen2.5-0.5B (hidden=896, 24 layers, GQA 14/2,
head_dim=64, rope_theta=1e6) into:

  * base_lm  : lower 4 layers + embed_tokens, final norm = Identity.
  * tts_lm   : upper 20 layers + final RMSNorm, consumes inputs_embeds.

IMPORTANT — why stateless: the macOS 27 beta Core AI *runtime* miscompiles the
stateful attention path (externalized SDPA composite + KV-cache state via
mutable_slice_update). Apple's own f32 Qwen2-attention parity test fails by ~12.0
abs error, and our stateful export drifted ~4-6% at the incremental step, which
tripped the EOS classifier early. RoPE/RMSNorm/matmul/softmax/cat all run
correctly through the runtime, so we re-express the stacks WITHOUT state and
WITHOUT the SDPA composite (see ``vvexport/stateless_qwen2.py``):

    forward(ids|embeds[1,q], position_ids[1,L+q], past_k, past_v)
        -> hidden[1,q,896], new_k, new_v

The KV cache is a plain input (past_k/past_v, length L) and freshly-computed
keys/values are returned for the caller (Swift) to append. This validates at
~132 dB graph-vs-HF through the runtime.

Run from the coreai-models checkout:
    PYTHONPATH=.../VibeVoiceCoreAI/export uv run --no-sync python \
      VibeVoiceCoreAI/export/export_llm.py --which base --verify
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn

from vvexport import common
from vvexport.coreai_utils import save_program

logger = logging.getLogger("export.llm")
HF_ID = "microsoft/VibeVoice-Realtime-0.5B"

# Apple primitives / export path
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from vvexport.stateless_qwen2 import StatelessBaseLM, StatelessTTSLM  # noqa: E402


# --------------------------------------------------------------------------- #
# Weight remap: VibeVoice (separate q/k/v) -> Apple (fused qkv)
# --------------------------------------------------------------------------- #
def _remap_layer(src: dict, i: int, dst_prefix: str) -> dict:
    out = {}
    qw = src[f"layers.{i}.self_attn.q_proj.weight"]
    kw = src[f"layers.{i}.self_attn.k_proj.weight"]
    vw = src[f"layers.{i}.self_attn.v_proj.weight"]
    qb = src[f"layers.{i}.self_attn.q_proj.bias"]
    kb = src[f"layers.{i}.self_attn.k_proj.bias"]
    vb = src[f"layers.{i}.self_attn.v_proj.bias"]
    out[f"{dst_prefix}.self_attn.qkv_proj.weight"] = torch.cat([qw, kw, vw], 0)
    out[f"{dst_prefix}.self_attn.qkv_proj.bias"] = torch.cat([qb, kb, vb], 0)
    out[f"{dst_prefix}.self_attn.o_proj.weight"] = src[f"layers.{i}.self_attn.o_proj.weight"]
    for p in ("gate_proj", "up_proj", "down_proj"):
        out[f"{dst_prefix}.mlp.{p}.weight"] = src[f"layers.{i}.mlp.{p}.weight"]
    out[f"{dst_prefix}.input_layernorm.weight"] = src[f"layers.{i}.input_layernorm.weight"]
    out[f"{dst_prefix}.post_attention_layernorm.weight"] = src[
        f"layers.{i}.post_attention_layernorm.weight"
    ]
    return out


def load_base(config, model_dir):
    src = common.load_subtree("model.language_model.", model_dir=model_dir, dtype=torch.float32)
    sd = {"embed_tokens.weight": src["embed_tokens.weight"]}
    for i in range(config.num_hidden_layers):
        sd.update(_remap_layer(src, i, f"layers.{i}"))
    m = StatelessBaseLM(config).eval().float()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rope" not in k and "sdpa" not in k]
    if missing or unexpected:
        raise RuntimeError(f"base_lm load mismatch missing={missing[:6]} unexpected={unexpected[:6]}")
    return m


def load_tts(config, model_dir):
    src = common.load_subtree("model.tts_language_model.", model_dir=model_dir, dtype=torch.float32)
    sd = {"norm.weight": src["norm.weight"]}
    for i in range(config.num_hidden_layers):
        sd.update(_remap_layer(src, i, f"layers.{i}"))
    m = StatelessTTSLM(config).eval().float()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rope" not in k and "sdpa" not in k]
    if missing or unexpected:
        raise RuntimeError(f"tts_lm load mismatch missing={missing[:6]} unexpected={unexpected[:6]}")
    return m


# --------------------------------------------------------------------------- #
# Reference-input + export
# --------------------------------------------------------------------------- #
def export_stack(which: str, model_dir: Path, out_dir: Path, max_ctx: int, verify: bool):
    base_cfg = common.load_config(model_dir).decoder_config
    cfg = copy.deepcopy(base_cfg)
    cfg.num_hidden_layers = 4 if which == "base" else 20
    cfg.max_position_embeddings = max_ctx

    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers

    # One query token per call (q=1); multi-token windows are fed sequentially by
    # the Swift runtime (autoregressive attention is order-invariant). Only the
    # prefix length L is dynamic, which keeps torch.export's shape solver happy.
    L = 16
    past_k = torch.randn(n_layers, 1, nkv, L, hd, dtype=torch.float32)
    past_v = torch.randn(n_layers, 1, nkv, L, hd, dtype=torch.float32)
    position_ids = torch.arange(L, L + 1, dtype=torch.int32).unsqueeze(0)

    l_dim = torch.export.Dim("L", min=1, max=max_ctx - 2)
    dyn_past = {3: l_dim}

    if which == "base":
        model = load_base(cfg, model_dir)
        x = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        in_names = ("input_ids", "position_ids", "past_k", "past_v")
        ref_inputs = {"input_ids": x, "position_ids": position_ids,
                      "past_k": past_k, "past_v": past_v}
        dyn = {"input_ids": None, "position_ids": None,
               "past_k": dyn_past, "past_v": dyn_past}
    else:
        model = load_tts(cfg, model_dir)
        x = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32)
        in_names = ("inputs_embeds", "position_ids", "past_k", "past_v")
        ref_inputs = {"inputs_embeds": x, "position_ids": position_ids,
                      "past_k": past_k, "past_v": past_v}
        dyn = {"inputs_embeds": None, "position_ids": None,
               "past_k": dyn_past, "past_v": dyn_past}

    logger.info("Exporting stateless %s_lm (%d layers, max_ctx=%d)...",
                which, cfg.num_hidden_layers, max_ctx)
    program = export_to_coreai(
        model, ref_inputs, dynamic_shapes=dyn,
        input_names=in_names, output_names=("hidden", "new_k", "new_v"),
    )
    program.optimize()
    asset = common.ExportPaths(out_dir).aimodel(f"{which}_lm")
    save_program(program, asset, hf_model_id=HF_ID, component=f"{which}_lm")
    logger.info("Saved %s", asset)

    if verify:
        _verify_against_hf(which, cfg, model, model_dir)


def _verify_against_hf(which, cfg, apple_model, model_dir):
    """Compare the stateless stack (incremental, L-prefix cache) vs HF Qwen2."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as HFQwen2

    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_layers = cfg.num_hidden_layers
    q, L = 1, 20

    hf_cfg = copy.deepcopy(cfg)
    hf = HFQwen2(hf_cfg).eval().float()
    prefix = "model.language_model." if which == "base" else "model.tts_language_model."
    src = common.load_subtree(prefix, model_dir=model_dir, dtype=torch.float32)
    hf_sd = {k: v for k, v in src.items() if not k.startswith("embed_tokens")}
    if which == "base":
        hf.embed_tokens.load_state_dict({"weight": src["embed_tokens.weight"]})
        hf.norm = nn.Identity()
    hf.load_state_dict(hf_sd, strict=False)

    past_k = torch.randn(n_layers, 1, nkv, L, hd)
    past_v = torch.randn(n_layers, 1, nkv, L, hd)
    pos = torch.arange(L, L + q, dtype=torch.int32).unsqueeze(0)
    hfc = DynamicCache()
    for i in range(n_layers):
        hfc.update(past_k[i].clone(), past_v[i].clone(), i)

    with torch.no_grad():
        if which == "base":
            ids = torch.randint(1, cfg.vocab_size, (1, q), dtype=torch.int32)
            emb = apple_model.embed_tokens(ids)
            ref = hf(inputs_embeds=emb, position_ids=torch.arange(L, L + q).unsqueeze(0).long(),
                     past_key_values=hfc, use_cache=True).last_hidden_state
            out, _, _ = apple_model(ids, pos, past_k, past_v)
        else:
            emb = torch.randn(1, q, cfg.hidden_size)
            ref = hf(inputs_embeds=emb, position_ids=torch.arange(L, L + q).unsqueeze(0).long(),
                     past_key_values=hfc, use_cache=True).last_hidden_state
            out, _, _ = apple_model(emb, pos, past_k, past_v)
    logger.info("PSNR(%s_lm stateless vs HF Qwen2) = %.2f dB", which, common.psnr(ref, out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["base", "tts", "both"], default="both")
    ap.add_argument("--model-dir", default=str(common.DEFAULT_MODEL_DIR))
    ap.add_argument("--out-dir", default=str(common.DEFAULT_EXPORT_DIR))
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    which = ["base", "tts"] if args.which == "both" else [args.which]
    for w in which:
        export_stack(w, Path(args.model_dir), Path(args.out_dir), args.max_ctx, args.verify)


if __name__ == "__main__":
    main()
