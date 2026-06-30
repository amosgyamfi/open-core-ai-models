#!/usr/bin/env python
"""Isolate the KV-cache incremental path: Apple authored Qwen2 stack (seeded with
a real voice prefill) vs HF Qwen2 (same prefill) for the first text window.

This removes the Core AI runtime from the equation -- both run in PyTorch -- so a
low PSNR points at the KVCache primitive / RoPE / offset convention, while a high
PSNR means the discrepancy is elsewhere (Core AI fp, etc.).

    cd coreai-models && uv run python ../VibeVoiceCoreAI/export/test_cached_path.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

EXPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORT_DIR))

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as HFQwen2  # noqa: E402

from vvexport import common  # noqa: E402
from export_llm import load_base  # noqa: E402
from coreai_models.primitives.macos.cache import KVCache  # noqa: E402

VOICE = "/Users/amosgyamfi/Desktop/VibeVoice/VibeVoice/demo/voices/streaming_model/en-Carter_man.pt"
WINDOW = [9707, 504, 389, 3671, 9518]  # first 5 tokens of the test prompt


def psnr(a, b):
    return common.psnr(a, b)


def main() -> None:
    model_dir = common.DEFAULT_MODEL_DIR
    base_cfg = common.load_config(model_dir).decoder_config
    cfg = copy.deepcopy(base_cfg)
    cfg.num_hidden_layers = 4
    cfg.max_position_embeddings = 8192

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        voice = torch.load(VOICE, map_location="cpu", weights_only=True)
    lm = voice["lm"]
    pkv = lm["past_key_values"]
    n_layers = len(pkv.key_cache)
    L = pkv.key_cache[0].shape[2]
    print(f"lm prefill: layers={n_layers} L={L} k0shape={tuple(pkv.key_cache[0].shape)}")

    ids = torch.tensor([WINDOW], dtype=torch.long)
    # HF wants the query positions only; Apple wants the FULL span [0 .. L+q) and
    # derives offset = seq_len - query_len internally.
    hf_pos = torch.arange(L, L + len(WINDOW)).unsqueeze(0)
    apple_pos = torch.arange(0, L + len(WINDOW)).unsqueeze(0)

    # ---- HF reference: continue from the cached prefill ----
    hf = HFQwen2(copy.deepcopy(cfg)).eval().float()
    src = common.load_subtree("model.language_model.", model_dir=model_dir, dtype=torch.float32)
    hf.embed_tokens.load_state_dict({"weight": src["embed_tokens.weight"]})
    import torch.nn as nn
    hf.norm = nn.Identity()
    hf.load_state_dict({k: v for k, v in src.items() if not k.startswith("embed_tokens")},
                       strict=False)

    hf_cache = DynamicCache()
    for i in range(n_layers):
        hf_cache.update(pkv.key_cache[i].clone().float(), pkv.value_cache[i].clone().float(), i)
    emb = hf.embed_tokens(ids)
    with torch.no_grad():
        ref = hf(inputs_embeds=emb, position_ids=hf_pos.long(),
                 past_key_values=hf_cache, use_cache=True).last_hidden_state
    print(f"HF   out[-1] norm={ref[0, -1].norm():.4f} head={[round(x,4) for x in ref[0,-1,:6].tolist()]}")

    # ---- Apple authored stack seeded with the same prefill ----
    apple = load_base(cfg, model_dir)
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=torch.float32)
    seq_len = k_cache.shape[KVCache.seq_len_dim()]
    print(f"apple cache shape={tuple(k_cache.shape)} seq_dim={KVCache.seq_len_dim()} seq_len={seq_len}")
    # seed: copy prefill into the cache at [0:L]
    kc = k_cache.clone()
    vc = v_cache.clone()
    idx = [slice(None)] * kc.ndim
    for i in range(n_layers):
        # source (1, kv_heads, L, head_dim); place into apple layout per layer.
        _seed_layer(kc, vc, i, pkv.key_cache[i].float(), pkv.value_cache[i].float(), L)
    with torch.no_grad():
        apple_out = apple(ids.int(), apple_pos.int(), kc, vc)
    print(f"AP   out[-1] norm={apple_out[0,-1].norm():.4f} head={[round(x,4) for x in apple_out[0,-1,:6].tolist()]}")
    print(f"PSNR(apple cached vs HF cached) = {psnr(ref, apple_out):.2f} dB")


def _seed_layer(kc, vc, layer, k_src, v_src, L):
    """Place (1, kv_heads, L, head_dim) prefill into the apple cache tensors.

    The apple cache shape is created by KVCache.create_cache_tensors; we discover
    the layout by matching dims. Common layout: (n_layers, 1, kv_heads, seq, hd).
    """
    # k_src: (1, kv_heads, L, hd)
    kv_heads = k_src.shape[1]
    hd = k_src.shape[3]
    # Try (n_layers, 1, kv_heads, seq, hd)
    if kc.ndim == 5 and kc.shape[0] >= layer + 1 and kc.shape[2] == kv_heads and kc.shape[4] == hd:
        kc[layer, 0, :, :L, :] = k_src[0]
        vc[layer, 0, :, :L, :] = v_src[0]
    else:
        raise RuntimeError(f"unexpected cache shape {tuple(kc.shape)} vs src {tuple(k_src.shape)}")


if __name__ == "__main__":
    main()
