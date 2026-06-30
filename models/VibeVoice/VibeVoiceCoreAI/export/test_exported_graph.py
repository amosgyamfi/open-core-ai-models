#!/usr/bin/env python
"""Run the EXPORTED base_lm.aimodel through the Core AI Python runtime with a real
voice prefill seeded into the KV-cache state, comparing to HF.

If this matches HF but the Swift run does not, the bug is in the Swift runtime
calls (seeding / state plumbing). If this also diverges, the export is at fault.

    cd coreai-models && PYTHONPATH=../VibeVoiceCoreAI/export uv run python \
        ../VibeVoiceCoreAI/export/test_exported_graph.py
"""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path

import numpy as np
import torch

EXPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORT_DIR))

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as HFQwen2  # noqa: E402

from vvexport import common  # noqa: E402

VOICE = "/Users/amosgyamfi/Desktop/VibeVoice/VibeVoice/demo/voices/streaming_model/en-Carter_man.pt"
AIMODEL = "/Users/amosgyamfi/Desktop/VibeVoice/VibeVoiceCoreAI/exports/base_lm.aimodel"
WINDOW = [9707, 504, 389, 3671, 9518]
MAXCTX = 4096  # mirror the Swift default GenerationOptions.maxContext


def main() -> None:
    model_dir = common.DEFAULT_MODEL_DIR
    cfg = copy.deepcopy(common.load_config(model_dir).decoder_config)
    cfg.num_hidden_layers = 4

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        voice = torch.load(VOICE, map_location="cpu", weights_only=True)
    pkv = voice["lm"]["past_key_values"]
    n_layers, kv_heads = len(pkv.key_cache), pkv.key_cache[0].shape[1]
    L, hd = pkv.key_cache[0].shape[2], pkv.key_cache[0].shape[3]
    print(f"prefill layers={n_layers} kv_heads={kv_heads} L={L} hd={hd}")

    ids = torch.tensor([WINDOW], dtype=torch.long)
    q = len(WINDOW)

    # ---- HF reference (continue from prefill) ----
    hf = HFQwen2(copy.deepcopy(cfg)).eval().float()
    src = common.load_subtree("model.language_model.", model_dir=model_dir, dtype=torch.float32)
    import torch.nn as nn
    hf.embed_tokens.load_state_dict({"weight": src["embed_tokens.weight"]})
    hf.norm = nn.Identity()
    hf.load_state_dict({k: v for k, v in src.items() if not k.startswith("embed_tokens")}, strict=False)
    hfc = DynamicCache()
    for i in range(n_layers):
        hfc.update(pkv.key_cache[i].clone().float(), pkv.value_cache[i].clone().float(), i)
    with torch.no_grad():
        ref = hf(inputs_embeds=hf.embed_tokens(ids),
                 position_ids=torch.arange(L, L + q).unsqueeze(0).long(),
                 past_key_values=hfc, use_cache=True).last_hidden_state
    print(f"HF   out[-1] norm={ref[0,-1].norm():.4f} head={[round(x,4) for x in ref[0,-1,:6].tolist()]}")

    # ---- exported graph via Core AI runtime, seeded exactly like Swift ----
    k = np.zeros((n_layers, 1, kv_heads, MAXCTX, hd), dtype=np.float32)
    v = np.zeros((n_layers, 1, kv_heads, MAXCTX, hd), dtype=np.float32)
    for i in range(n_layers):
        k[i, 0, :, :L, :] = pkv.key_cache[i][0].float().numpy()
        v[i, 0, :, :L, :] = pkv.value_cache[i][0].float().numpy()
    position_ids = np.arange(0, L + q, dtype=np.int32)[None, :]
    input_ids = np.array([WINDOW], dtype=np.int32)

    out = asyncio.run(_run_graph(input_ids, position_ids, k, v))
    o = torch.from_numpy(out)
    print(f"GR   out[-1] norm={o[0,-1].norm():.4f} head={[round(x,4) for x in o[0,-1,:6].tolist()]}")
    print(f"PSNR(exported graph vs HF) = {common.psnr(ref, o):.2f} dB")


async def _run_graph(input_ids, position_ids, k, v):
    from coreai.runtime import AIModel, NDArray

    model = await AIModel.load(AIMODEL)
    fn = model.load_function("main")
    inputs = {
        "input_ids": NDArray(data=np.ascontiguousarray(input_ids)),
        "position_ids": NDArray(data=np.ascontiguousarray(position_ids)),
    }
    state = {
        "keyCache": NDArray(data=np.ascontiguousarray(k)),
        "valueCache": NDArray(data=np.ascontiguousarray(v)),
    }
    results = await fn(inputs, state=state)
    out = next(iter(results.values()))
    return np.array(out.numpy())


if __name__ == "__main__":
    main()
