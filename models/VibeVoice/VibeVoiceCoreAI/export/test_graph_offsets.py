#!/usr/bin/env python
"""Run the EXPORTED base_lm graph (fp32 Core AI runtime) at offset 0 (fresh cache,
full sequence) and at offset L (seeded prefill) and compare each to HF.

    cd coreai-models && PYTHONPATH=../VibeVoiceCoreAI/export uv run python \
        ../VibeVoiceCoreAI/export/test_graph_offsets.py
"""

from __future__ import annotations

import asyncio
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORT_DIR))

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as HFQwen2  # noqa: E402

from vvexport import common  # noqa: E402
from coreai_models.export._constants import KEY_CACHE_NAME, VALUE_CACHE_NAME  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.primitives.macos.cache import KVCache  # noqa: E402

import os  # noqa: E402
if os.environ.get("VV_MANUAL_SDPA", "1") == "1":
    import coreai_models.models.macos.qwen2 as _q  # noqa: E402
    from vvexport.manual_sdpa import ManualSDPA  # noqa: E402
    _q.SDPA = ManualSDPA  # patch before any Attention is constructed

from export_llm import load_base  # noqa: E402

VOICE = "/Users/amosgyamfi/Desktop/VibeVoice/VibeVoice/demo/voices/streaming_model/en-Carter_man.pt"
WINDOW = [9707, 504, 389, 3671, 9518]
MAXCTX = 4096


def hf_stack(cfg, model_dir):
    hf = HFQwen2(copy.deepcopy(cfg)).eval().float()
    src = common.load_subtree("model.language_model.", model_dir=model_dir, dtype=torch.float32)
    hf.embed_tokens.load_state_dict({"weight": src["embed_tokens.weight"]})
    hf.norm = nn.Identity()
    hf.load_state_dict({k: v for k, v in src.items() if not k.startswith("embed_tokens")}, strict=False)
    return hf


async def run_graph(cfg, model_dir, ids, pos, k, v):
    from coreai.runtime import NDArray
    model = load_base(cfg, model_dir)
    k0, v0 = KVCache.create_cache_tensors(cfg, dtype=torch.float32)
    ref_inputs = {"input_ids": ids, "position_ids": pos, "k_cache": k0, "v_cache": v0}
    program = export_to_coreai(
        model, ref_inputs, dynamic_shapes=None,
        input_names=("input_ids", "position_ids"), output_names=("hidden",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
    )
    program.optimize()
    with tempfile.TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = program.save_asset(Path(tmp))
        async with asset.executable() as aimodel:
            fn = aimodel.load_function("main")
            res = await fn(
                {"input_ids": NDArray(data=ids.contiguous()),
                 "position_ids": NDArray(data=pos.contiguous())},
                state={KEY_CACHE_NAME: NDArray(data=k), VALUE_CACHE_NAME: NDArray(data=v)},
            )
    return torch.from_numpy(np.array(next(iter(res.values())).numpy()))


def main() -> None:
    model_dir = common.DEFAULT_MODEL_DIR
    cfg = copy.deepcopy(common.load_config(model_dir).decoder_config)
    cfg.num_hidden_layers = 4
    cfg.max_position_embeddings = MAXCTX
    hf = hf_stack(cfg, model_dir)
    q = len(WINDOW)
    ids = torch.tensor([WINDOW], dtype=torch.int32)

    # ---- offset 0: fresh cache, positions [0:q] ----
    pos0 = torch.arange(0, q, dtype=torch.int32).unsqueeze(0)
    k0 = np.zeros((cfg.num_hidden_layers, 1, cfg.num_key_value_heads, MAXCTX, 64), np.float32)
    v0 = np.zeros_like(k0)
    with torch.no_grad():
        ref0 = hf(inputs_embeds=hf.embed_tokens(ids.long()), position_ids=pos0.long()).last_hidden_state
    g0 = asyncio.run(run_graph(cfg, model_dir, ids, pos0, k0, v0))
    print(f"offset0  PSNR(graph vs HF) = {common.psnr(ref0, g0):.2f} dB  "
          f"(HF {ref0[0,-1].norm():.3f} / graph {g0[0,-1].norm():.3f})")

    # ---- offset L: seeded prefill, positions [0:L+q] ----
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        voice = torch.load(VOICE, map_location="cpu", weights_only=True)
    pkv = voice["lm"]["past_key_values"]
    L = pkv.key_cache[0].shape[2]
    posL = torch.arange(0, L + q, dtype=torch.int32).unsqueeze(0)
    kL = np.zeros((cfg.num_hidden_layers, 1, cfg.num_key_value_heads, MAXCTX, 64), np.float32)
    vL = np.zeros_like(kL)
    for i in range(cfg.num_hidden_layers):
        kL[i, 0, :, :L, :] = pkv.key_cache[i][0].float().numpy()
        vL[i, 0, :, :L, :] = pkv.value_cache[i][0].float().numpy()
    hfc = DynamicCache()
    for i in range(cfg.num_hidden_layers):
        hfc.update(pkv.key_cache[i].clone().float(), pkv.value_cache[i].clone().float(), i)
    with torch.no_grad():
        refL = hf(inputs_embeds=hf.embed_tokens(ids.long()),
                  position_ids=torch.arange(L, L + q).unsqueeze(0).long(),
                  past_key_values=hfc, use_cache=True).last_hidden_state
    gL = asyncio.run(run_graph(cfg, model_dir, ids, posL, kL, vL))
    print(f"offsetL  PSNR(graph vs HF) = {common.psnr(refL, gL):.2f} dB  "
          f"(HF {refL[0,-1].norm():.3f} / graph {gL[0,-1].norm():.3f})")


if __name__ == "__main__":
    main()
