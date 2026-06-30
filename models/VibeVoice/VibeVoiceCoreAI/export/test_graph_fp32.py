#!/usr/bin/env python
"""Export base_lm fresh and run it through the Core AI runtime (executable path,
fp32 CPU) with a real voice prefill seeded into the cache, comparing to HF.

High PSNR here => the exported graph is correct and the Swift GPU (fp16) run is
what introduces the early-EOS drift.

    cd coreai-models && PYTHONPATH=../VibeVoiceCoreAI/export uv run python \
        ../VibeVoiceCoreAI/export/test_graph_fp32.py
"""

from __future__ import annotations

import asyncio
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

EXPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORT_DIR))

from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as HFQwen2  # noqa: E402

from vvexport import common  # noqa: E402
from export_llm import load_base  # noqa: E402
from coreai_models.export._constants import KEY_CACHE_NAME, VALUE_CACHE_NAME  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.primitives.macos.cache import KVCache  # noqa: E402

VOICE = "/Users/amosgyamfi/Desktop/VibeVoice/VibeVoice/demo/voices/streaming_model/en-Carter_man.pt"
WINDOW = [9707, 504, 389, 3671, 9518]
MAXCTX = 4096


def main() -> None:
    model_dir = common.DEFAULT_MODEL_DIR
    cfg = copy.deepcopy(common.load_config(model_dir).decoder_config)
    cfg.num_hidden_layers = 4
    cfg.max_position_embeddings = MAXCTX

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        voice = torch.load(VOICE, map_location="cpu", weights_only=True)
    pkv = voice["lm"]["past_key_values"]
    n_layers, kv_heads = len(pkv.key_cache), pkv.key_cache[0].shape[1]
    L, hd = pkv.key_cache[0].shape[2], pkv.key_cache[0].shape[3]
    q = len(WINDOW)

    # HF reference continuing from prefill
    hf = HFQwen2(copy.deepcopy(cfg)).eval().float()
    src = common.load_subtree("model.language_model.", model_dir=model_dir, dtype=torch.float32)
    import torch.nn as nn
    hf.embed_tokens.load_state_dict({"weight": src["embed_tokens.weight"]})
    hf.norm = nn.Identity()
    hf.load_state_dict({k: v for k, v in src.items() if not k.startswith("embed_tokens")}, strict=False)
    hfc = DynamicCache()
    for i in range(n_layers):
        hfc.update(pkv.key_cache[i].clone().float(), pkv.value_cache[i].clone().float(), i)
    ids = torch.tensor([WINDOW], dtype=torch.long)
    with torch.no_grad():
        ref = hf(inputs_embeds=hf.embed_tokens(ids),
                 position_ids=torch.arange(L, L + q).unsqueeze(0).long(),
                 past_key_values=hfc, use_cache=True).last_hidden_state
    print(f"HF  norm={ref[0,-1].norm():.4f} head={[round(x,4) for x in ref[0,-1,:6].tolist()]}")

    out = asyncio.run(_export_and_run(cfg, model_dir, pkv, n_layers, kv_heads, L, hd, q))
    o = torch.from_numpy(out)
    print(f"GR  norm={o[0,-1].norm():.4f} head={[round(x,4) for x in o[0,-1,:6].tolist()]}")
    print(f"PSNR(exported fp32 runtime vs HF) = {common.psnr(ref, o):.2f} dB")


async def _export_and_run(cfg, model_dir, pkv, n_layers, kv_heads, L, hd, q):
    from coreai.runtime import NDArray

    model = load_base(cfg, model_dir)
    # static shapes for a clean, inferable output
    k0, v0 = KVCache.create_cache_tensors(cfg, dtype=torch.float32)
    pos = torch.arange(0, L + q, dtype=torch.int32).unsqueeze(0)
    ids = torch.tensor([WINDOW], dtype=torch.int32)
    ref_inputs = {"input_ids": ids, "position_ids": pos, "k_cache": k0, "v_cache": v0}
    program = export_to_coreai(
        model, ref_inputs, dynamic_shapes=None,
        input_names=("input_ids", "position_ids"), output_names=("hidden",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
    )
    program.optimize()

    k = np.zeros((n_layers, 1, kv_heads, MAXCTX, hd), dtype=np.float32)
    v = np.zeros((n_layers, 1, kv_heads, MAXCTX, hd), dtype=np.float32)
    for i in range(n_layers):
        k[i, 0, :, :L, :] = pkv.key_cache[i][0].float().numpy()
        v[i, 0, :, :L, :] = pkv.value_cache[i][0].float().numpy()

    with tempfile.TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = program.save_asset(Path(tmp))
        async with asset.executable() as aimodel:
            fn = aimodel.load_function("main")
            inputs = {
                "input_ids": NDArray(data=ids.contiguous()),
                "position_ids": NDArray(data=pos.contiguous()),
            }
            state = {KEY_CACHE_NAME: NDArray(data=k), VALUE_CACHE_NAME: NDArray(data=v)}
            res = await fn(inputs, state=state)
    return np.array(next(iter(res.values())).numpy())


if __name__ == "__main__":
    main()
