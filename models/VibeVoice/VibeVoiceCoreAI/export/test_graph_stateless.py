#!/usr/bin/env python
"""Export base_lm with the KV cache as regular (non-state) fixed-shape I/O and
manual SDPA, then run through the Core AI runtime at offset 0 and offset L.

If this matches HF where the stateful export did not, the runtime's *state*
feature is the culprit and we should export the LLM statelessly.
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
import coreai_models.models.macos.qwen2 as _q  # noqa: E402
from vvexport.manual_sdpa import ManualSDPA  # noqa: E402
_q.SDPA = ManualSDPA

from export_llm import load_base  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.primitives.macos.cache import KVCache  # noqa: E402

VOICE = "/Users/amosgyamfi/Desktop/VibeVoice/VibeVoice/demo/voices/streaming_model/en-Carter_man.pt"
WINDOW = [9707, 504, 389, 3671, 9518]
MAXCTX = 256


class StatelessBase(nn.Module):
    """base_lm with cache passed in and the fetched slice used (no mutation/state)."""

    def __init__(self, cfg, model_dir):
        super().__init__()
        self.inner = load_base(cfg, model_dir)

    def forward(self, input_ids, position_ids, k_cache, v_cache):
        return self.inner(input_ids, position_ids, k_cache, v_cache)


def hf_stack(cfg, model_dir):
    hf = HFQwen2(copy.deepcopy(cfg)).eval().float()
    src = common.load_subtree("model.language_model.", model_dir=model_dir, dtype=torch.float32)
    hf.embed_tokens.load_state_dict({"weight": src["embed_tokens.weight"]})
    hf.norm = nn.Identity()
    hf.load_state_dict({k: v for k, v in src.items() if not k.startswith("embed_tokens")}, strict=False)
    return hf


async def run_graph(model, ids, pos, k, v):
    from coreai.runtime import NDArray
    ref = {"input_ids": ids, "position_ids": pos, "k_cache": torch.from_numpy(k),
           "v_cache": torch.from_numpy(v)}
    prog = export_to_coreai(model, ref, dynamic_shapes=None,
                            input_names=("input_ids", "position_ids", "k_cache", "v_cache"),
                            output_names=("hidden",))
    prog.optimize()
    with tempfile.TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = prog.save_asset(Path(tmp))
        async with asset.executable() as aimodel:
            fn = aimodel.load_function("main")
            res = await fn({"input_ids": NDArray(data=ids.contiguous()),
                            "position_ids": NDArray(data=pos.contiguous()),
                            "k_cache": NDArray(data=k), "v_cache": NDArray(data=v)})
    return torch.from_numpy(np.array(next(iter(res.values())).numpy()))


def main() -> None:
    model_dir = common.DEFAULT_MODEL_DIR
    cfg = copy.deepcopy(common.load_config(model_dir).decoder_config)
    cfg.num_hidden_layers = 4
    cfg.max_position_embeddings = MAXCTX
    hf = hf_stack(cfg, model_dir)
    model = StatelessBase(cfg, model_dir).eval()
    q = len(WINDOW)
    ids = torch.tensor([WINDOW], dtype=torch.int32)
    NKV = cfg.num_key_value_heads

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        voice = torch.load(VOICE, map_location="cpu", weights_only=False)
    pkv = voice["lm"]["past_key_values"]
    L = pkv.key_cache[0].shape[2]

    # offset L: the cache must be sized seq_len = L + q for the fetch to span it.
    seq = L + q
    posL = torch.arange(0, seq, dtype=torch.int32).unsqueeze(0)
    kL = np.zeros((cfg.num_hidden_layers, 1, NKV, seq, 64), np.float32)
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
        eagerL = model(ids, posL, torch.from_numpy(kL), torch.from_numpy(vL))
    gL = asyncio.run(run_graph(model, ids, posL, kL, vL))
    print(f"offsetL eager vs HF = {common.psnr(refL, eagerL):.2f} dB")
    print(f"offsetL graph vs HF = {common.psnr(refL, gL):.2f} dB  "
          f"(HF {refL[0,-1].norm():.3f} / graph {gL[0,-1].norm():.3f} / eager {eagerL[0,-1].norm():.3f})")
    print(f"offsetL graph vs eager = {common.psnr(eagerL, gL):.2f} dB")


if __name__ == "__main__":
    main()
