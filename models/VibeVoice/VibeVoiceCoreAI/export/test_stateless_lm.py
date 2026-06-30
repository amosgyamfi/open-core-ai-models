#!/usr/bin/env python
"""Stateless base_lm (concat cache + manual SDPA) through the Core AI runtime
vs HF at the incremental (offset L) step that real VibeVoice uses."""
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
from vvexport.stateless_qwen2 import StatelessBaseLM  # noqa: E402
from export_llm import _remap_layer  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402

VOICE = "/Users/amosgyamfi/Desktop/VibeVoice/VibeVoice/demo/voices/streaming_model/en-Carter_man.pt"
WINDOW = [9707, 504, 389, 3671, 9518]


def load_stateless_base(cfg, model_dir):
    src = common.load_subtree("model.language_model.", model_dir=model_dir, dtype=torch.float32)
    sd = {"embed_tokens.weight": src["embed_tokens.weight"]}
    for i in range(cfg.num_hidden_layers):
        sd.update(_remap_layer(src, i, f"layers.{i}"))
    m = StatelessBaseLM(cfg).eval().float()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rope" not in k]
    assert not missing and not unexpected, f"missing={missing[:4]} unexpected={unexpected[:4]}"
    return m


def hf_stack(cfg, model_dir):
    hf = HFQwen2(copy.deepcopy(cfg)).eval().float()
    src = common.load_subtree("model.language_model.", model_dir=model_dir, dtype=torch.float32)
    hf.embed_tokens.load_state_dict({"weight": src["embed_tokens.weight"]})
    hf.norm = nn.Identity()
    hf.load_state_dict({k: v for k, v in src.items() if not k.startswith("embed_tokens")}, strict=False)
    return hf


async def run_graph(model, ids, L0, pk, pv):
    """Feed the window one token at a time (q=1); only L is dynamic."""
    from coreai.runtime import NDArray
    sample_ids = ids[:, :1]
    sample_pos = torch.tensor([[L0]], dtype=torch.int32)
    ref = {"input_ids": sample_ids, "position_ids": sample_pos, "past_k": pk, "past_v": pv}
    seqdim = torch.export.Dim("L", min=2, max=4091)
    dyn = {"input_ids": None, "position_ids": None,
           "past_k": {3: seqdim}, "past_v": {3: seqdim}}
    prog = export_to_coreai(model, ref, dynamic_shapes=dyn,
                            input_names=("input_ids", "position_ids", "past_k", "past_v"),
                            output_names=("hidden", "new_k", "new_v"))
    prog.optimize()
    q = ids.shape[1]
    with tempfile.TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = prog.save_asset(Path(tmp))
        async with asset.executable() as aimodel:
            fn = aimodel.load_function("main")
            ck, cv = pk, pv
            last = None
            for t in range(q):
                tok = ids[:, t:t + 1]
                pos = torch.tensor([[L0 + t]], dtype=torch.int32)
                res = await fn({"input_ids": NDArray(data=tok.contiguous()),
                                "position_ids": NDArray(data=pos.contiguous()),
                                "past_k": NDArray(data=ck.contiguous()),
                                "past_v": NDArray(data=cv.contiguous())})
                out = {k: torch.from_numpy(np.array(v.numpy())) for k, v in res.items()}
                ck = torch.cat([ck, out["new_k"]], dim=3)
                cv = torch.cat([cv, out["new_v"]], dim=3)
                last = out["hidden"]
    return last


def main() -> None:
    model_dir = common.DEFAULT_MODEL_DIR
    cfg = copy.deepcopy(common.load_config(model_dir).decoder_config)
    cfg.num_hidden_layers = 4
    cfg.max_position_embeddings = 4096
    q = len(WINDOW)
    ids = torch.tensor([WINDOW], dtype=torch.int32)

    model = load_stateless_base(cfg, model_dir)
    hf = hf_stack(cfg, model_dir)

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        voice = torch.load(VOICE, map_location="cpu", weights_only=False)
    pkv = voice["lm"]["past_key_values"]
    L = pkv.key_cache[0].shape[2]
    pk = torch.stack([pkv.key_cache[i].float() for i in range(cfg.num_hidden_layers)], 0)  # (nl,1,nkv,L,hd)
    pv = torch.stack([pkv.value_cache[i].float() for i in range(cfg.num_hidden_layers)], 0)
    pos = torch.arange(L, L + q, dtype=torch.int32).unsqueeze(0)

    hfc = DynamicCache()
    for i in range(cfg.num_hidden_layers):
        hfc.update(pkv.key_cache[i].clone().float(), pkv.value_cache[i].clone().float(), i)
    with torch.no_grad():
        ref = hf(inputs_embeds=hf.embed_tokens(ids.long()),
                 position_ids=torch.arange(L, L + q).unsqueeze(0).long(),
                 past_key_values=hfc, use_cache=True).last_hidden_state
        eager, _, _ = model(ids, pos, pk, pv)
    g_last = asyncio.run(run_graph(model, ids, L, pk, pv))
    print(f"eager vs HF    = {common.psnr(ref, eager):.2f} dB")
    print(f"graph(last) vs HF    = {common.psnr(ref[:, -1:], g_last):.2f} dB  "
          f"(HF {ref[0,-1].norm():.3f} / graph {g_last[0,-1].norm():.3f})")
    print(f"graph(last) vs eager = {common.psnr(eager[:, -1:], g_last):.2f} dB")


if __name__ == "__main__":
    main()
