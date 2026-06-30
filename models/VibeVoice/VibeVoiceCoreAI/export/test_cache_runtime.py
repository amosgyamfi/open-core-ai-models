#!/usr/bin/env python
"""Isolate the KV-cache *state* op through the Core AI runtime.

Graph: write k,v into the cache state at `offset`, fetch [0:seq_len], return the
fetched k. Compare runtime vs torch eager. Exercises mutable_slice_update + state
without any attention/rope.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORT_DIR))

from coreai_models.export._constants import KEY_CACHE_NAME, VALUE_CACHE_NAME  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.primitives.macos.cache import KVCache  # noqa: E402

NL, NKV, MAXCTX, HD = 1, 2, 64, 8
Q, OFFSET = 3, 5


class CacheProbe(nn.Module):
    def forward(self, new_k, new_v, position_ids, k_cache, v_cache):
        cache = KVCache(k_cache, v_cache)
        seq_len = position_ids.shape[-1]
        query_len = new_k.shape[-2]
        offset = seq_len - query_len
        k, v = cache.update_and_fetch(0, offset, new_k, new_v, seq_len=seq_len, query_len=query_len)
        return k


def main() -> None:
    torch.manual_seed(0)
    new_k = torch.randn(1, NKV, Q, HD)
    new_v = torch.randn(1, NKV, Q, HD)
    pos = torch.arange(0, OFFSET + Q, dtype=torch.int32).unsqueeze(0)

    k0 = torch.zeros(NL, 1, NKV, MAXCTX, HD)
    v0 = torch.zeros(NL, 1, NKV, MAXCTX, HD)
    seed_k = torch.randn(NKV, OFFSET, HD)
    seed_v = torch.randn(NKV, OFFSET, HD)

    # eager
    ke = k0.clone(); ve = v0.clone()
    ke[0, 0, :, :OFFSET, :] = seed_k
    ve[0, 0, :, :OFFSET, :] = seed_v
    with torch.no_grad():
        eager = CacheProbe()(new_k.clone(), new_v.clone(), pos, ke, ve)
    print(f"eager fetched k: shape={tuple(eager.shape)} norm={eager.norm():.4f}")

    out = asyncio.run(_run(new_k, new_v, pos, seed_k, seed_v))
    o = torch.from_numpy(out)
    print(f"rt    fetched k: shape={tuple(o.shape)} norm={o.norm():.4f}")
    # align shapes
    e = eager.float()
    if o.shape == e.shape:
        diff = (o - e).abs().max().item()
        print(f"max abs diff (runtime vs eager) = {diff:.6f}")
    else:
        print("shape mismatch", o.shape, e.shape)


async def _run(new_k, new_v, pos, seed_k, seed_v):
    from coreai.runtime import NDArray
    model = CacheProbe()
    k0 = torch.zeros(NL, 1, NKV, MAXCTX, HD)
    v0 = torch.zeros(NL, 1, NKV, MAXCTX, HD)
    ref_inputs = {"new_k": new_k, "new_v": new_v, "position_ids": pos,
                  "k_cache": k0, "v_cache": v0}
    program = export_to_coreai(
        model, ref_inputs, dynamic_shapes=None,
        input_names=("new_k", "new_v", "position_ids"), output_names=("k",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
    )
    program.optimize()
    k = np.zeros((NL, 1, NKV, MAXCTX, HD), np.float32)
    v = np.zeros((NL, 1, NKV, MAXCTX, HD), np.float32)
    k[0, 0, :, :OFFSET, :] = seed_k.numpy()
    v[0, 0, :, :OFFSET, :] = seed_v.numpy()
    with tempfile.TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = program.save_asset(Path(tmp))
        async with asset.executable() as aimodel:
            fn = aimodel.load_function("main")
            res = await fn(
                {"new_k": NDArray(data=new_k.contiguous()),
                 "new_v": NDArray(data=new_v.contiguous()),
                 "position_ids": NDArray(data=pos.contiguous())},
                state={KEY_CACHE_NAME: NDArray(data=k), VALUE_CACHE_NAME: NDArray(data=v)},
            )
    return np.array(next(iter(res.values())).numpy())


if __name__ == "__main__":
    main()
