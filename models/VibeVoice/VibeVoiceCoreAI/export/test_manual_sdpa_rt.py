#!/usr/bin/env python
"""Isolate ManualSDPA through the Core AI runtime vs torch eager."""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

EXPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPORT_DIR))

from coreai_models.export.macos import export_to_coreai  # noqa: E402
from vvexport.manual_sdpa import ManualSDPA  # noqa: E402

B, NH, NKV, Q, K, HD = 1, 8, 4, 10, 10, 8


def main() -> None:
    torch.manual_seed(0)
    q = torch.randn(B, NH, Q, HD)
    k = torch.randn(B, NKV, K, HD)
    v = torch.randn(B, NKV, K, HD)
    m = ManualSDPA(is_causal=True).eval()
    with torch.no_grad():
        eager = m(q, k, v)
    print(f"eager norm={eager.norm():.4f}")
    rt = torch.from_numpy(asyncio.run(_run(m, q, k, v)))
    print(f"rt    norm={rt.norm():.4f}")
    print(f"max abs diff = {(rt - eager).abs().max().item():.6f}")


async def _run(m, q, k, v):
    from coreai.runtime import NDArray
    ref = {"query": q, "key": k, "value": v}
    prog = export_to_coreai(m, ref, dynamic_shapes=None,
                            input_names=("query", "key", "value"), output_names=("out",))
    prog.optimize()
    with tempfile.TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = prog.save_asset(Path(tmp))
        async with asset.executable() as aimodel:
            fn = aimodel.load_function("main")
            res = await fn({"query": NDArray(data=q.contiguous()),
                            "key": NDArray(data=k.contiguous()),
                            "value": NDArray(data=v.contiguous())})
    return np.array(next(iter(res.values())).numpy())


if __name__ == "__main__":
    main()
