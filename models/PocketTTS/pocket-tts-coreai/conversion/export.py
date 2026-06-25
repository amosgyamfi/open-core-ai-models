# Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
"""Export the three Pocket-TTS sub-networks to Core AI `.aimodel` bundles and gate each on the
Core AI runtime against the captured oracle.

  python export.py backbone [--dtype fp16|fp32] [--cache-len 2048]
  python export.py flow     [--dtype fp16|fp32]
  python export.py mimi     [--dtype fp16|fp32]
  python export.py all      [--dtype fp16|fp32]

Bundles land in artifacts/bundles/<name>/<name>.aimodel. Gate criterion: cosine >= 0.99 on
cpu_only (numeric-parity device), per the zoo conversion guide.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import coreai.runtime as rt
from coreai_torch import TorchConverter, get_decomp_table
from pockettts_coreai import load_components
from verify_eager import seed_caches_from_voice

ART = Path(__file__).resolve().parent / "artifacts"
BUNDLES = ART / "bundles"
CACHE_LEN = 2048


def _np(dtype: str):
    return np.float16 if dtype == "fp16" else np.float32


def _td(dtype: str):
    return torch.float16 if dtype == "fp16" else torch.float32


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return float(F.cosine_similarity(a, b, dim=0))


def convert(module, sample_kwargs, input_names, output_names, state_names, dynamic_shapes, out_dir):
    module = module.eval()
    ep = torch.export.export(
        module, args=(), kwargs=sample_kwargs, dynamic_shapes=dynamic_shapes
    ).run_decompositions(get_decomp_table())
    prog = TorchConverter().add_exported_program(
        ep,
        input_names=input_names,
        output_names=output_names,
        state_names=state_names,
    ).to_coreai()
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


async def load_fn(aim: Path):
    # Prefer cpu_only for numeric parity; some graphs (large masked-write KV state) don't compile
    # on the CPU backend -> fall back to the default (GPU) specialization.
    for opt in (rt.SpecializationOptions.cpu_only(), rt.SpecializationOptions.default()):
        try:
            model = await rt.AIModel.load(str(aim), opt)
            return model, model.load_function("main")
        except RuntimeError:
            continue
    raise RuntimeError(f"could not load {aim} on cpu_only or default")


# ----------------------------------------------------------------------------- backbone
async def export_backbone(dtype: str, cache_len: int) -> None:
    DT, NP = _td(dtype), _np(dtype)
    c = load_components(cache_len=cache_len)
    bb = c.backbone.to(DT).eval()
    L, H, Dh, D = c.num_layers, c.num_heads, c.dim_per_head, c.d_model
    sample = {
        "inputs_embeds": torch.zeros(1, 1, D, dtype=DT),
        "pos": torch.tensor([7], dtype=torch.int32),
        "k_cache": torch.zeros(L, 1, H, cache_len, Dh, dtype=DT),
        "v_cache": torch.zeros(L, 1, H, cache_len, Dh, dtype=DT),
    }
    out = BUNDLES / f"pockettts_backbone_{dtype}"
    aim = convert(
        bb, sample,
        input_names=("inputs_embeds", "pos"),
        output_names=("hidden",),
        state_names=("keyCache", "valueCache"),
        dynamic_shapes=None,
        out_dir=out,
    )
    print(f"  saved {aim}")

    # ---- engine gate vs oracle (seed voice KV, replay text prefill + decode positions) ----
    o = np.load(ART / "oracle.npz")
    kc_t, vc_t, _ = seed_caches_from_voice(c, "alba", cache_len)
    model, fn = await load_fn(aim)
    kc = rt.NDArray(kc_t.to(DT).numpy().astype(NP))
    vc = rt.NDArray(vc_t.to(DT).numpy().astype(NP))
    state = {"keyCache": kc, "valueCache": vc}
    cs = []
    n_bb = int(o["n_bb"][0])
    offs = o["bb_offset"]
    for i in range(n_bb):
        x = o[f"bb_in__{i}"].astype(NP)
        ref = o[f"bb_hidden__{i}"]
        off = int(offs[i])
        for j in range(x.shape[1]):
            res = await fn(
                inputs={
                    "inputs_embeds": rt.NDArray(np.ascontiguousarray(x[:, j : j + 1])),
                    "pos": rt.NDArray(np.array([off + j], np.int32)),
                },
                state=state,
            )
            cs.append(cos(res["hidden"].numpy(), ref[:, j : j + 1]))
    lo = min(cs)
    print(f">>> backbone[{dtype}] engine vs oracle: steps={len(cs)} min_cos={lo:.6f} "
          f"-> {'PASS' if lo >= 0.99 else 'FAIL'}")


# ----------------------------------------------------------------------------- flow
async def export_flow(dtype: str) -> None:
    DT, NP = _td(dtype), _np(dtype)
    c = load_components()
    flow = c.flow.to(DT).eval()
    sample = {
        "cond": torch.zeros(1, c.d_model, dtype=DT),
        "z": torch.zeros(1, c.ldim, dtype=DT),
    }
    out = BUNDLES / f"pockettts_flow_{dtype}"
    aim = convert(
        flow, sample,
        input_names=("cond", "z"),
        output_names=("latent",),
        state_names=None,
        dynamic_shapes=None,
        out_dir=out,
    )
    print(f"  saved {aim}")
    o = np.load(ART / "oracle.npz")
    model, fn = await load_fn(aim)
    cs = []
    for i in range(int(o["n_flow"][0])):
        res = await fn(inputs={
            "cond": rt.NDArray(np.ascontiguousarray(o[f"flow_cond__{i}"].astype(NP))),
            "z": rt.NDArray(np.ascontiguousarray(o[f"flow_noise__{i}"].astype(NP))),
        })
        cs.append(cos(res["latent"].numpy(), o[f"flow_latent__{i}"]))
    lo = min(cs)
    print(f">>> flow[{dtype}] engine vs oracle: steps={len(cs)} min_cos={lo:.6f} "
          f"-> {'PASS' if lo >= 0.99 else 'FAIL'}")


# ----------------------------------------------------------------------------- mimi
async def export_mimi(dtype: str) -> None:
    DT, NP = _td(dtype), _np(dtype)
    c = load_components()
    mimi = c.mimi.to(DT).eval()
    o = np.load(ART / "oracle.npz")
    n_mimi = int(o["n_mimi"][0])
    n_flow = int(o["n_flow"][0])
    n_prefill = n_flow - n_mimi - 1
    lats = [torch.tensor(o[f"flow_latent__{n_prefill + i}"]) for i in range(n_mimi)]
    latseq = torch.cat([lat.view(1, 1, c.ldim) for lat in lats], dim=1).to(DT)

    T = torch.export.Dim("T", min=1, max=4096)
    sample = {"latents": torch.zeros(1, 8, c.ldim, dtype=DT)}
    out = BUNDLES / f"pockettts_mimi_{dtype}"
    aim = convert(
        mimi, sample,
        input_names=("latents",),
        output_names=("wav",),
        state_names=None,
        dynamic_shapes={"latents": {1: T}},
        out_dir=out,
    )
    print(f"  saved {aim}")
    model, fn = await load_fn(aim)
    res = await fn(inputs={"latents": rt.NDArray(np.ascontiguousarray(latseq.numpy().astype(NP)))})
    wav = np.asarray(res["wav"].numpy()).reshape(-1)
    ref = o["audio"].reshape(-1)
    n = min(len(wav), len(ref))
    cc = cos(wav[:n], ref[:n])
    print(f">>> mimi[{dtype}] engine vs oracle: frames={n_mimi} wav={len(wav)} cos={cc:.6f} "
          f"-> {'PASS' if cc >= 0.99 else 'FAIL'}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["backbone", "flow", "mimi", "all"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--cache-len", type=int, default=CACHE_LEN)
    a = ap.parse_args()
    if a.which in ("backbone", "all"):
        await export_backbone(a.dtype, a.cache_len)
    if a.which in ("flow", "all"):
        await export_flow(a.dtype)
    if a.which in ("mimi", "all"):
        await export_mimi(a.dtype)


if __name__ == "__main__":
    asyncio.run(main())
