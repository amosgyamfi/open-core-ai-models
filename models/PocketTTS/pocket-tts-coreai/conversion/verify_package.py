# Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
"""Drive synthesis using ONLY the shipped package (manifest.json + *.bin + tokenizer.model +
.aimodel bundles). No `pocket_tts` import — this is the exact contract the Swift host implements.

  python verify_package.py --reproduce-oracle          # gate full pipeline vs oracle
  python verify_package.py --text "..." --voice alba --out pkg_out.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import coreai.runtime as rt

HERE = Path(__file__).resolve().parent
PKG = HERE / "dist" / "PocketTTSCoreAI"
ART = HERE / "artifacts"


class Package:
    def __init__(self, root: Path):
        self.root = root
        self.m = json.loads((root / "manifest.json").read_text())
        self.s = self.m["scalars"]
        self.np = np.float16 if self.m["dtype"] == "fp16" else np.float32

    def tensor(self, spec) -> torch.Tensor:
        raw = np.frombuffer((self.root / spec["file"]).read_bytes(), dtype=np.float32)
        return torch.from_numpy(raw.reshape(spec["shape"]).copy())

    def glue(self, key) -> torch.Tensor:
        return self.tensor(self.m["glue"][key])

    def voice(self, name):
        v = self.m["voices"][name]
        return self.tensor(v["key"]), self.tensor(v["value"]), int(v["offset"])

    async def load(self, short):
        path = str(self.root / self.m["bundles"][short])
        for opt in (rt.SpecializationOptions.cpu_only(), rt.SpecializationOptions.default()):
            try:
                mdl = await rt.AIModel.load(path, opt)
                return mdl, mdl.load_function("main")
            except RuntimeError:
                continue
        raise RuntimeError(f"cannot load {short}")


def na(x, dt):
    return rt.NDArray(np.ascontiguousarray(np.asarray(x, dtype=dt)))


async def run(reproduce, text, voice, seed, out_path):
    pkg = Package(PKG)
    DT, S = pkg.np, pkg.s
    CL = S["cache_len"]
    embed = pkg.glue("embed")
    input_linear = pkg.glue("input_linear")
    bos_emb = pkg.glue("bos_emb")
    eos_w, eos_b = pkg.glue("out_eos_w"), pkg.glue("out_eos_b")
    eos_thr = S["eos_threshold"]

    bb_m, bb = await pkg.load("backbone")
    fl_m, fl = await pkg.load("flow")
    mi_m, mi = await pkg.load("mimi")

    # voice KV -> full [L,1,H,CL,Dh]
    vk, vv, off = pkg.voice(voice)
    L, _, H, _, Dh = vk.shape
    kc = torch.zeros(L, 1, H, CL, Dh); kc[:, :, :, :off, :] = vk
    vc = torch.zeros(L, 1, H, CL, Dh); vc[:, :, :, :off, :] = vv
    state = {"keyCache": na(kc.numpy(), DT), "valueCache": na(vc.numpy(), DT)}

    o = np.load(ART / "oracle.npz") if reproduce else None

    async def step(emb, pos):
        r = await bb(inputs={"inputs_embeds": na(emb.numpy(), DT), "pos": na([pos], np.int32)},
                     state=state)
        return torch.tensor(np.asarray(r["hidden"].numpy(), np.float32))

    # text prefill
    if reproduce:
        tokens = o["text_tokens"].reshape(-1).tolist()
    else:
        import sentencepiece
        sp = sentencepiece.SentencePieceProcessor(str(PKG / pkg.m["tokenizer"]))
        tokens = sp.encode(_prep(text), out_type=int)
    pos = off
    for tid in tokens:
        await step(embed[int(tid)].view(1, 1, -1), pos)
        pos += 1

    # decode loop
    torch.manual_seed(seed)
    n_mimi = int(o["n_mimi"][0]) if reproduce else None
    n_flow = int(o["n_flow"][0]) if reproduce else None
    n_prefill = (n_flow - n_mimi - 1) if reproduce else 0
    latents, prev = [], None
    eos_step, di = None, 0
    while True:
        seq = bos_emb if prev is None else prev.view(-1)
        emb = F.linear(seq, input_linear).view(1, 1, -1)
        hidden = await step(emb, pos); pos += 1
        cond = hidden.view(1, -1)
        if reproduce:
            z = torch.tensor(o[f"flow_noise__{n_prefill + di}"], dtype=torch.float32)
        else:
            z = torch.randn(1, S["ldim"]) * (S["temp"] ** 0.5)
        r = await fl(inputs={"cond": na(cond.numpy(), DT), "z": na(z.numpy(), DT)})
        latent = torch.tensor(np.asarray(r["latent"].numpy(), np.float32))
        if reproduce:
            latents.append(latent); prev = latent; di += 1
            if len(latents) >= n_mimi:
                break
        else:
            logit = F.linear(hidden.view(1, -1), eos_w, eos_b)
            if bool((logit > eos_thr).item()) and eos_step is None:
                eos_step = di
            if eos_step is not None and di >= eos_step + 2:
                break
            latents.append(latent); prev = latent; di += 1
            if di >= 400:
                break

    latseq = torch.cat([x.view(1, 1, S["ldim"]) for x in latents], dim=1)
    r = await mi(inputs={"latents": na(latseq.numpy(), DT)})
    wav = np.asarray(r["wav"].numpy(), np.float32).reshape(-1)

    if reproduce:
        ref = o["audio"].reshape(-1)
        n = min(len(wav), len(ref))
        def mag(x):
            return torch.stft(torch.tensor(x), 512, 128, window=torch.hann_window(512),
                              return_complex=True).abs().reshape(-1)
        sc = float(F.cosine_similarity(mag(wav[:n]), mag(ref[:n]), dim=0))
        print(f">>> PACKAGE engine[{pkg.m['dtype']}] vs oracle: frames={len(latents)} "
              f"magspec_cos={sc:.5f} -> {'PASS' if sc >= 0.99 else 'CHECK'}")
    else:
        pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(out_path), "w") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(pkg.m["sample_rate"])
            f.writeframes(pcm.tobytes())
        print(f"[package] '{text[:40]}' voice={voice} -> {len(wav)/pkg.m['sample_rate']:.2f}s -> {out_path}")


def _prep(text: str) -> str:
    return text if text.endswith((".", "!", "?")) else text + "."


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduce-oracle", action="store_true")
    ap.add_argument("--text", default="Hello world, this is a test of pocket text to speech.")
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pkg_out.wav")
    a = ap.parse_args()
    await run(a.reproduce_oracle, a.text, a.voice, a.seed, Path(a.out))


if __name__ == "__main__":
    asyncio.run(main())
