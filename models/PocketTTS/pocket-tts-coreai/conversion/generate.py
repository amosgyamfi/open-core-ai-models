# Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
"""Full Pocket-TTS synthesis on the Core AI engine: the host AR loop wires the three exported
.aimodel bundles (backbone / flow / mimi) plus tiny host glue (token embed, latent projection,
BOS, EOS head, noise) — exactly what the Swift host does on device.

  python generate.py --reproduce-oracle [--dtype fp16|fp32]   # feed oracle noise, gate vs oracle
  python generate.py --text "..." --voice alba [--out out.wav] [--seed 0]
"""
from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import coreai.runtime as rt
from pockettts_coreai import load_components
from verify_eager import seed_caches_from_voice

ART = Path(__file__).resolve().parent / "artifacts"
BUNDLES = ART / "bundles"
CACHE_LEN = 2048


def _np(dtype):
    return np.float16 if dtype == "fp16" else np.float32


def na(x, dt):
    return rt.NDArray(np.ascontiguousarray(np.asarray(x, dtype=dt)))


async def load(name):
    for opt in (rt.SpecializationOptions.cpu_only(), rt.SpecializationOptions.default()):
        try:
            m = await rt.AIModel.load(str(BUNDLES / name / f"{name}.aimodel"), opt)
            return m, m.load_function("main")
        except RuntimeError:
            continue
    raise RuntimeError(f"cannot load {name}")


class HostGlue:
    """Tiny host-side ops (fp32 math), mirroring the Swift Accelerate path."""

    def __init__(self, c):
        self.embed = c.embed.float()
        self.input_linear = c.input_linear.float()      # [D, ldim]
        self.bos_emb = c.bos_emb.float()                # [ldim]
        self.out_eos_w = c.out_eos_w.float()            # [1, D]
        self.out_eos_b = c.out_eos_b.float()            # [1]
        self.eos_threshold = c.eos_threshold
        self.ldim = c.ldim

    def text_embed(self, token_id: int) -> torch.Tensor:
        return self.embed[token_id].view(1, 1, -1)

    def project_latent(self, latent: torch.Tensor | None) -> torch.Tensor:
        # NaN backbone input -> bos_emb (first step); then input_linear(latent).
        seq = self.bos_emb if latent is None else latent.view(-1)
        return F.linear(seq, self.input_linear).view(1, 1, -1)

    def is_eos(self, hidden: torch.Tensor) -> bool:
        logit = F.linear(hidden.view(1, -1), self.out_eos_w, self.out_eos_b)
        return bool((logit > self.eos_threshold).item())


async def run(dtype, text, voice, seed, reproduce, out_path):
    DT = _np(dtype)
    c = load_components(cache_len=CACHE_LEN)
    glue = HostGlue(c)
    o = np.load(ART / "oracle.npz") if reproduce else None

    bb_m, bb = await load(f"pockettts_backbone_{dtype}")
    fl_m, fl = await load(f"pockettts_flow_{dtype}")
    mi_m, mi = await load(f"pockettts_mimi_{dtype}")

    # voice KV state
    kc_t, vc_t, voice_len = seed_caches_from_voice(c, voice, CACHE_LEN)
    state = {"keyCache": na(kc_t.numpy(), DT), "valueCache": na(vc_t.numpy(), DT)}

    async def step(input_emb, pos):
        res = await bb(inputs={"inputs_embeds": na(input_emb.numpy(), DT),
                               "pos": na([pos], np.int32)}, state=state)
        return torch.tensor(np.asarray(res["hidden"].numpy(), np.float32))

    # ---- text prefill (prefill-via-decode) ----
    if reproduce:
        tokens = o["text_tokens"].reshape(-1).tolist()
    else:
        prepared = c.src.flow_lm.conditioner.prepare(_prepare(c, text))
        tokens = prepared.tokens.reshape(-1).tolist()
    pos = voice_len
    for tid in tokens:
        await step(glue.text_embed(int(tid)), pos)
        pos += 1

    # ---- autoregressive decode ----
    torch.manual_seed(seed)
    latents: list[torch.Tensor] = []
    prev_latent: torch.Tensor | None = None
    max_steps = (o["n_mimi"][0] if reproduce else 400)
    eos_step = None
    frames_after_eos = 2
    n_target = int(o["n_mimi"][0]) if reproduce else None
    step_i = 0
    while True:
        hidden = await step(glue.project_latent(prev_latent), pos)
        pos += 1
        cond = hidden.view(1, -1)
        if reproduce:
            z = torch.tensor(o[f"flow_noise__{_oracle_flow_index(o, step_i)}"], dtype=torch.float32)
        else:
            std = c.temp ** 0.5
            z = torch.randn(1, glue.ldim) * std
        res = await fl(inputs={"cond": na(cond.numpy(), DT), "z": na(z.numpy(), DT)})
        latent = torch.tensor(np.asarray(res["latent"].numpy(), np.float32))
        if reproduce:
            latents.append(latent)
            prev_latent = latent
            step_i += 1
            if len(latents) >= n_target:
                break
        else:
            if glue.is_eos(hidden) and eos_step is None:
                eos_step = step_i
            if eos_step is not None and step_i >= eos_step + frames_after_eos:
                break
            latents.append(latent)
            prev_latent = latent
            step_i += 1
            if step_i >= max_steps:
                break

    latseq = torch.cat([x.view(1, 1, glue.ldim) for x in latents], dim=1)
    res = await mi(inputs={"latents": na(latseq.numpy(), DT)})
    wav = np.asarray(res["wav"].numpy(), np.float32).reshape(-1)

    if reproduce:
        ref = o["audio"].reshape(-1)
        n = min(len(wav), len(ref))
        c0 = float(F.cosine_similarity(torch.tensor(wav[:n]), torch.tensor(ref[:n]), dim=0))
        A = _magspec(wav[:n]); B = _magspec(ref[:n])
        sc = float(F.cosine_similarity(A.reshape(-1), B.reshape(-1), dim=0))
        ok = sc >= 0.99
        print(f">>> END-TO-END engine[{dtype}] vs oracle: frames={len(latents)} "
              f"raw_cos={c0:.5f} magspec_cos={sc:.5f} -> {'PASS' if ok else 'CHECK'}")
    else:
        _write_wav(out_path, wav, c.sample_rate)
        print(f"[generate] '{text[:40]}...' voice={voice} -> {len(wav)} samp "
              f"= {len(wav)/c.sample_rate:.2f}s -> {out_path}")


def _prepare(c, text):
    from pocket_tts.models.tts_model import prepare_text_prompt
    t, _ = prepare_text_prompt(text, c.src.pad_with_spaces_for_short_inputs, c.src.remove_semicolons)
    return t


def _oracle_flow_index(o, decode_i):
    n_mimi = int(o["n_mimi"][0]); n_flow = int(o["n_flow"][0])
    n_prefill = n_flow - n_mimi - 1
    return n_prefill + decode_i


def _magspec(x, n=512, hop=128):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.stft(x, n_fft=n, hop_length=hop, window=torch.hann_window(n),
                      return_complex=True).abs()


def _write_wav(path, wav, sr):
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(pcm.tobytes())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--text", default="Hello world, this is a test of pocket text to speech.")
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="engine_out.wav")
    ap.add_argument("--reproduce-oracle", action="store_true")
    a = ap.parse_args()
    await run(a.dtype, a.text, a.voice, a.seed, a.reproduce_oracle, Path(a.out))


if __name__ == "__main__":
    asyncio.run(main())
