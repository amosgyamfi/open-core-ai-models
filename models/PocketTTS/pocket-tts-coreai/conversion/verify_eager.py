# Community port — NOT an Apple model.
"""Eager-PyTorch parity of the re-authored modules vs the captured oracle.

Validates the re-authoring (KV/RoPE/mask/stateless-conv rewrites) independently of the Core AI
conversion. Run AFTER capture_oracle.py.

  python verify_eager.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pockettts_coreai import load_components

ART = Path(__file__).resolve().parent / "artifacts"
CACHE_LEN = 2048


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return float(F.cosine_similarity(a, b, dim=0))


def maxdiff(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return float((a - b).abs().max())


def seed_caches_from_voice(c, voice: str, cache_len: int):
    """Transcode a pocket-tts predefined-voice KV state into our [L,1,H,CL,Dh] layout.

    The pocket-tts per-layer cache is [2, B, T, H, Dh] (index 0 = rotated K, 1 = V), with
    ``offset`` == number of written (prompt) positions.
    """
    L, H, Dh = c.num_layers, c.num_heads, c.dim_per_head
    kc = torch.zeros(L, 1, H, cache_len, Dh, dtype=torch.float32)
    vc = torch.zeros(L, 1, H, cache_len, Dh, dtype=torch.float32)
    state = c.src.get_state_for_audio_prompt(voice)
    start = 0
    for li in range(L):
        ms = state[f"transformer.layers.{li}.self_attn"]
        cache = ms["cache"]            # [2,1,T,H,Dh]
        off = int(ms["offset"].view(-1)[0].item())
        k = cache[0, 0, :off].permute(1, 0, 2)   # [H,T,Dh]
        v = cache[1, 0, :off].permute(1, 0, 2)
        kc[li, 0, :, :off, :] = k
        vc[li, 0, :, :off, :] = v
        start = off
    return kc, vc, start


@torch.no_grad()
def main() -> None:
    o = np.load(ART / "oracle.npz")
    c = load_components(cache_len=CACHE_LEN)

    # ---------- backbone (seed predefined-voice KV first) ----------
    L, H, Dh = c.num_layers, c.num_heads, c.dim_per_head
    kc, vc, voice_len = seed_caches_from_voice(c, "alba", CACHE_LEN)
    n_bb = int(o["n_bb"][0])
    offsets = o["bb_offset"]
    bb_cos, bb_md = [], []
    for i in range(n_bb):
        x = torch.tensor(o[f"bb_in__{i}"])          # [1,S,D]
        ref = torch.tensor(o[f"bb_hidden__{i}"])    # [1,S,D]
        off = int(offsets[i])
        S = x.shape[1]
        for j in range(S):
            pos = torch.tensor(off + j, dtype=torch.int32)
            out = c.backbone(x[:, j : j + 1], pos, kc, vc)  # [1,1,D]
            bb_cos.append(cos(out, ref[:, j : j + 1]))
            bb_md.append(maxdiff(out, ref[:, j : j + 1]))
    print(f"[backbone] steps={len(bb_cos)} min_cos={min(bb_cos):.6f} max_maxdiff={max(bb_md):.3e}")

    # ---------- flow decoder ----------
    n_flow = int(o["n_flow"][0])
    fl_cos, fl_md = [], []
    for i in range(n_flow):
        cond = torch.tensor(o[f"flow_cond__{i}"])   # [1,D]
        z = torch.tensor(o[f"flow_noise__{i}"])     # [1,ldim]
        ref = torch.tensor(o[f"flow_latent__{i}"])  # [1,ldim]
        out = c.flow(cond, z)
        fl_cos.append(cos(out, ref))
        fl_md.append(maxdiff(out, ref))
    print(f"[flow]     steps={len(fl_cos)} min_cos={min(fl_cos):.6f} max_maxdiff={max(fl_md):.3e}")

    # ---------- mimi decoder (full-sequence) ----------
    n_mimi = int(o["n_mimi"][0])
    n_flow = int(o["n_flow"][0])
    # Queued decode latents = flow_latent[n_prefill : n_prefill+n_mimi].
    # n_prefill = n_flow - n_mimi - 1 (the final decode latent is computed but not queued: EOS break).
    n_prefill = n_flow - n_mimi - 1
    lats = [torch.tensor(o[f"flow_latent__{n_prefill + i}"]) for i in range(n_mimi)]
    latseq = torch.stack(lats, dim=1).reshape(1, n_mimi, c.ldim)  # [1,T,ldim]
    wav = c.mimi(latseq).reshape(-1)  # [1920*T]
    ref_audio = torch.tensor(o["audio"]).reshape(-1)
    n = min(wav.shape[0], ref_audio.shape[0])
    print(f"[mimi]     frames={n_mimi} wav={wav.shape[0]} ref={ref_audio.shape[0]} "
          f"cos={cos(wav[:n], ref_audio[:n]):.6f} maxdiff={maxdiff(wav[:n], ref_audio[:n]):.3e}")


if __name__ == "__main__":
    main()
