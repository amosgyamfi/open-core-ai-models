# Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
"""Assemble a self-contained on-device package the Swift host can ship:

    dist/PocketTTSCoreAI/
      bundles/{backbone,flow,mimi}.aimodel      # the 3 Core AI graphs
      resources/tokenizer.model                 # sentencepiece
      resources/glue/*.bin                       # host-side tensors (f32 LE)
      resources/voices/<name>/{key,value}.bin    # precomputed KV voice state
      manifest.json                              # shapes + scalars + voice offsets

Tensors are raw little-endian float32 so Swift can `Data` -> `[Float]` with no deps.

  python package.py [--dtype fp16] [--voices alba,marius,...]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from pockettts_coreai import load_components
from verify_eager import seed_caches_from_voice

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DIST = HERE / "dist" / "PocketTTSCoreAI"
CACHE_LEN = 2048
FRAME_SAMPLES = 1920  # 24 kHz / 12.5 Hz mimi frame rate
PREDEFINED = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]


def write_bin(path: Path, t: torch.Tensor) -> list[int]:
    arr = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(arr.tobytes())
    return list(arr.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--voices", default=",".join(PREDEFINED))
    args = ap.parse_args()
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]

    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "bundles").mkdir(parents=True)
    glue_dir = DIST / "resources" / "glue"
    glue_dir.mkdir(parents=True)

    c = load_components(cache_len=CACHE_LEN)
    manifest: dict = {
        "name": "PocketTTSCoreAI",
        "source": "kyutai/pocket-tts (CC-BY-4.0) — community Core AI port, NOT an Apple model",
        "dtype": args.dtype,
        "sample_rate": c.sample_rate,
        "frame_samples": FRAME_SAMPLES,
        "scalars": {
            "ldim": c.ldim,
            "num_layers": c.num_layers,
            "num_heads": c.num_heads,
            "dim_per_head": c.dim_per_head,
            "cache_len": CACHE_LEN,
            "temp": c.temp,
            "eos_threshold": c.eos_threshold,
            "n_bins": int(c.embed.shape[0] - 1),
        },
        "bundles": {}, "glue": {}, "voices": {},
    }

    # ---- bundles ----
    for short, full in (("backbone", f"pockettts_backbone_{args.dtype}"),
                        ("flow", f"pockettts_flow_{args.dtype}"),
                        ("mimi", f"pockettts_mimi_{args.dtype}")):
        src = ART / "bundles" / full / f"{full}.aimodel"
        dst = DIST / "bundles" / f"{short}.aimodel"
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy(src, dst)
        manifest["bundles"][short] = f"bundles/{short}.aimodel"

    # ---- host glue ----
    for key, t in {
        "embed": c.embed, "input_linear": c.input_linear, "bos_emb": c.bos_emb,
        "out_eos_w": c.out_eos_w, "out_eos_b": c.out_eos_b,
        "emb_mean": c.emb_mean, "emb_std": c.emb_std,
    }.items():
        shape = write_bin(glue_dir / f"{key}.bin", t)
        manifest["glue"][key] = {"file": f"resources/glue/{key}.bin", "shape": shape}

    # ---- tokenizer: ship the raw .model AND a Swift-friendly unigram JSON (+ self-test) ----
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as spb
    from pocket_tts.utils.utils import download_if_necessary
    from pocket_tts.conditioners.text import DEFAULT_TOKENIZER_PATH
    tok = Path(download_if_necessary(DEFAULT_TOKENIZER_PATH))
    shutil.copy(tok, DIST / "resources" / "tokenizer.model")
    sp = spm.SentencePieceProcessor(str(tok))
    proto = spb.ModelProto(); proto.ParseFromString(tok.read_bytes())
    pieces = [{"piece": p.piece, "score": p.score, "type": int(p.type)} for p in proto.pieces]
    samples = ["Hello world, this is a test.", "The quick brown fox jumps over the lazy dog.",
               "Pocket TTS runs on-device with Core AI."]
    tok_json = {
        "type": "unigram",
        "unk_id": sp.unk_id(),
        "byte_fallback": bool(proto.trainer_spec.byte_fallback),
        "add_dummy_prefix": bool(proto.normalizer_spec.add_dummy_prefix),
        "remove_extra_whitespaces": bool(proto.normalizer_spec.remove_extra_whitespaces),
        "space": "\u2581",
        "pieces": pieces,
        "selftest": [{"text": s, "ids": sp.encode(s, out_type=int)} for s in samples],
    }
    (DIST / "resources" / "tokenizer.json").write_text(json.dumps(tok_json, ensure_ascii=False))
    manifest["tokenizer"] = "resources/tokenizer.model"
    manifest["tokenizer_json"] = "resources/tokenizer.json"

    # ---- voices (precomputed KV state, cropped to written length) ----
    for v in voices:
        try:
            kc, vc, off = seed_caches_from_voice(c, v, CACHE_LEN)
        except Exception as e:  # noqa: BLE001 — voice download/gating may fail
            print(f"  [skip voice {v}] {type(e).__name__}: {e}")
            continue
        vk = kc[:, :, :, :off, :].contiguous()   # [L,1,H,off,Dh]
        vv = vc[:, :, :, :off, :].contiguous()
        ks = write_bin(DIST / "resources" / "voices" / v / "key.bin", vk)
        vs = write_bin(DIST / "resources" / "voices" / v / "value.bin", vv)
        manifest["voices"][v] = {
            "offset": off,
            "key": {"file": f"resources/voices/{v}/key.bin", "shape": ks},
            "value": {"file": f"resources/voices/{v}/value.bin", "shape": vs},
        }
        print(f"  [voice {v}] offset={off}")

    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"\npackaged -> {DIST}  ({total/1e6:.1f} MB, {len(manifest['voices'])} voices)")


if __name__ == "__main__":
    main()
