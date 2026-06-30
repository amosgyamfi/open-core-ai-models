#!/usr/bin/env python
"""Convert VibeVoice voice-prefill ``.pt`` files into Core AI voice bundles.

A VibeVoice "voice" is a *cached prompt*: the prefilled KV caches and final
hidden states of the four Qwen2 stacks (``lm``, ``tts_lm``, ``neg_lm``,
``neg_tts_lm``) for that speaker's reference prompt. The same voice bundle is
reused for any text -- the text is streamed in afterwards.

Each source ``.pt`` is a dict ``{stack: BaseModelOutputWithPast}`` where every
stack carries:

  * ``last_hidden_state`` : (1, L, 896)
  * ``past_key_values``   : DynamicCache, per-layer k/v of (1, 2, L, 64)

We re-pack this into the layout expected by the exported Core AI LLM state
(:class:`coreai_models.primitives.macos.cache.KVCache`):

  * ``<stack>.k`` / ``<stack>.v`` : (n_layers, 1, n_kv_heads, L, 64)
  * ``<stack>.h``                 : (1, L, 896)  (only the last row is used as a
                                     diffusion condition, but we keep all rows)

and write one ``<name>.safetensors`` per voice (fp16) with a JSON metadata
blob recording the per-stack prefill lengths, language and gender. The Swift
runtime memory-maps these directly into the model's KV-cache state.

Run from the coreai-models checkout::

    uv run python VibeVoiceCoreAI/export/convert_voices.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

logger = logging.getLogger("convert_voices")

import os
FP16 = os.environ.get("VV_VOICE_FP16", "0") == "1"

STACKS = ("lm", "tts_lm", "neg_lm", "neg_tts_lm")

# Human-readable language names keyed by the filename prefix used in the packs.
LANGS = {
    "en": "English",
    "in": "English (Indian)",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "jp": "Japanese",
    "kr": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "sp": "Spanish",
}


def _stack_kv(pkv: DynamicCache) -> tuple[torch.Tensor, torch.Tensor]:
    """DynamicCache -> (k, v) of shape (n_layers, 1, n_kv_heads, L, 64)."""
    k = torch.stack([t.contiguous() for t in pkv.key_cache], dim=0)
    v = torch.stack([t.contiguous() for t in pkv.value_cache], dim=0)
    return k, v


def _parse_name(stem: str) -> tuple[str, str, str]:
    """``en-Carter_man`` -> (name, language, gender)."""
    lang_code = stem.split("-", 1)[0].lower() if "-" in stem else "en"
    language = LANGS.get(lang_code, lang_code)
    gender = "unknown"
    low = stem.lower()
    if low.endswith("_man") or low.endswith("-man"):
        gender = "male"
    elif low.endswith("_woman") or low.endswith("-woman"):
        gender = "female"
    return stem, language, gender


def convert_one(pt_path: Path, out_dir: Path) -> dict:
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)

    # fp32 keeps the prefill KV cache faithful to the source model; the EOS
    # classifier is a single-logit threshold and is sensitive to drift.
    dtype = torch.float16 if FP16 else torch.float32
    tensors: dict[str, torch.Tensor] = {}
    lengths: dict[str, int] = {}
    for stack in STACKS:
        out = data[stack]
        h = out["last_hidden_state"].to(dtype).contiguous()
        k, v = _stack_kv(out["past_key_values"])
        tensors[f"{stack}.h"] = h
        tensors[f"{stack}.k"] = k.to(dtype).contiguous()
        tensors[f"{stack}.v"] = v.to(dtype).contiguous()
        lengths[stack] = int(h.shape[1])

    name, language, gender = _parse_name(pt_path.stem)
    meta = {
        "name": name,
        "language": language,
        "gender": gender,
        "lengths": json.dumps(lengths),
        "hidden_size": str(tensors["lm.h"].shape[-1]),
        "base_layers": str(tensors["lm.k"].shape[0]),
        "tts_layers": str(tensors["tts_lm.k"].shape[0]),
        "kv_heads": str(tensors["lm.k"].shape[2]),
        "head_dim": str(tensors["lm.k"].shape[-1]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.safetensors"
    save_file(tensors, str(out_path), metadata=meta)
    return {"name": name, "language": language, "gender": gender,
            "lengths": lengths, "file": out_path.name}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--voices-dir",
        default="/Users/amosgyamfi/Desktop/VibeVoice/VibeVoice/demo/voices/streaming_model",
        help="Directory holding the source *.pt voice prefills (scanned recursively).",
    )
    ap.add_argument(
        "--out-dir",
        default="/Users/amosgyamfi/Desktop/VibeVoice/VibeVoiceCoreAI/exports/voices",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    voices_dir = Path(args.voices_dir)
    out_dir = Path(args.out_dir)
    pt_files = sorted(voices_dir.rglob("*.pt"))
    if not pt_files:
        raise SystemExit(f"No .pt voices found under {voices_dir}")

    catalog = []
    for pt in pt_files:
        try:
            info = convert_one(pt, out_dir)
            catalog.append(info)
            logger.info("converted %-22s [%s, %s] lengths=%s",
                        info["name"], info["language"], info["gender"], info["lengths"])
        except Exception as exc:  # noqa: BLE001
            logger.error("FAILED %s: %s", pt.name, exc)

    catalog.sort(key=lambda c: (c["language"], c["name"]))
    (out_dir / "voices.json").write_text(json.dumps(catalog, indent=2))
    langs = sorted({c["language"] for c in catalog})
    logger.info("Wrote %d voice bundles across %d languages -> %s",
                len(catalog), len(langs), out_dir)
    logger.info("Languages: %s", ", ".join(langs))


if __name__ == "__main__":
    main()
