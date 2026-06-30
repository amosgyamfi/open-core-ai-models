"""Shared helpers for VibeVoice -> Core AI export.

Responsibilities:
* locate the downloaded HuggingFace checkpoint + the VibeVoice source tree
* load the composite ``VibeVoiceStreamingConfig``
* pull a subset of weights (by key prefix) out of ``model.safetensors`` so each
  component can be rebuilt and loaded in isolation (avoids instantiating the
  full inference model, which drags in the generate loop / scheduler / deps)
* PSNR helper for verifying Core AI outputs against the PyTorch reference
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

# <repo>/VibeVoiceCoreAI/export/vvexport/common.py  ->  <repo>
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]                 # .../VibeVoiceCoreAI
WORKSPACE_ROOT = PROJECT_ROOT.parent            # .../VibeVoice (workspace)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "VibeVoice-Realtime-0.5B"
DEFAULT_VIBEVOICE_SRC = WORKSPACE_ROOT / "VibeVoice"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "exports"

# Frame geometry (from preprocessor_config.json / acoustic tokenizer config).
SAMPLE_RATE = 24000
SPEECH_TOK_COMPRESS_RATIO = 3200   # audio samples per acoustic frame (7.5 Hz)
ACOUSTIC_VAE_DIM = 64


def ensure_vibevoice_importable(src: Path | None = None) -> Path:
    """Put the VibeVoice source tree on ``sys.path`` so ``import vibevoice`` works.

    We import only the leaf modules we need (diffusion head, acoustic tokenizer,
    configs) which depend solely on stable transformers APIs.
    """
    src = Path(src or DEFAULT_VIBEVOICE_SRC).resolve()
    if not (src / "vibevoice").is_dir():
        raise FileNotFoundError(
            f"VibeVoice source not found at {src}. Clone microsoft/VibeVoice there "
            "or pass --vibevoice-src."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


def load_config(model_dir: Path | None = None):
    """Load the composite VibeVoiceStreamingConfig from the checkpoint."""
    ensure_vibevoice_importable()
    from vibevoice.modular.configuration_vibevoice_streaming import (
        VibeVoiceStreamingConfig,
    )

    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    with open(model_dir / "config.json") as f:
        cfg_dict = json.load(f)
    return VibeVoiceStreamingConfig(**cfg_dict)


# ----------------------------------------------------------------------------
# Weight loading
# ----------------------------------------------------------------------------


def safetensors_path(model_dir: Path | None = None) -> Path:
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    p = model_dir / "model.safetensors"
    if not p.is_file():
        raise FileNotFoundError(f"Missing weights: {p}")
    return p


def load_subtree(
    prefix: str,
    *,
    model_dir: Path | None = None,
    dtype: torch.dtype = torch.float32,
    strip: bool = True,
) -> dict[str, torch.Tensor]:
    """Load all tensors whose key starts with ``prefix`` from the checkpoint.

    Args:
        prefix: e.g. ``"model.prediction_head."``
        strip: if True the prefix is removed from the returned keys (so they can
            be ``load_state_dict``-ed directly into the rebuilt submodule).
        dtype: cast floating tensors to this dtype (Core AI export wants fp32
            graphs; compute precision is chosen later by the optimizer).
    """
    out: dict[str, torch.Tensor] = {}
    with safe_open(safetensors_path(model_dir), framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            if not key.startswith(prefix):
                continue
            t = f.get_tensor(key)
            if t.is_floating_point():
                t = t.to(dtype)
            out[key[len(prefix):] if strip else key] = t
    if not out:
        raise KeyError(f"No tensors found with prefix {prefix!r}")
    return out


def load_scalar(name: str, *, model_dir: Path | None = None) -> torch.Tensor:
    with safe_open(safetensors_path(model_dir), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------


def psnr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB between two tensors (higher = closer).

    Core AI's working-with-coreai skill thresholds: fp32 end-to-end > 70 dB,
    fp16 on-device > 50 dB.
    """
    ref = reference.detach().float()
    act = actual.detach().float().reshape(ref.shape)
    mse = torch.mean((ref - act) ** 2).item()
    if mse == 0:
        return float("inf")
    peak = (ref.max() - ref.min()).item()
    if peak == 0:
        peak = ref.abs().max().item() or 1.0
    import math

    return 20 * math.log10(peak) - 10 * math.log10(mse)


@dataclass
class ExportPaths:
    out_dir: Path

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def aimodel(self, name: str) -> Path:
        return self.out_dir / f"{name}.aimodel"
