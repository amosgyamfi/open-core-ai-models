#!/usr/bin/env python
"""Export the VibeVoice diffusion head (prediction_head) to a Core AI .aimodel.

The diffusion head is the per-frame DPM denoiser:

    forward(noisy_latent[B,64], timestep[B], condition[B,896]) -> velocity[B,64]

At runtime the Swift pipeline runs this ~5 times per acoustic frame inside a
DPM-Solver loop, with B=2 (classifier-free guidance: positive + negative).

Run (from the coreai-models checkout so the coreai env is active):

    PYTHONPATH=$(pwd)/VibeVoiceCoreAI/export \
      uv run python VibeVoiceCoreAI/export/export_diffusion_head.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from vvexport import common
from vvexport.coreai_utils import export_stateless, run_program, save_program

logger = logging.getLogger("export.diffusion_head")

HF_ID = "microsoft/VibeVoice-Realtime-0.5B"
CFG_BATCH = 2  # positive + negative condition (classifier-free guidance)


class DiffusionHeadWrapper(torch.nn.Module):
    """Fixed-signature wrapper around VibeVoiceDiffusionHead for export."""

    def __init__(self, head: torch.nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(
        self,
        noisy_latent: torch.Tensor,  # [B, 64]
        timestep: torch.Tensor,      # [B]
        condition: torch.Tensor,     # [B, 896]
    ) -> torch.Tensor:
        return self.head(noisy_latent, timestep, condition)


def build_head(model_dir: Path) -> torch.nn.Module:
    common.ensure_vibevoice_importable()
    from vibevoice.modular.modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead

    cfg = common.load_config(model_dir)
    head = VibeVoiceDiffusionHead(cfg.diffusion_head_config)
    sd = common.load_subtree("model.prediction_head.", model_dir=model_dir, dtype=torch.float32)
    missing, unexpected = head.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected diffusion-head keys: {unexpected}")
    # t_embedder frequency buffers etc. are non-persistent; missing is acceptable
    head.eval().float()
    return head


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=str(common.DEFAULT_MODEL_DIR))
    ap.add_argument("--out-dir", default=str(common.DEFAULT_EXPORT_DIR))
    ap.add_argument("--verify", action="store_true", help="Run PSNR check vs PyTorch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_dir = Path(args.model_dir)
    paths = common.ExportPaths(Path(args.out_dir))

    head = build_head(model_dir)
    wrapper = DiffusionHeadWrapper(head).eval()

    dim = head.config.latent_size
    cond_dim = head.config.hidden_size
    dummy = (
        torch.randn(CFG_BATCH, dim, dtype=torch.float32),
        torch.full((CFG_BATCH,), 500.0, dtype=torch.float32),
        torch.randn(CFG_BATCH, cond_dim, dtype=torch.float32),
    )

    logger.info("Tracing + converting diffusion head (B=%d, latent=%d, cond=%d)...",
                CFG_BATCH, dim, cond_dim)
    program = export_stateless(
        wrapper,
        dummy,
        input_names=("noisy_latent", "timestep", "condition"),
        output_names=("velocity",),
    )

    asset = paths.aimodel("diffusion_head")
    save_program(program, asset, hf_model_id=HF_ID, component="diffusion_head")
    logger.info("Saved %s", asset)

    if args.verify:
        with torch.no_grad():
            ref = wrapper(*dummy)
        out = run_program(asset, {
            "noisy_latent": dummy[0],
            "timestep": dummy[1],
            "condition": dummy[2],
        })
        actual = next(iter(out.values()))
        db = common.psnr(ref, actual)
        logger.info("PSNR(diffusion_head) = %.2f dB (fp32 target > 70 dB)", db)
        if db < 60:
            raise SystemExit(f"PSNR too low: {db:.2f} dB")


if __name__ == "__main__":
    main()
