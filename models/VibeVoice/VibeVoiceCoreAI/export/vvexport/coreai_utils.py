"""Thin wrappers around the Core AI (coreai-torch) export API.

Mirrors the pattern Apple uses in coreai-models/diffusion/gpu.py:

    torch.export.export -> run_decompositions(get_decomp_table())
        -> TorchConverter().add_pytorch_module(...) -> to_coreai() -> optimize()

and adds a small ``save`` helper plus optional weight compression.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import coreai_torch
import torch

logger = logging.getLogger(__name__)


def export_stateless(
    wrapper: torch.nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    dynamic_shapes=None,
):
    """Export a stateless module to an optimized Core AI AIProgram.

    ``dynamic_shapes`` is forwarded to ``torch.export.export`` so callers can mark
    variable-length axes (e.g. the frame count of the acoustic decoder).
    """
    wrapper.eval()

    def export_fn(module: torch.nn.Module) -> torch.export.ExportedProgram:
        with torch.no_grad():
            exported = torch.export.export(
                module, args=dummy_inputs, dynamic_shapes=dynamic_shapes
            )
        return exported.run_decompositions(coreai_torch.get_decomp_table())

    converter = coreai_torch.TorchConverter()
    converter.add_pytorch_module(
        wrapper,
        export_fn=export_fn,
        input_names=list(input_names),
        output_names=list(output_names),
    )
    program = converter.to_coreai()
    program.optimize()
    return program


def save_program(program, asset_path: Path, *, hf_model_id: str, component: str) -> Path:
    """Persist an AIProgram to ``asset_path`` (.aimodel), attaching metadata."""
    asset_path = Path(asset_path)
    if asset_path.exists():
        shutil.rmtree(asset_path) if asset_path.is_dir() else asset_path.unlink()
    metadata = None
    try:
        from coreai_models.export.metadata import build_aimodel_metadata

        metadata = build_aimodel_metadata(hf_model_id, component=component)
    except Exception as exc:  # pragma: no cover - metadata is best-effort
        logger.warning("metadata unavailable (%s); saving without it", exc)
    try:
        program.save_asset(asset_path, metadata) if metadata is not None else program.save_asset(
            asset_path
        )
    except TypeError:
        program.save_asset(asset_path)
    return asset_path


def run_program(asset_path: Path, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Load an exported .aimodel via the Core AI Python runtime and run it once.

    Used for numerical verification against the PyTorch reference.
    """
    import asyncio

    import numpy as np
    from coreai.runtime import AIModel, NDArray

    async def _run() -> dict[str, torch.Tensor]:
        model = await AIModel.load(str(asset_path))
        fn = model.load_function("main")
        nd_inputs = {
            k: NDArray(v.detach().cpu().numpy().astype(np.float32)) for k, v in inputs.items()
        }
        outputs = await fn(nd_inputs)
        return {k: torch.from_numpy(np.array(v.numpy())) for k, v in outputs.items()}

    return asyncio.run(_run())
