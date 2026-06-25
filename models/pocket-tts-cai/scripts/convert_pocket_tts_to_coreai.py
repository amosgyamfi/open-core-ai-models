#!/usr/bin/env python
"""Convert Pocket TTS export-friendly submodules to Core AI `.aimodel` assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from pocket_tts import TTSModel
from pocket_tts_coreai import (
    DeterministicFlowLMContextStep,
    MimiDecodeChunk,
    TextConditionerExport,
    patch_export_activations,
    patch_export_normalizations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="english", help="Pocket TTS language config name.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/coreai/pocket-tts-english"),
        help="Directory where .aimodel assets and metadata are written.",
    )
    parser.add_argument("--max-text-tokens", type=int, default=96)
    parser.add_argument("--max-latent-context", type=int, default=64)
    parser.add_argument("--decode-latent-frames", type=int, default=1)
    parser.add_argument("--lsd-decode-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Save unoptimized AIProgram assets for debugging converter issues.",
    )
    return parser.parse_args()


def require_coreai_torch():
    try:
        from coreai_torch import TorchConverter, get_decomp_table
    except Exception as exc:
        raise RuntimeError(
            "coreai-torch is not importable in this Python environment. Install Apple's "
            "Core AI PyTorch Extensions on a supported platform, then rerun this script."
        ) from exc
    return TorchConverter, get_decomp_table


def export_program(module: torch.nn.Module, args: tuple[torch.Tensor, ...]):
    try:
        return torch.export.export(module, args=args)
    except Exception:
        return torch.export.export(module, args=args, strict=False)


def convert_module(
    *,
    name: str,
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    output_dir: Path,
    skip_optimize: bool,
) -> Path:
    TorchConverter, get_decomp_table = require_coreai_torch()
    module.eval()

    exported = export_program(module, args)
    exported = exported.run_decompositions(get_decomp_table())
    program = (
        TorchConverter()
        .add_exported_program(
            exported,
            input_names=input_names,
            output_names=output_names,
        )
        .to_coreai()
    )
    if not skip_optimize:
        program.optimize()

    asset_path = output_dir / f"{name}.aimodel"
    if asset_path.exists():
        shutil.rmtree(asset_path)
    program.save_asset(asset_path)
    return asset_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    tts_model = TTSModel.load_model(
        language=args.language,
        temp=args.temperature,
        lsd_decode_steps=args.lsd_decode_steps,
    ).eval()
    patch_export_activations(tts_model)
    patch_export_normalizations(tts_model)

    flow_lm = tts_model.flow_lm.eval()
    mimi = tts_model.mimi.eval()
    dtype = next(flow_lm.parameters()).dtype

    text_tokens = torch.zeros((1, args.max_text_tokens), dtype=torch.long)
    text_embeddings = torch.zeros((1, args.max_text_tokens, flow_lm.dim), dtype=dtype)
    latent_context = torch.full((1, args.max_latent_context, flow_lm.ldim), float("nan"), dtype=dtype)
    noise = torch.zeros((1, flow_lm.ldim), dtype=dtype)
    decode_latents = torch.zeros((1, args.decode_latent_frames, flow_lm.ldim), dtype=dtype)

    assets = {}
    assets["text_conditioner"] = convert_module(
        name="text_conditioner",
        module=TextConditionerExport(flow_lm),
        args=(text_tokens,),
        input_names=["text_tokens"],
        output_names=["text_embeddings"],
        output_dir=args.output_dir,
        skip_optimize=args.skip_optimize,
    )
    assets["flow_lm_context_step"] = convert_module(
        name="flow_lm_context_step",
        module=DeterministicFlowLMContextStep(
            flow_lm,
            lsd_decode_steps=args.lsd_decode_steps,
            temperature=args.temperature,
        ),
        args=(latent_context, text_embeddings, noise),
        input_names=["latent_context", "text_embeddings", "noise"],
        output_names=["next_latent", "eos_logit"],
        output_dir=args.output_dir,
        skip_optimize=args.skip_optimize,
    )
    assets["mimi_decode_chunk"] = convert_module(
        name="mimi_decode_chunk",
        module=MimiDecodeChunk(flow_lm, mimi),
        args=(decode_latents,),
        input_names=["normalized_latents"],
        output_names=["audio"],
        output_dir=args.output_dir,
        skip_optimize=args.skip_optimize,
    )

    metadata = {
        "source_model": "kyutai/pocket-tts",
        "language": args.language,
        "sample_rate": tts_model.sample_rate,
        "max_text_tokens": args.max_text_tokens,
        "max_latent_context": args.max_latent_context,
        "decode_lsd_steps": args.lsd_decode_steps,
        "temperature": args.temperature,
        "latent_dim": flow_lm.ldim,
        "flow_dim": flow_lm.dim,
        "mimi_frame_rate": mimi.frame_rate,
        "assets": {key: str(path) for key, path in assets.items()},
        "notes": [
            "Noise is an explicit model input for deterministic Core AI execution.",
            "Swift must perform SentencePiece tokenization and autoregressive orchestration.",
            "EOS should be computed from eos_logit using the app's chosen threshold.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote Core AI assets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
