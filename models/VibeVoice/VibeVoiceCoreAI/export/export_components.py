#!/usr/bin/env python
"""Export the stateless VibeVoice components + dump small runtime resources.

Components exported to .aimodel:
  * acoustic_connector  : latent[B,1,64] -> lm_embed[B,1,896]
  * eos_classifier      : hidden[B,896]  -> eos_logit[B,1]
  * acoustic_decoder    : latents[1,64,T] -> audio[1, T*3200]   (non-streaming)

Resources written to <out>/resources/:
  * type_embedding.json     : tts_input_types weights (2 x 896)
  * scale_bias.json         : speech_scaling_factor / speech_bias_factor
  * audio.json              : sample rate, frame size

Run from the coreai-models checkout:
    PYTHONPATH=.../VibeVoiceCoreAI/export uv run python \
      VibeVoiceCoreAI/export/export_components.py --verify
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from vvexport import common
from vvexport.coreai_utils import export_stateless, run_program, save_program

logger = logging.getLogger("export.components")
HF_ID = "microsoft/VibeVoice-Realtime-0.5B"


# --------------------------------------------------------------------------- #
# acoustic_connector
# --------------------------------------------------------------------------- #
class ConnectorWrapper(torch.nn.Module):
    def __init__(self, connector: torch.nn.Module) -> None:
        super().__init__()
        self.connector = connector

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # [B,1,64] -> [B,1,896]
        return self.connector(latent)


def build_connector(model_dir: Path):
    common.ensure_vibevoice_importable()
    from vibevoice.modular.modeling_vibevoice_streaming import SpeechConnector

    cfg = common.load_config(model_dir)
    conn = SpeechConnector(cfg.acoustic_vae_dim, cfg.decoder_config.hidden_size)
    sd = common.load_subtree("model.acoustic_connector.", model_dir=model_dir, dtype=torch.float32)
    conn.load_state_dict(sd, strict=True)
    return conn.eval().float()


# --------------------------------------------------------------------------- #
# eos_classifier
# --------------------------------------------------------------------------- #
class EosWrapper(torch.nn.Module):
    def __init__(self, clf: torch.nn.Module) -> None:
        super().__init__()
        self.clf = clf

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # [B,896] -> [B,1]
        return self.clf(hidden)


def build_eos(model_dir: Path):
    common.ensure_vibevoice_importable()
    from vibevoice.modular.modeling_vibevoice_streaming import BinaryClassifier

    cfg = common.load_config(model_dir)
    clf = BinaryClassifier(cfg.decoder_config.hidden_size)
    sd = common.load_subtree("tts_eos_classifier.", model_dir=model_dir, dtype=torch.float32)
    clf.load_state_dict(sd, strict=True)
    return clf.eval().float()


# --------------------------------------------------------------------------- #
# acoustic_decoder (non-streaming whole-sequence decode)
# --------------------------------------------------------------------------- #
class DecoderWrapper(torch.nn.Module):
    """Wraps the acoustic tokenizer decoder for a fixed-length latent block.

    Input  latents : [1, 64, T]   (channels-first; already de-scaled)
    Output audio   : [1, T*3200]
    """

    def __init__(self, tokenizer: torch.nn.Module) -> None:
        super().__init__()
        self.tok = tokenizer

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        audio = self.tok.decode(latents, use_cache=False)  # [1,1,T*3200]
        return audio.squeeze(1)


def build_acoustic_tokenizer(model_dir: Path):
    common.ensure_vibevoice_importable()
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel

    cfg = common.load_config(model_dir)
    tok = VibeVoiceAcousticTokenizerModel(cfg.acoustic_tokenizer_config)
    # Only decoder weights are present in the streaming checkpoint.
    sd = common.load_subtree(
        "model.acoustic_tokenizer.", model_dir=model_dir, dtype=torch.float32
    )
    missing, unexpected = tok.load_state_dict(sd, strict=False)
    enc_missing = [m for m in missing if not m.startswith("encoder.")]
    if enc_missing:
        raise RuntimeError(f"Missing non-encoder tokenizer keys: {enc_missing[:8]} ...")
    if unexpected:
        raise RuntimeError(f"Unexpected tokenizer keys: {unexpected[:8]} ...")
    return tok.eval().float()


# --------------------------------------------------------------------------- #
# resources
# --------------------------------------------------------------------------- #
def dump_resources(model_dir: Path, res_dir: Path) -> None:
    res_dir.mkdir(parents=True, exist_ok=True)

    type_emb = common.load_subtree(
        "model.tts_input_types.", model_dir=model_dir, dtype=torch.float32
    )["weight"]  # [2, 896]
    (res_dir / "type_embedding.json").write_text(
        json.dumps({"shape": list(type_emb.shape), "data": type_emb.flatten().tolist()})
    )

    scale = common.load_scalar("model.speech_scaling_factor", model_dir=model_dir).float().item()
    bias = common.load_scalar("model.speech_bias_factor", model_dir=model_dir).float().item()
    (res_dir / "scale_bias.json").write_text(
        json.dumps({"speech_scaling_factor": scale, "speech_bias_factor": bias})
    )

    (res_dir / "audio.json").write_text(
        json.dumps(
            {
                "sample_rate": common.SAMPLE_RATE,
                "samples_per_frame": common.SPEECH_TOK_COMPRESS_RATIO,
                "acoustic_vae_dim": common.ACOUSTIC_VAE_DIM,
                "frame_rate_hz": common.SAMPLE_RATE / common.SPEECH_TOK_COMPRESS_RATIO,
            }
        )
    )
    logger.info("Wrote resources to %s", res_dir)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=str(common.DEFAULT_MODEL_DIR))
    ap.add_argument("--out-dir", default=str(common.DEFAULT_EXPORT_DIR))
    ap.add_argument("--decoder-frames", type=int, default=8,
                    help="Fixed latent block length T for the acoustic_decoder graph")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_dir = Path(args.model_dir)
    paths = common.ExportPaths(Path(args.out_dir))

    # ---- connector ---- (batch 1: one acoustic latent per step at runtime)
    conn = ConnectorWrapper(build_connector(model_dir)).eval()
    cd = torch.randn(1, 1, common.ACOUSTIC_VAE_DIM, dtype=torch.float32)
    prog = export_stateless(conn, (cd,), ("latent",), ("lm_embed",))
    a = save_program(prog, paths.aimodel("acoustic_connector"),
                     hf_model_id=HF_ID, component="acoustic_connector")
    logger.info("Saved %s", a)
    if args.verify:
        ref = conn(cd)
        out = next(iter(run_program(a, {"latent": cd}).values()))
        logger.info("PSNR(connector) = %.2f dB", common.psnr(ref, out))

    # ---- eos classifier ---- (batch 1: one hidden state per step at runtime)
    eos = EosWrapper(build_eos(model_dir)).eval()
    hd = torch.randn(1, 896, dtype=torch.float32)
    prog = export_stateless(eos, (hd,), ("hidden",), ("eos_logit",))
    a = save_program(prog, paths.aimodel("eos_classifier"),
                     hf_model_id=HF_ID, component="eos_classifier")
    logger.info("Saved %s", a)
    if args.verify:
        ref = eos(hd)
        out = next(iter(run_program(a, {"hidden": hd}).values()))
        logger.info("PSNR(eos) = %.2f dB", common.psnr(ref, out))

    # ---- acoustic decoder (non-streaming, fixed block of T frames) ----
    # The transposed-conv upsampling defeats torch.export's dynamic-shape solver,
    # so we export a fixed block size. The Swift runtime decodes a whole utterance
    # by zero-padding to T (single clean call for <= T frames) or by overlapped
    # blocks with a short crossfade for longer audio.
    if not args.skip_decoder:
        dec = DecoderWrapper(build_acoustic_tokenizer(model_dir)).eval()
        T = args.decoder_frames
        latents = torch.randn(1, common.ACOUSTIC_VAE_DIM, T, dtype=torch.float32) * 0.3
        logger.info("Exporting acoustic_decoder (T=%d frames -> %d samples)...",
                    T, T * common.SPEECH_TOK_COMPRESS_RATIO)
        prog = export_stateless(dec, (latents,), ("latents",), ("audio",))
        a = save_program(prog, paths.aimodel("acoustic_decoder"),
                         hf_model_id=HF_ID, component="acoustic_decoder")
        logger.info("Saved %s", a)
        if args.verify:
            with torch.no_grad():
                ref = dec(latents)
            out = next(iter(run_program(a, {"latents": latents}).values()))
            logger.info("PSNR(acoustic_decoder) = %.2f dB", common.psnr(ref, out))

    dump_resources(model_dir, paths.out_dir / "resources")
    logger.info("Done.")


if __name__ == "__main__":
    main()
