"""Export-friendly Pocket TTS module boundaries.

Pocket TTS is a streaming TTS system with tokenizer, autoregressive control flow,
explicit model state, queueing, and audio I/O around its PyTorch modules. Core AI
conversion is more reliable when the Swift app owns that orchestration and each
`.aimodel` contains a deterministic tensor subgraph.
"""

from __future__ import annotations

import math
import types

import torch
from torch import nn

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.modules.stateful_module import init_states


def _variance_via_mean(x: torch.Tensor, unbiased: bool) -> torch.Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    squared = centered * centered
    if unbiased and x.shape[-1] > 1:
        return squared.sum(dim=-1, keepdim=True) / (x.shape[-1] - 1)
    return squared.mean(dim=-1, keepdim=True)


def _export_rms_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    x_dtype = x.dtype
    var = self.eps + _variance_via_mean(x, unbiased=True)
    return (x * (self.alpha.to(var) * torch.rsqrt(var))).to(x_dtype)


def _export_layer_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = _variance_via_mean(x, unbiased=False)
    x = (x - mean) * torch.rsqrt(var + self.eps)
    if hasattr(self, "weight"):
        x = x * self.weight + self.bias
    return x


def patch_export_normalizations(module: nn.Module) -> None:
    """Avoid ATen var ops that coreai-torch 0.4.0 cannot lower."""

    from pocket_tts.modules.mlp import LayerNorm, RMSNorm

    for child in module.modules():
        if isinstance(child, RMSNorm):
            child.forward = types.MethodType(_export_rms_norm_forward, child)
        elif isinstance(child, LayerNorm):
            child.forward = types.MethodType(_export_layer_norm_forward, child)


class ExportELU(nn.Module):
    """ELU(alpha=1) expressed with lower-level ops for coreai-torch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, torch.exp(x) - 1)


def patch_export_activations(module: nn.Module) -> None:
    """Replace activation modules that coreai-torch 0.4.0 cannot lower."""

    for name, child in list(module.named_children()):
        if isinstance(child, nn.ELU):
            if child.alpha != 1.0:
                raise ValueError(f"Only ELU(alpha=1) is supported, got alpha={child.alpha}")
            setattr(module, name, ExportELU())
        else:
            patch_export_activations(child)


class TextConditionerExport(nn.Module):
    """Token ids to FlowLM text-conditioning embeddings."""

    def __init__(self, flow_lm: nn.Module):
        super().__init__()
        self.conditioner = flow_lm.conditioner

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.conditioner(TokenizedText(text_tokens))


class DeterministicFlowLMContextStep(nn.Module):
    """Generate one normalized latent from a full latent context.

    The original Pocket TTS generation path samples noise inside PyTorch and uses
    a mutable KV cache. This wrapper removes both from the Core AI boundary:
    Swift supplies the latent context and a standard-normal noise tensor, and the
    model returns the next latent plus an EOS logit.
    """

    def __init__(
        self,
        flow_lm: nn.Module,
        lsd_decode_steps: int,
        temperature: float,
    ):
        super().__init__()
        if lsd_decode_steps < 1:
            raise ValueError("lsd_decode_steps must be >= 1 for inference export")
        self.flow_lm = flow_lm
        self.lsd_decode_steps = int(lsd_decode_steps)
        self.temperature = float(temperature)

    def forward(
        self,
        latent_context: torch.Tensor,
        text_embeddings: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent_context.shape[0]
        bos = self.flow_lm.bos_emb.view(1, 1, -1).expand(batch_size, 1, -1)
        sequence = torch.cat([bos, latent_context], dim=1)
        transformer_in = self.flow_lm.input_linear(sequence)
        transformer_in = torch.cat([text_embeddings, transformer_in], dim=1)
        transformer_out = self.flow_lm.transformer(transformer_in, model_state=None)
        if self.flow_lm.out_norm:
            transformer_out = self.flow_lm.out_norm(transformer_out)
        transformer_out = transformer_out[:, -sequence.shape[1] :]
        transformer_out = transformer_out.to(torch.float32)[:, -1]
        eos_logit = self.flow_lm.out_eos(transformer_out)

        current = noise.to(transformer_out.dtype) * math.sqrt(self.temperature)
        for i in range(self.lsd_decode_steps):
            s_value = i / self.lsd_decode_steps
            t_value = (i + 1) / self.lsd_decode_steps
            s = torch.full_like(current[..., :1], s_value)
            t = torch.full_like(current[..., :1], t_value)
            flow_dir = self.flow_lm.flow_net(transformer_out, s, t, current)
            current = current + flow_dir / self.lsd_decode_steps
        return current, eos_logit


class MimiDecodeChunk(nn.Module):
    """Decode normalized FlowLM latents to PCM audio samples."""

    def __init__(self, flow_lm: nn.Module, mimi: nn.Module):
        super().__init__()
        self.flow_lm = flow_lm
        self.mimi = mimi

    def forward(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        latents = normalized_latents * self.flow_lm.emb_std + self.flow_lm.emb_mean
        quantized = self.mimi.quantizer(latents.transpose(1, 2))

        batch_size = normalized_latents.shape[0]
        frame_count = normalized_latents.shape[1]
        mimi_state = init_states(self.mimi, batch_size=batch_size, sequence_length=frame_count)

        emb = self.mimi._to_encoder_framerate(quantized, mimi_state)
        (emb,) = self.mimi.decoder_transformer(emb, model_state=None)
        audio = self.mimi.decoder(emb, mimi_state)
        return audio
