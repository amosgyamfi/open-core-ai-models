# Community port of kyutai/pocket-tts to Apple Core AI — NOT an Apple model.
"""Export-friendly re-authoring of the Pocket-TTS sub-networks.

All modules reuse the *original* pocket-tts weights (loaded via ``pocket_tts.TTSModel``); only the
non-lowerable streaming machinery is replaced:

  - data-dependent KV cache (``.item()`` offset, NaN-init) -> fixed cache + explicit ``pos`` input,
    explicit ``arange<=pos`` causal mask, baked RoPE cos/sin (no float ``arange``).
  - streaming convs / conv-transposes -> stateless causal ops (zero left-pad / drop partial tail),
    so the whole codec decodes a full latent sequence in one pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pocket_tts import TTSModel
from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.mlp import LayerNorm as FlowLayerNorm
from pocket_tts.modules.mlp import RMSNorm as FlowRMSNorm
from pocket_tts.modules.seanet import SEANetResnetBlock


def _patch_export_norms(module: nn.Module) -> None:
    """Replace the flow net's custom norms (which emit unsupported ``aten.var.correction``) with
    numerically-identical var-free forwards, in place."""

    def ln_forward(self, x):  # matches mlp.LayerNorm (unbiased=False variance)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if hasattr(self, "weight"):
            x = x * self.weight + self.bias
        return x

    def rms_forward(self, x):  # matches mlp._rms_norm (unbiased=True variance, no mean-subtract on x)
        n = x.shape[-1]
        mean = x.mean(dim=-1, keepdim=True)
        var = self.eps + ((x - mean) ** 2).sum(dim=-1, keepdim=True) / (n - 1)
        return x * (self.alpha.to(var) * torch.rsqrt(var))

    import types

    for m in module.modules():
        if isinstance(m, FlowLayerNorm):
            m.forward = types.MethodType(ln_forward, m)
        elif isinstance(m, FlowRMSNorm):
            m.forward = types.MethodType(rms_forward, m)


# --------------------------------------------------------------------------------------------
# Backbone — q=1 stateful KV decode (also used for prefill by looping, bit-identical)
# --------------------------------------------------------------------------------------------
class Backbone(nn.Module):
    """One decode step of the FlowLM transformer backbone.

    forward(inputs_embeds[1,1,D], pos[int32 scalar], k_cache[L,1,H,CL,Dh], v_cache[L,1,H,CL,Dh])
      -> hidden[1,1,D]   (== out_norm(transformer_out))

    ``k_cache`` / ``v_cache`` are mutated in place at time index ``pos`` and become Core AI states.
    """

    def __init__(self, src: TTSModel, cache_len: int):
        super().__init__()
        fl = src.flow_lm
        self.layers = fl.transformer.layers
        self.out_norm = fl.out_norm
        self.num_heads = self.layers[0].self_attn.num_heads
        self.dim_per_head = self.layers[0].self_attn.dim_per_head
        self.scale = 1.0 / math.sqrt(self.dim_per_head)
        self.cache_len = cache_len
        self.max_period = float(fl.transformer.rope.max_period)
        d = self.dim_per_head
        # RoPE inverse frequencies as a plain fp32 attribute (NOT a buffer) so .half() can't
        # underflow the small frequencies to zero (conversion-guide fp16 rule).
        self.inv_freq = torch.exp(
            torch.arange(d // 2, dtype=torch.float32) * (-math.log(self.max_period) * 2 / d)
        )

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [1, T, H, D] -> rotate adjacent pairs (matches pocket_tts.apply_rope layout)
        b, t, h, d = x.shape
        x = x.view(b, t, h, d // 2, 2)
        xr, xi = x[..., 0], x[..., 1]
        out_r = xr * cos - xi * sin
        out_i = xr * sin + xi * cos
        return torch.stack([out_r, out_i], dim=-1).view(b, t, h, d)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        H, Dh, CL = self.num_heads, self.dim_per_head, self.cache_len
        pos_i = pos.to(torch.int64)                     # [1]  (avoid 0-d reshape)
        pos_f = pos.to(inputs_embeds.dtype)             # [1]
        ang = self.inv_freq.to(inputs_embeds.dtype) * pos_f  # [Dh/2]
        cos = torch.cos(ang).view(1, 1, 1, -1)
        sin = torch.sin(ang).view(1, 1, 1, -1)
        # float causal mask over the static cache: keep keys at positions <= pos.
        idx = torch.arange(CL, device=inputs_embeds.device)
        keep = (idx <= pos_i).view(1, 1, 1, CL)
        neg = torch.full((1, 1, 1, CL), float("-inf"), dtype=inputs_embeds.dtype)
        mask = torch.where(keep, torch.zeros_like(neg), neg)
        # one-hot KV write selector at `pos` (avoids data-dependent dynamic indexing, which
        # breaks torch.export / the in-graph KV-write path — see zoo "input-mask escape").
        onehot = (idx == pos_i).to(inputs_embeds.dtype).view(1, 1, CL, 1)  # [1,1,CL,1]

        x = inputs_embeds
        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = layer.norm1(x)
            proj = attn.in_proj(h)
            b, t, _ = proj.shape
            packed = proj.view(b, t, 3, H, Dh)
            q, k, v = torch.unbind(packed, dim=2)  # each [1,1,H,Dh]
            q = self._rope(q, cos, sin)
            k = self._rope(k, cos, sin)
            kp = k.permute(0, 2, 1, 3)  # [1,H,1,Dh] (broadcasts over CL)
            vp = v.permute(0, 2, 1, 3)
            # masked in-place KV write at time index `pos` (becomes a Core AI state mutation)
            k_all = k_cache[li] * (1.0 - onehot) + kp * onehot  # [1,H,CL,Dh]
            v_all = v_cache[li] * (1.0 - onehot) + vp * onehot
            k_cache[li] = k_all
            v_cache[li] = v_all
            qp = q.permute(0, 2, 1, 3)  # [1,H,1,Dh]
            scores = torch.matmul(qp, k_all.transpose(-1, -2)) * self.scale  # [1,H,1,CL]
            scores = scores + mask
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v_all)  # [1,H,1,Dh]
            ctx = ctx.permute(0, 2, 1, 3).reshape(b, t, H * Dh)
            x = x + attn.out_proj(ctx)
            # feed-forward (layer_scale is Identity for the flow-lm backbone)
            hf = layer.norm2(x)
            x = x + layer.linear2(F.gelu(layer.linear1(hf)))
        return self.out_norm(x)


# --------------------------------------------------------------------------------------------
# Flow decoder — single-step LSD (lsd_decode_steps == 1): latent = z + flow_net(cond, 0, 1, z)
# --------------------------------------------------------------------------------------------
class FlowDecoder(nn.Module):
    def __init__(self, src: TTSModel):
        super().__init__()
        self.flow_net = src.flow_lm.flow_net
        self.steps = int(src.lsd_decode_steps)
        _patch_export_norms(self.flow_net)

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # cond: [1, Dmodel]  z: [1, ldim]
        current = z
        n = self.steps
        for i in range(n):
            s = torch.full_like(z[..., :1], i / n)
            t = torch.full_like(z[..., :1], (i + 1) / n)
            flow_dir = self.flow_net(cond, s, t, current)
            current = current + flow_dir / n
        return current


# --------------------------------------------------------------------------------------------
# Mimi decoder — stateless, full-sequence: latents[1,T,ldim] -> waveform[1,1,1920*T]
# --------------------------------------------------------------------------------------------
def _conv1d_stateless(conv: StreamingConv1d, x: torch.Tensor) -> torch.Tensor:
    # Causal left-pad with zeros by (effective_kernel - stride); pad_mode is "constant" here.
    eff_k = conv._effective_kernel_size
    pad = eff_k - conv._stride
    if pad > 0:
        x = F.pad(x, (pad, 0))
    return conv.conv(x)


def _convtr1d_stateless(convtr: StreamingConvTranspose1d, x: torch.Tensor) -> torch.Tensor:
    # Fresh streaming state => front partial is zero and the trailing (K-S) partial is dropped.
    y = convtr.convtr(x)
    pt = convtr._kernel_size - convtr._stride
    if pt > 0:
        y = y[..., :-pt]
    return y


def _resnet_stateless(block: SEANetResnetBlock, x: torch.Tensor) -> torch.Tensor:
    v = x
    for layer in block.block:
        if isinstance(layer, StreamingConv1d):
            v = _conv1d_stateless(layer, v)
        else:
            v = layer(v)
    return x + v


def _patch_elu(module: nn.Module) -> None:
    """Replace nn.ELU (unsupported ``aten.elu``) with its explicit formula (alpha == 1)."""
    import types

    def elu_forward(self, x):
        a = float(self.alpha)
        return torch.where(x > 0, x, a * (torch.exp(x) - 1.0))

    for m in module.modules():
        if isinstance(m, nn.ELU):
            m.forward = types.MethodType(elu_forward, m)


class MimiDecoder(nn.Module):
    def __init__(self, src: TTSModel):
        super().__init__()
        self.mimi = src.mimi
        self.decoder = src.mimi.decoder
        self.decoder_transformer = src.mimi.decoder_transformer
        self.upsample = src.mimi.upsample
        self.quantizer = src.mimi.quantizer
        self.register_buffer("emb_std", src.flow_lm.emb_std.detach().clone())
        self.register_buffer("emb_mean", src.flow_lm.emb_mean.detach().clone())
        _patch_elu(self.decoder)

    def _decode_seanet(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder.model:
            if isinstance(layer, StreamingConvTranspose1d):
                z = _convtr1d_stateless(layer, z)
            elif isinstance(layer, SEANetResnetBlock):
                z = _resnet_stateless(layer, z)
            elif isinstance(layer, StreamingConv1d):
                z = _conv1d_stateless(layer, z)
            else:
                z = layer(z)
        return z

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [1, T, ldim] (raw flow latents, pre de-normalization)
        x = latents * self.emb_std + self.emb_mean
        x = x.transpose(1, 2)  # [1, ldim, T]
        q = self.quantizer(x)  # [1, 512, T]
        emb = _convtr1d_stateless(self.upsample.convtr, q)  # [1, 512, 16T]
        (emb,) = self.decoder_transformer(emb, None)  # stateless attention (clean branch)
        wav = self._decode_seanet(emb)  # [1, 1, 1920T]
        return wav


@dataclass
class Components:
    backbone: Backbone
    flow: FlowDecoder
    mimi: MimiDecoder
    # host-side glue tensors (fp32)
    embed: torch.Tensor          # [n_bins+1, Dmodel] text token embedding table
    input_linear: torch.Tensor   # [Dmodel, ldim] latent -> model dim
    bos_emb: torch.Tensor        # [ldim] substituted for NaN backbone input
    bos_before_voice: torch.Tensor  # [1,1,Dmodel]
    out_eos_w: torch.Tensor      # [1, Dmodel]
    out_eos_b: torch.Tensor      # [1]
    emb_std: torch.Tensor        # [ldim]
    emb_mean: torch.Tensor       # [ldim]
    num_layers: int
    num_heads: int
    dim_per_head: int
    d_model: int
    ldim: int
    temp: float
    eos_threshold: float
    sample_rate: int
    src: TTSModel


def load_components(cache_len: int = 2048, language: str | None = None) -> Components:
    src = TTSModel.load_model(language=language) if language else TTSModel.load_model()
    src.eval()
    fl = src.flow_lm
    bb = Backbone(src, cache_len=cache_len).eval()
    return Components(
        backbone=bb,
        flow=FlowDecoder(src).eval(),
        mimi=MimiDecoder(src).eval(),
        embed=fl.conditioner.embed.weight.detach().clone(),
        input_linear=fl.input_linear.weight.detach().clone(),
        bos_emb=fl.bos_emb.detach().clone(),
        bos_before_voice=fl.bos_before_voice.detach().clone(),
        out_eos_w=fl.out_eos.weight.detach().clone(),
        out_eos_b=fl.out_eos.bias.detach().clone(),
        emb_std=fl.emb_std.detach().clone(),
        emb_mean=fl.emb_mean.detach().clone(),
        num_layers=len(fl.transformer.layers),
        num_heads=bb.num_heads,
        dim_per_head=bb.dim_per_head,
        d_model=fl.dim,
        ldim=fl.ldim,
        temp=float(src.temp),
        eos_threshold=float(src.eos_threshold),
        sample_rate=src.sample_rate,
        src=src,
    )
