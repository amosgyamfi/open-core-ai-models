"""Manual scaled-dot-product attention for the VibeVoice Qwen2 export.

The Core AI *runtime* in the macOS 27 beta miscompiles the externalized
``coreai_torch.composite_ops.scaled_dot_product_attention`` op in f32 (Apple's
own ``test_qwen2.py`` attention parity test fails with ~10.0 abs error, and the
ios SDPA primitive test fails too). RoPE, RMSNorm, matmul and softmax all run
correctly through the runtime, so we re-express attention with those basic ops.

The math matches an incremental causal decode: ``query`` holds ``q_len`` tokens
whose absolute positions are the last ``q_len`` of the ``k_len`` cached keys, so
query ``i`` may attend to keys ``j <= (k_len - q_len) + i``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ManualSDPA(nn.Module):
    """Drop-in replacement for ``primitives.macos.sdpa.SDPA`` (causal, GQA)."""

    def __init__(self, scale: float | None = None, is_causal: bool = True) -> None:
        super().__init__()
        self.scale = scale
        self.is_causal = is_causal

    def forward(
        self,
        query: torch.Tensor,  # (B, n_heads, q_len, hd)
        key: torch.Tensor,  # (B, n_kv_heads, k_len, hd)
        value: torch.Tensor,  # (B, n_kv_heads, k_len, hd)
        attn_mask: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n_heads, q_len, hd = query.shape
        n_kv_heads = key.shape[1]
        k_len = key.shape[2]
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(hd)

        # GQA: expand kv heads to match query heads.
        if n_kv_heads != n_heads:
            rep = n_heads // n_kv_heads
            key = key.unsqueeze(2).expand(b, n_kv_heads, rep, k_len, hd).reshape(
                b, n_heads, k_len, hd
            )
            value = value.unsqueeze(2).expand(b, n_kv_heads, rep, k_len, hd).reshape(
                b, n_heads, k_len, hd
            )

        scores = torch.matmul(query, key.transpose(-1, -2)) * scale  # (B, n_heads, q_len, k_len)

        if attn_mask is not None:
            scores = scores + attn_mask
        elif self.is_causal:
            q_idx = torch.arange(q_len, device=scores.device).unsqueeze(-1)  # (q_len, 1)
            k_idx = torch.arange(k_len, device=scores.device).unsqueeze(0)  # (1, k_len)
            allowed = k_idx <= (k_len - q_len) + q_idx  # (q_len, k_len)
            neg = torch.finfo(scores.dtype).min
            bias = torch.where(allowed, torch.zeros((), dtype=scores.dtype),
                               torch.full((), neg, dtype=scores.dtype))
            scores = scores + bias

        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)  # (B, n_heads, q_len, hd)
