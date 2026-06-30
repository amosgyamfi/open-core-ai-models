"""Stateless Qwen2 stacks for VibeVoice Core AI export.

The macOS 27 beta Core AI runtime miscompiles the stateful attention path
(externalized SDPA composite + KV-cache *state* via mutable_slice_update): Apple's
own f32 attention parity test fails by ~12.0 abs, and our exported base_lm drifts
~4-6% at the incremental step which trips the EOS classifier early.

Every op we need individually runs correctly through the runtime (matmul, softmax,
cat, RoPE, RMSNorm — verified by the primitive tests and isolation probes). So we
re-express the stacks WITHOUT state and WITHOUT the SDPA composite:

  * the KV cache is a plain input (past_k/past_v, length L) and the freshly
    computed keys/values are returned as outputs for the caller to append;
  * attention is manual matmul + softmax with an explicit causal bias.

forward(hidden_or_ids, position_ids, past_k, past_v) ->
    (hidden, new_k, new_v)

Shapes (per stack):
  past_k/past_v : (n_layers, 1, n_kv_heads, L, head_dim)   # L >= 1
  new_k/new_v   : (n_layers, 1, n_kv_heads, q, head_dim)
  position_ids  : (1, q)  the absolute positions of the q query tokens ([L, L+q))

`position_ids` holds only the query positions (length q) and the prefix length L
is read from ``past_k`` — torch.export can't express a ``L + q`` derived dim from
two independent dynamic dims, so the two stay decoupled.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models._hf import resolve_rope_theta


class StatelessAttention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", dim // self.n_heads)
        self.qkv_proj = nn.Linear(
            dim,
            (self.n_heads + 2 * self.n_kv_heads) * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.rope = initialize_rope(base=resolve_rope_theta(config))

    def forward(self, x, position_ids, past_k, past_v):
        b, q_len, _ = x.shape
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        qkv = self.qkv_proj(x).reshape(b, q_len, nh + 2 * nkv, hd).permute(0, 2, 1, 3)
        query = qkv.narrow(1, 0, nh)
        key = qkv.narrow(1, nh, nkv)
        value = qkv.narrow(1, nh + nkv, nkv)

        # position_ids already holds exactly the q query positions ([L, L+q)).
        query = self.rope(query, position_ids=position_ids)
        key = self.rope(key, position_ids=position_ids)

        # concat the prefix cache (length L = offset) with the new keys/values
        full_k = torch.cat([past_k, key], dim=2)  # (b, nkv, L+q, hd)
        full_v = torch.cat([past_v, value], dim=2)
        k_len = full_k.shape[2]
        scale = 1.0 / math.sqrt(hd)

        # GQA via query grouping (avoids expand+reshape of K/V, which produces
        # unprovable dynamic-shape guards). query: (b, nh, q, hd) ->
        # (b, nkv, rep, q, hd); broadcast against (b, nkv, 1, k_len, hd).
        rep = nh // nkv
        qg = query.contiguous().reshape(b, nkv, rep, q_len, hd)
        kg = full_k.unsqueeze(2)  # (b, nkv, 1, k_len, hd)
        vg = full_v.unsqueeze(2)
        scores = torch.matmul(qg, kg.transpose(-1, -2)) * scale  # (b, nkv, rep, q, k_len)

        q_idx = torch.arange(q_len, device=x.device).unsqueeze(-1)
        k_idx = torch.arange(k_len, device=x.device).unsqueeze(0)
        allowed = k_idx <= (k_len - q_len) + q_idx
        neg = torch.finfo(scores.dtype).min
        scores = scores + torch.where(
            allowed, torch.zeros((), dtype=scores.dtype), torch.full((), neg, dtype=scores.dtype)
        )
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, vg)  # (b, nkv, rep, q, hd)
        out = out.reshape(b, nh, q_len, hd).permute(0, 2, 1, 3).contiguous().reshape(b, q_len, nh * hd)
        return self.o_proj(out), key, value


class StatelessBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        from coreai_models.primitives.macos.mlp import MLP

        self.self_attn = StatelessAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, position_ids, past_k, past_v):
        r, nk, nv = self.self_attn(self.input_layernorm(x), position_ids, past_k, past_v)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, nk, nv


class StatelessBaseLM(nn.Module):
    """Lower text stack: ids -> hidden (norm = Identity), returns new k/v."""

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [StatelessBlock(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids, position_ids, past_k, past_v):
        h = self.embed_tokens(input_ids)
        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            h, nk, nv = layer(h, position_ids, past_k[i], past_v[i])
            new_k.append(nk)
            new_v.append(nv)
        return h, torch.stack(new_k, 0), torch.stack(new_v, 0)


class StatelessTTSLM(nn.Module):
    """Upper acoustic stack: inputs_embeds -> hidden (final RMSNorm)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [StatelessBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs_embeds, position_ids, past_k, past_v):
        h = inputs_embeds
        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            h, nk, nv = layer(h, position_ids, past_k[i], past_v[i])
            new_k.append(nk)
            new_v.append(nv)
        return self.norm(h), torch.stack(new_k, 0), torch.stack(new_v, 0)
