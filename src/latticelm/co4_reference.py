"""Causal LM adaptation of the ARIA-Funded-TREND/IHMS MOD mechanism."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .model import apply_rope


def rms_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free per-head RMS normalization used by optional QK-Norm."""
    return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype)


class CausalCo4Attention(nn.Module):
    """Apply the reference awake MOD law to Q/K/V before causal attention.

    IHMS is currently a vision/RL reference and its published vision operator
    uses non-causal CLS-to-patch aggregation. For autoregressive LM use, this
    adaptation retains the exact elementwise MOD law and learned latent
    receptive streams, but replaces non-causal top-k patch readout with causal
    SDPA. It is therefore not presented as an exact reproduction of the vision
    architecture.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.real_gqa = bool(config.real_gqa)
        self.kv_heads = config.n_kv_heads if self.real_gqa else config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.real_gqa:
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            self.k_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=False)
        else:
            # Checkpoint-compatible historical path. Final-recipe configs are
            # required to set real_gqa=true and are certified separately.
            self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.latent_q = nn.Parameter(torch.empty(1, self.heads, 1, self.head_dim))
        self.latent_k = nn.Parameter(torch.empty(1, self.kv_heads, 1, self.head_dim))
        self.latent_v = nn.Parameter(torch.empty(1, self.kv_heads, 1, self.head_dim))
        self.qk_norm = bool(config.qk_norm)
        nn.init.normal_(self.latent_q, std=0.02)
        nn.init.normal_(self.latent_k, std=0.02)
        nn.init.normal_(self.latent_v, std=0.02)

    @staticmethod
    def mod(receptive: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return F.relu6(receptive.square() + 2 * receptive + context * (1 + receptive.abs()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        if self.real_gqa:
            q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        else:
            q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = self.mod(self.latent_q, q)
        k = self.mod(self.latent_k, k)
        v = self.mod(self.latent_v, v)
        if self.qk_norm:
            q, k = rms_unit(q), rms_unit(k)
        q, k = apply_rope(q), apply_rope(k)
        repeat = self.heads // self.kv_heads
        # Explicit repetition is the numerical reference path and works on the
        # currently deployed CPU SDPA backend. The projections themselves are
        # true grouped-query projections, so K/V parameters and GEMM work scale
        # with n_kv_heads rather than n_heads.
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.output(result.transpose(1, 2).contiguous().view(batch, length, -1))
