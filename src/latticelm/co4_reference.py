"""Causal LM adaptation of the ARIA-Funded-TREND/IHMS MOD mechanism."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .model import apply_rope


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
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.latent_q = nn.Parameter(torch.empty(1, self.heads, 1, self.head_dim))
        self.latent_k = nn.Parameter(torch.empty(1, self.heads, 1, self.head_dim))
        self.latent_v = nn.Parameter(torch.empty(1, self.heads, 1, self.head_dim))
        nn.init.normal_(self.latent_q, std=0.02)
        nn.init.normal_(self.latent_k, std=0.02)
        nn.init.normal_(self.latent_v, std=0.02)

    @staticmethod
    def mod(receptive: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return F.relu6(receptive.square() + 2 * receptive + context * (1 + receptive.abs()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope(self.mod(self.latent_q, q))
        k = apply_rope(self.mod(self.latent_k, k))
        v = self.mod(self.latent_v, v)
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.output(result.transpose(1, 2).contiguous().view(batch, length, -1))
