"""Auditable partial adaptation of the published SLlama reductions.

This deliberately uses the ``InspiredExperimental`` name.  It implements
RRHP, SPMLP, and whole-layer sharing from Omolaoye et al. (EMNLP 2025), but
uses this project's causal GQA instead of the paper's underspecified PWA
permutation construction.  It must not be described as a faithful reference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .memory import RMSNorm
from .model import CausalAttention


class SharedProjectionMLP(nn.Module):
    """SwiGLU whose reduction is the transpose of the value expansion."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.empty(hidden, dim))
        self.gate = nn.Linear(dim, hidden, bias=False)
        nn.init.normal_(self.value, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.linear(x, self.value)
        return F.linear(F.silu(self.gate(x)) * value, self.value.t())


class SharedBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SharedProjectionMLP(config.d_model, config.ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class SLlamaInspiredExperimental(nn.Module):
    """RRHP + SPMLP + repeated shared layer; PWA intentionally omitted."""

    def __init__(self, config) -> None:
        super().__init__()
        if config.d_model % 4:
            raise ValueError("SLlama RRHP requires d_model divisible by four")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model // 4)
        self.block = SharedBlock(config)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.token_embedding(tokens).repeat(1, 1, 4)
        for _ in range(self.config.n_layers):
            x = self.block(x)
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()) if targets is not None else None
        return logits, loss

    def parameter_breakdown(self) -> dict[str, int | bool]:
        embedding = self.token_embedding.weight.numel()
        head = self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"token_embeddings_and_head": embedding + head, "embeddings_tied": False,
                "conditional_memory": 0, "backbone_and_norms": total - embedding - head, "total": total}
