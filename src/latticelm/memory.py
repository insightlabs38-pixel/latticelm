from __future__ import annotations

import torch
from torch import nn

from .ngrams import causal_ngram_ids


class ConditionalMemory(nn.Module):
    def __init__(self, slots: int, memory_dim: int, d_model: int) -> None:
        super().__init__()
        self.slots, self.memory_dim = slots, memory_dim
        self.bigram = nn.Embedding(slots, memory_dim)
        self.trigram = nn.Embedding(slots, memory_dim)
        self.fourgram = nn.Embedding(slots, memory_dim)
        self.projection = nn.Linear(memory_dim, d_model, bias=False)
        self.gate_norm = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)
        for table in (self.bigram, self.trigram, self.fourgram):
            nn.init.normal_(table.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h2, h3, h4 = causal_ngram_ids(tokens, self.slots)
        memory = self.bigram(h2) + self.trigram(h3) + self.fourgram(h4)
        return x + torch.sigmoid(self.gate(self.gate_norm(x))) * self.projection(memory)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight
