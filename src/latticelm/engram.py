"""Small, causal adaptation of the official DeepSeek Engram demo."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from .memory import RMSNorm


class MiniEngram(nn.Module):
    """Reference-style multi-head n-gram lookup, contextual gate and short conv."""

    def __init__(self, config) -> None:
        super().__init__()
        self.orders = tuple(config.memory_orders)
        self.heads = config.memory_heads
        self.slots = config.memory_slots
        self.head_dim = config.memory_dim
        if not config.memory_token_map_path:
            raise ValueError("mini_engram requires memory_token_map_path")
        report = json.loads(Path(config.memory_token_map_path).read_text())
        self.register_buffer("compressed_ids", torch.tensor(report["memory_token_map"], dtype=torch.long))
        self.tables = nn.ModuleList([
            nn.Embedding(self.slots, self.head_dim)
            for _order in self.orders for _head in range(self.heads)
        ])
        width = len(self.orders) * self.heads * self.head_dim
        self.key_proj = nn.Linear(width, config.d_model, bias=False)
        self.value_proj = nn.Linear(width, config.d_model, bias=False)
        self.key_norm, self.query_norm = RMSNorm(config.d_model), RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.memory_dropout)
        self.conv_enabled = config.memory_conv_enabled
        if self.conv_enabled:
            self.short_conv = nn.Conv1d(config.d_model, config.d_model,
                                        kernel_size=config.memory_conv_kernel,
                                        dilation=max(self.orders), groups=config.d_model,
                                        bias=False, padding=(config.memory_conv_kernel - 1) * max(self.orders))
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.value_proj.weight, std=0.02)

    def _hashes(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        tokens = self.compressed_ids[tokens]
        batch, length = tokens.shape
        hashes = []
        for order_index, order in enumerate(self.orders):
            shifted = []
            for offset in range(order):
                if offset == 0:
                    shifted.append(tokens)
                else:
                    shifted.append(torch.cat((torch.zeros(batch, offset, dtype=tokens.dtype, device=tokens.device),
                                              tokens[:, :length - offset]), dim=1))
            for head in range(self.heads):
                # Stable independent odd multipliers per order/head; arithmetic
                # stays below int64 overflow for this 4K compressed vocabulary.
                value = torch.zeros_like(tokens)
                seed = 10_007 * (1 + order_index * self.heads + head)
                for position, part in enumerate(shifted):
                    value ^= part * (seed + 2 * position + 1)
                hashes.append(value.remainder(self.slots))
        return hashes

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        embeddings = torch.cat([table(ids) for table, ids in zip(self.tables, self._hashes(tokens))], dim=-1)
        key = self.key_norm(self.key_proj(embeddings))
        query = self.query_norm(hidden)
        similarity = (key * query).sum(-1) / math.sqrt(hidden.shape[-1])
        transformed = similarity.sign() * similarity.abs().clamp_min(1e-6).sqrt()
        value = torch.sigmoid(transformed).unsqueeze(-1) * self.value_proj(self.dropout(embeddings))
        if self.conv_enabled:
            length = value.shape[1]
            convolved = self.short_conv(value.transpose(1, 2))[..., :length].transpose(1, 2)
            value = value + torch.nn.functional.silu(convolved)
        return hidden + value

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
