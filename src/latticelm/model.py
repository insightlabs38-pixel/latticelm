from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import LatticeConfig
from .memory import ConditionalMemory, RMSNorm


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """RoPE for [B,H,T,D], with an even head dimension."""
    _, _, length, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = torch.arange(length, device=x.device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    output = torch.empty_like(x)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output


class CausalAttention(nn.Module):
    def __init__(self, config: LatticeConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.head_dim * config.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.head_dim * config.n_kv_heads, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q), apply_rope(k)
        repeat = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, length, -1))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self, config: LatticeConfig) -> None:
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(config.d_model), CausalAttention(config)
        self.ffn_norm, self.ffn = RMSNorm(config.d_model), SwiGLU(config.d_model, config.ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class LatticeLM(nn.Module):
    def __init__(self, config: LatticeConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        # Construct memory only after common modules are initialized: with the
        # same seed, dense and Lattice begin from identical backbone weights.
        self.memory = ConditionalMemory(config.memory_slots, config.memory_dim, config.d_model) if config.memory_enabled else None

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.token_embedding(tokens)
        if self.memory is not None:
            x = self.memory(tokens, x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    def parameter_breakdown(self) -> dict[str, int]:
        memory = self.memory.parameter_count if self.memory is not None else 0
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embeddings = self.token_embedding.weight.numel()
        return {"token_embeddings_tied_head": embeddings, "conditional_memory": memory, "backbone_and_norms": total - embeddings - memory, "total": total}

    def generate(self, tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(tokens[:, -self.config.context_length :])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
