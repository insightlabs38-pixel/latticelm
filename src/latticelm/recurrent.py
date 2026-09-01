from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import LatticeConfig
from .model import apply_rope


class MatrixLSTMMixer(nn.Module):
    """Causal matrix-memory mixer following the published mLSTM recurrence.

    With fixed head width, the sequential axis is O(sequence length). It is a
    clear PyTorch reference implementation, not a fused scan kernel.
    """
    def __init__(self, config: LatticeConfig) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.gates = nn.Linear(config.d_model, config.n_heads * 3)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (part.view(shape).transpose(1, 2) for part in (q, k, v))
        q, k = apply_rope(q), apply_rope(k)
        gates = self.gates(x).view(batch, length, self.heads, 3).permute(0, 2, 1, 3)
        forget, input_gate, output_gate = torch.sigmoid(gates).unbind(-1)
        memory = x.new_zeros(batch, self.heads, self.head_dim, self.head_dim)
        normalizer = x.new_zeros(batch, self.heads, self.head_dim)
        outputs = []
        for t in range(length):
            f, i, o = (gate[:, :, t] for gate in (forget, input_gate, output_gate))
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
            memory = f[..., None, None] * memory + i[..., None, None] * torch.einsum("bhd,bhe->bhde", vt, kt)
            normalizer = f[..., None] * normalizer + i[..., None] * kt
            numerator = torch.einsum("bhde,bhe->bhd", memory, qt)
            denominator = torch.einsum("bhd,bhd->bh", normalizer, qt).abs().clamp_min(1.0)
            outputs.append(o[..., None] * numerator / denominator[..., None])
        y = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out(y)


class LocalCausalAttention(nn.Module):
    """Causal sliding-window attention, O(T * window) attention pairs."""
    def __init__(self, config: LatticeConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim, self.window = config.d_model // config.n_heads, config.local_attention_window
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
        index = torch.arange(length, device=x.device)
        mask = (index[None, :] <= index[:, None]) & (index[None, :] > index[:, None] - self.window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, length, -1))
