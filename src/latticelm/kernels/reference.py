from __future__ import annotations

import torch


def residual_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r = x + residual
    return r * torch.rsqrt(r.float().square().mean(-1, keepdim=True) + eps).to(r.dtype) * weight


def conditional_memory_gather(table2: torch.Tensor, table3: torch.Tensor, table4: torch.Tensor, id2: torch.Tensor, id3: torch.Tensor, id4: torch.Tensor) -> torch.Tensor:
    return table2[id2] + table3[id3] + table4[id4]
