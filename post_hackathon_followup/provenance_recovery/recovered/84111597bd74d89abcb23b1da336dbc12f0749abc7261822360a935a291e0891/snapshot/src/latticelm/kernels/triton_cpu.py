"""Optional Triton-CPU forward kernels with explicit PyTorch fallback.

The kernels are intentionally not enabled for training: memory-table backward
uses repeated-index scatter accumulation and the RMSNorm kernel has no custom
backward. `available()` must be true and numerical tests must be run before a
future benchmark may select either kernel.
"""
from __future__ import annotations

import os
import torch

from .reference import conditional_memory_gather, residual_rmsnorm

try:
    import triton
    import triton.language as tl
    _TRITON = True

    @triton.jit
    def _residual_rmsnorm_kernel(x, residual, weight, output, cols: tl.constexpr, BLOCK: tl.constexpr, eps: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < cols
        value = tl.load(x + row * cols + offsets, mask=mask, other=0.0) + tl.load(residual + row * cols + offsets, mask=mask, other=0.0)
        rms = tl.rsqrt(tl.sum(value * value, axis=0) / cols + eps)
        tl.store(output + row * cols + offsets, value * rms * tl.load(weight + offsets, mask=mask, other=0.0), mask=mask)

    @triton.jit
    def _memory_gather_kernel(table2, table3, table4, id2, id3, id4, output, dim: tl.constexpr, BLOCK: tl.constexpr):
        position = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < dim
        base2 = tl.load(id2 + position) * dim
        base3 = tl.load(id3 + position) * dim
        base4 = tl.load(id4 + position) * dim
        value = tl.load(table2 + base2 + offsets, mask=mask, other=0.0) + tl.load(table3 + base3 + offsets, mask=mask, other=0.0) + tl.load(table4 + base4 + offsets, mask=mask, other=0.0)
        tl.store(output + position * dim + offsets, value, mask=mask)
except ImportError:
    _TRITON = False


def available() -> bool:
    if not _TRITON:
        return False
    try:
        # Importability alone is insufficient: a CUDA-only Triton wheel may be
        # installed on a CPU host while exposing no active runtime driver.
        triton.runtime.driver.active.get_current_device()
    except (RuntimeError, AttributeError):
        return False
    return True


def fused_residual_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not available():
        return residual_rmsnorm(x, residual, weight, eps)
    os.environ.setdefault("TRITON_CPU_BACKEND", "1")
    rows, cols = x.numel() // x.shape[-1], x.shape[-1]
    output = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    _residual_rmsnorm_kernel[(rows,)](x, residual, weight, output, cols=cols, BLOCK=block, eps=eps)
    return output


def fused_memory_gather(table2: torch.Tensor, table3: torch.Tensor, table4: torch.Tensor, id2: torch.Tensor, id3: torch.Tensor, id4: torch.Tensor) -> torch.Tensor:
    if not available():
        return conditional_memory_gather(table2, table3, table4, id2, id3, id4)
    dim = table2.shape[1]
    output = torch.empty((*id2.shape, dim), dtype=table2.dtype, device=table2.device)
    _memory_gather_kernel[(id2.numel(),)](table2, table3, table4, id2.reshape(-1), id3.reshape(-1), id4.reshape(-1), output, dim=dim, BLOCK=triton.next_power_of_2(dim))
    return output
