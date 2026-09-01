import torch

from latticelm.kernels.reference import conditional_memory_gather, residual_rmsnorm
from latticelm.kernels.triton_cpu import fused_memory_gather, fused_residual_rmsnorm


def test_triton_wrappers_match_reference_or_fallback() -> None:
    torch.manual_seed(1337)
    x, residual, weight = torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.randn(8)
    assert torch.allclose(fused_residual_rmsnorm(x, residual, weight), residual_rmsnorm(x, residual, weight))
    tables = [torch.randn(16, 4) for _ in range(3)]
    ids = [torch.randint(0, 16, (2, 3)) for _ in range(3)]
    assert torch.allclose(fused_memory_gather(*tables, *ids), conditional_memory_gather(*tables, *ids))
