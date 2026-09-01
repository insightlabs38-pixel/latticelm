import torch

from latticelm.kernels.reference import residual_rmsnorm
from latticelm.memory import RMSNorm


def test_residual_rmsnorm_matches_composition() -> None:
    torch.manual_seed(1337)
    x, residual, weight = torch.randn(2, 3, 16), torch.randn(2, 3, 16), torch.randn(16)
    expected = RMSNorm(16); expected.weight.data.copy_(weight)
    assert torch.allclose(residual_rmsnorm(x, residual, weight), expected(x + residual), atol=1e-6)
