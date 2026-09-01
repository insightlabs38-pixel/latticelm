import torch

from latticelm.config import LatticeConfig
from latticelm.model import build_model


def test_co4_inspired_is_causal_and_under_cap() -> None:
    config = LatticeConfig(architecture="co4_inspired", vocab_size=300, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, ffn_hidden=64, memory_enabled=True, memory_slots=64, memory_dim=8)
    model = build_model(config).eval()
    a, b = torch.tensor([[5, 6, 7, 8]]), torch.tensor([[5, 6, 99, 98]])
    with torch.no_grad():
        left, _ = model(a); right, _ = model(b)
    assert torch.allclose(left[:, :2], right[:, :2], atol=1e-5)
    assert model.parameter_breakdown()["total"] < 50_000_000
