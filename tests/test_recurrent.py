import torch

from latticelm.config import LatticeConfig
from latticelm.model import LatticeLM


def test_hybrid_mixer_is_causal_and_has_finite_gradients() -> None:
    config = LatticeConfig(vocab_size=300, d_model=32, n_layers=2, n_heads=4, n_kv_heads=1, ffn_hidden=64, memory_enabled=True, memory_slots=64, memory_dim=8, mixer_strategy="hybrid", local_attention_window=4)
    model = LatticeLM(config)
    a, b = torch.tensor([[5, 6, 7, 8, 9]]), torch.tensor([[5, 6, 7, 80, 81]])
    logits_a, loss = model(a, torch.tensor([[6, 7, 8, 9, 10]]))
    logits_b, _ = model(b)
    loss.backward()
    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-5)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
