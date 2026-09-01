import torch

from latticelm.config import LatticeConfig
from latticelm.model import LatticeLM


def test_memory_gradients_are_nonzero() -> None:
    config = LatticeConfig(vocab_size=300, d_model=32, n_layers=1, n_heads=4, n_kv_heads=1, ffn_hidden=64, memory_enabled=True, memory_slots=64, memory_dim=8)
    model = LatticeLM(config)
    tokens = torch.tensor([[5, 6, 7, 8]])
    _, loss = model(tokens, torch.tensor([[6, 7, 8, 9]]))
    loss.backward()
    assert model.memory is not None
    assert model.memory.bigram.weight.grad is not None
    assert model.memory.bigram.weight.grad.abs().sum() > 0
