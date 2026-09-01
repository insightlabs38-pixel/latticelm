import torch

from latticelm.config import LatticeConfig
from latticelm.model import LatticeLM


def test_co4_adaptation_is_causal() -> None:
    config = LatticeConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                           ffn_hidden=64, architecture="co4_causal")
    torch.manual_seed(9)
    model = LatticeLM(config).eval()
    first = torch.randint(0, 64, (1, 12))
    second = first.clone(); second[:, 8:] = torch.randint(0, 64, (1, 4))
    logits_a, _ = model(first)
    logits_b, _ = model(second)
    assert torch.allclose(logits_a[:, :8], logits_b[:, :8], atol=1e-6)
