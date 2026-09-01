import torch

from latticelm.config import LatticeConfig
from latticelm.model import LatticeLM
from latticelm.ngrams import causal_ngram_ids


def test_ngram_ids_do_not_change_for_prior_positions_when_future_changes() -> None:
    a = torch.tensor([[5, 6, 7, 8, 9]])
    b = torch.tensor([[5, 6, 7, 42, 43]])
    a_ids, b_ids = causal_ngram_ids(a, 101), causal_ngram_ids(b, 101)
    assert all(torch.equal(x[:, :3], y[:, :3]) for x, y in zip(a_ids, b_ids))


def test_model_logits_are_causal() -> None:
    config = LatticeConfig(vocab_size=300, d_model=32, n_layers=1, n_heads=4, n_kv_heads=1, ffn_hidden=64, memory_enabled=True, memory_slots=64, memory_dim=8)
    model = LatticeLM(config).eval()
    a, b = torch.tensor([[5, 6, 7, 8, 9]]), torch.tensor([[5, 6, 7, 99, 98]])
    with torch.no_grad():
        logits_a, _ = model(a); logits_b, _ = model(b)
    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-5)
