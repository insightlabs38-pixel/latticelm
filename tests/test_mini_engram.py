import json

import torch

from latticelm.config import LatticeConfig
from latticelm.model import LatticeLM


def test_mini_engram_is_causal_and_has_memory_gradients(tmp_path) -> None:
    report = tmp_path / "tokenizer.report.json"
    report.write_text(json.dumps({"memory_token_map": list(range(64))}))
    config = LatticeConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                           ffn_hidden=64, architecture="mini_engram", memory_enabled=True,
                           memory_slots=101, memory_dim=4, memory_heads=2,
                           memory_token_map_path=str(report), memory_insert_layers=(1,))
    torch.manual_seed(7)
    model = LatticeLM(config).eval()
    first = torch.randint(0, 64, (1, 12))
    second = first.clone(); second[:, 8:] = torch.randint(0, 64, (1, 4))
    logits_a, _ = model(first)
    logits_b, _ = model(second)
    assert torch.allclose(logits_a[:, :8], logits_b[:, :8], atol=1e-6)
    _, loss = model(first, torch.roll(first, -1, dims=1)); loss.backward()
    assert model.memory.tables[0].weight.grad is not None
    assert model.memory.tables[0].weight.grad.abs().sum() > 0
