import torch

from latticelm.config import LatticeConfig
from latticelm.model import LatticeLM
from latticelm.train import save_checkpoint


def test_checkpoint_round_trip(tmp_path) -> None:
    config = LatticeConfig(vocab_size=300, d_model=32, n_layers=1, n_heads=4, n_kv_heads=1, ffn_hidden=64)
    first = LatticeLM(config); optimizer = torch.optim.AdamW(first.parameters())
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(destination, first, optimizer, 3, config, "test")
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    second = LatticeLM(config); second.load_state_dict(payload["model"])
    assert payload["step"] == 3
    assert all(torch.equal(x, y) for x, y in zip(first.parameters(), second.parameters()))
