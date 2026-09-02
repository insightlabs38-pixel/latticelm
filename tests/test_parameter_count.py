from latticelm.config import LatticeConfig
from latticelm.model import LatticeLM, build_model
import torch


def test_tied_embeddings_and_parameter_cap() -> None:
    model = LatticeLM(LatticeConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=1, ffn_hidden=128, memory_enabled=True, memory_slots=128, memory_dim=16))
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    assert model.parameter_breakdown()["total"] < 50_000_000
    assert model.parameter_breakdown()["conditional_memory"] > 0


def test_dense_and_lattice_share_seeded_backbone_initialization() -> None:
    base = dict(vocab_size=300, d_model=32, n_layers=1, n_heads=4, n_kv_heads=1, ffn_hidden=64)
    torch.manual_seed(1337); dense = LatticeLM(LatticeConfig(**base))
    torch.manual_seed(1337); lattice = LatticeLM(LatticeConfig(**base, memory_enabled=True, memory_slots=64, memory_dim=8))
    assert torch.equal(dense.token_embedding.weight, lattice.token_embedding.weight)
    assert all(torch.equal(a, b) for a, b in zip(dense.blocks.parameters(), lattice.blocks.parameters()))


def test_phase4_parameter_controls() -> None:
    matched = build_model(LatticeConfig.from_json("configs/phase4_dense_param_matched.json"))
    co4 = build_model(LatticeConfig.from_json("configs/tournament_co4_round3.json"))
    untied = build_model(LatticeConfig.from_json("configs/phase4_co4_untied_500k.json"))
    assert abs(sum(p.numel() for p in matched.parameters()) / sum(p.numel() for p in co4.parameters()) - 1) < 0.01
    assert sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in co4.parameters()) == 4096 * 256
