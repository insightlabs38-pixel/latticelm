import torch
import sys
from pathlib import Path

from latticelm.config import LatticeConfig
from latticelm.final_recipe import make_optimizer, muon_parameter_partition
from latticelm.posthack_variants import MoECo4LM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from posthack_train import update_adaptive_birth_tracker


def config():
    return LatticeConfig(vocab_size=128, d_model=64, ffn_hidden=128,
                         n_heads=4, n_kv_heads=2, n_layers=2,
                         context_length=32, architecture="co4_causal",
                         real_gqa=True, qk_norm=True)


def test_expert_birth_parity_optimizer_state_and_resume(tmp_path):
    torch.manual_seed(43)
    model = MoECo4LM(config(), 4)
    total = sum(p.numel() for p in model.parameters())
    other_paths = sum(sum(p.numel() for p in expert.parameters())
                      for block in model.blocks for expert in block.ffn.experts[1:])
    per_token_active = total - other_paths
    assert model.active_parameter_count() == per_token_active
    model.set_domains(["finemath", "fineweb"])
    optimizer, partition = make_optimizer(model, "muon_hybrid", 8e-4, .1, (.9, .95))
    assert all(group == "adamw" for name, group in partition.items() if "router" in name)
    x = torch.randint(0, 128, (2, 16))
    model(x)[0].square().mean().backward()
    optimizer.step()
    old = model.blocks[0].ffn.experts[0].in_proj.weight
    child = model.blocks[0].ffn.experts[4].in_proj.weight
    muon = optimizer.optimizers[0]
    assert old in muon.state and child not in muon.state
    state_before = {key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in muon.state[old].items()}
    with torch.no_grad():
        before = model(x)[0]
        birth = model.birth("clone")
        after = model(x)[0]
    assert birth["strategy"] == "clone"
    assert model.active_parameter_count() == per_token_active
    torch.testing.assert_close(before, after, atol=1e-6, rtol=0)
    assert child not in muon.state
    for key, value in state_before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, muon.state[old][key])
        else:
            assert value == muon.state[old][key]
    path = tmp_path / "moe.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    clone = MoECo4LM(config(), 4)
    clone_optimizer, _ = make_optimizer(clone, "muon_hybrid", 8e-4, .1, (.9, .95))
    loaded = torch.load(path, weights_only=False)
    clone.load_state_dict(loaded["model"])
    clone_optimizer.load_state_dict(loaded["optimizer"])
    clone.set_domains(["finemath", "fineweb"])
    torch.testing.assert_close(model(x)[0], clone(x)[0], atol=1e-6, rtol=0)


def test_finemath_catchup_routes_to_new_children():
    torch.manual_seed(44)
    model = MoECo4LM(config(), 4)
    x = torch.randint(0, 128, (2, 16))
    model.set_domains(["finemath", "fineweb"])
    model(x)
    model.birth("clone")
    model.set_catchup(True)
    model(x)
    counts = model.blocks[0].ffn.last_domain_counts
    assert counts[3, 4:].sum() >= 8
    assert counts[0, :].sum() == 0
    model.set_catchup(False)
    for block in model.blocks:
        block.ffn.forced_expert = 5
    model(x)
    assert all(int(block.ffn.last_counts[5]) == x.numel() for block in model.blocks)
    for block in model.blocks:
        block.ffn.forced_expert = None
        block.ffn.ablate_expert = 5
    assert torch.isfinite(model(x)[0]).all()


def test_adaptive_birth_requires_sustained_registered_windows():
    tracker = {"steps": 0, "counts": [[0] * 4 for _ in range(4)],
               "loss_sum": [0.0] * 4, "loss_count": [0] * 4,
               "eligible_windows": 0}
    rule = {"window_steps": 2, "consecutive_windows": 2,
            "min_max_load": .31, "min_domain_expert_mi": .01,
            "min_finemath_loss_gap": .03, "min_finemath_examples": 2}
    counts = [[60, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 60]]
    labels = ["fineweb_edu", "finemath"]
    assert update_adaptive_birth_tracker(tracker, counts, labels, [.5, .8], rule) is None
    first = update_adaptive_birth_tracker(tracker, counts, labels, [.5, .8], rule)
    assert first["eligible"] and not first["trigger"]
    assert update_adaptive_birth_tracker(tracker, counts, labels, [.5, .8], rule) is None
    second = update_adaptive_birth_tracker(tracker, counts, labels, [.5, .8], rule)
    assert second["trigger"] and second["consecutive_eligible_windows"] == 2
