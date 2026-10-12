from __future__ import annotations

import torch

from latticelm.config import LatticeConfig
from latticelm.final_recipe import make_optimizer
from latticelm.model import build_model
from latticelm.posthack_variants import ProgressiveDepthCo4LM


def config() -> LatticeConfig:
    return LatticeConfig(vocab_size=64, d_model=48, n_layers=4, n_heads=6,
                         n_kv_heads=2, ffn_hidden=96, context_length=16,
                         architecture="co4_causal", real_gqa=True, qk_norm=True,
                         tie_embeddings=False)


def test_full_depth_identity_and_growth_parity(tmp_path) -> None:
    cfg = config()
    torch.manual_seed(91)
    control = build_model(cfg).eval()
    torch.manual_seed(91)
    progressive = ProgressiveDepthCo4LM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 9))
    progressive.active_depth.fill_(cfg.n_layers)
    for gate in progressive.growth_gates:
        gate.data.fill_(1)
    assert torch.allclose(progressive(tokens)[0], control(tokens)[0], atol=1e-6)
    progressive.active_depth.fill_(progressive.initial_depth)
    before = progressive(tokens)[0]
    progressive.grow(3)
    after = progressive(tokens)[0]
    assert torch.equal(before, after)
    progressive.grow(4)
    assert torch.equal(before, progressive(tokens)[0])
    assert all(float(gate.detach()) == 0 for gate in progressive.growth_gates)
    path = tmp_path / "growth.pt"
    torch.save(progressive.state_dict(), path)
    clone = ProgressiveDepthCo4LM(cfg)
    clone.load_state_dict(torch.load(path, weights_only=True))
    assert int(clone.active_depth) == 4
    assert torch.equal(progressive(tokens)[0], clone(tokens)[0])


def test_existing_optimizer_state_survives_growth_and_new_gate_learns() -> None:
    model = ProgressiveDepthCo4LM(config())
    optimizer, _ = make_optimizer(model, "muon_hybrid", 8e-4, .1, (.9, .95))
    tokens = torch.randint(0, 64, (2, 8))
    _, loss = model(tokens, tokens)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    old_parameter = model.blocks[0].attn.q_proj.weight
    old_state = optimizer.optimizers[0].state[old_parameter]
    assert old_state
    old_state_keys = set(old_state)
    model.grow(3)
    assert optimizer.optimizers[0].state[old_parameter] is old_state
    assert set(old_state) == old_state_keys
    _, loss = model(tokens, tokens)
    loss.backward()
    assert model.growth_gates[0].grad is not None
    assert torch.isfinite(model.growth_gates[0].grad)
    assert old_parameter.grad is not None
    optimizer.step()
    assert torch.isfinite(model.growth_gates[0])
