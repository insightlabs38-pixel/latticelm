from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.final_recipe import make_optimizer
from latticelm.model import build_model
from latticelm.posthack_variants import PHMLinear, build_posthack_model


def config() -> LatticeConfig:
    return LatticeConfig(vocab_size=64, d_model=48, n_layers=2, n_heads=6,
                         n_kv_heads=2, ffn_hidden=96, context_length=16,
                         architecture="co4_causal", real_gqa=True, qk_norm=True,
                         tie_embeddings=False)


def test_phm4_materialization_forward_and_gradients() -> None:
    with pytest.raises(ValueError):
        PHMLinear(7, 12)
    layer = PHMLinear(8, 12).double()
    x = torch.randn(3, 8, dtype=torch.double, requires_grad=True)
    materialized = torch.zeros(12, 8, dtype=torch.double)
    for index in range(4):
        for out_group in range(4):
            for in_group in range(4):
                materialized[out_group*3:(out_group+1)*3, in_group*2:(in_group+1)*2] += (
                    layer.algebra[index, out_group, in_group] * layer.blocks[index])
    assert torch.allclose(layer.materialized_weight(), materialized, atol=1e-12)
    assert torch.allclose(layer(x), F.linear(x, materialized), atol=1e-12)
    layer(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.algebra.grad is not None and torch.isfinite(layer.algebra.grad).all()
    assert layer.blocks.grad is not None and torch.isfinite(layer.blocks.grad).all()


def test_phm4_ffn_scope_count_causality_and_state_roundtrip(tmp_path) -> None:
    cfg = config()
    torch.manual_seed(17)
    control = build_model(cfg)
    torch.manual_seed(17)
    model = build_posthack_model(cfg, "phm4_ffn")
    assert model.blocks[0].attn.state_dict().keys() == control.blocks[0].attn.state_dict().keys()
    assert all(torch.equal(model.blocks[0].attn.state_dict()[k], control.blocks[0].attn.state_dict()[k])
               for k in control.blocks[0].attn.state_dict())
    dense_ffn = 3 * cfg.d_model * cfg.ffn_hidden
    phm_ffn = 128 + 3 * cfg.d_model * cfg.ffn_hidden // 4
    assert sum(p.numel() for p in control.parameters()) - sum(p.numel() for p in model.parameters()) == (
        dense_ffn - phm_ffn) * cfg.n_layers
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    y = torch.randint(0, cfg.vocab_size, (2, 10))
    first, loss = model(x, y)
    altered = x.clone(); altered[:, 7:] = torch.randint(0, cfg.vocab_size, (2, 3))
    later, _ = model(altered)
    assert torch.allclose(first[:, :7], later[:, :7], atol=1e-6)
    optimizer, _ = make_optimizer(model, "muon_hybrid", 8e-4, .1, (.9, .95))
    loss.backward()
    assert torch.isfinite(loss)
    optimizer.step()
    checkpoint = tmp_path / "phm.pt"
    torch.save({"model": model.state_dict(), "optimizers": optimizer.state_dict()}, checkpoint)
    clone = build_posthack_model(cfg, "phm4_ffn")
    other, _ = make_optimizer(clone, "muon_hybrid", 8e-4, .1, (.9, .95))
    state = torch.load(checkpoint, weights_only=False)
    clone.load_state_dict(state["model"])
    other.load_state_dict(state["optimizers"])
    assert torch.equal(model(x)[0], clone(x)[0])
    assert torch.equal(model.generate(x[:1], 2), clone.generate(x[:1], 2))
