from __future__ import annotations

import torch

from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.posthack_variants import ValueResidualCo4LM, build_posthack_model


def config() -> LatticeConfig:
    return LatticeConfig(vocab_size=64, d_model=48, n_layers=3, n_heads=6,
                         n_kv_heads=2, ffn_hidden=96, context_length=16,
                         architecture="co4_causal", real_gqa=True, qk_norm=True,
                         tie_embeddings=False)


def test_value_residual_noop_matches_control_and_counts_parameters() -> None:
    cfg = config()
    torch.manual_seed(13)
    control = build_model(cfg).double().eval()
    torch.manual_seed(13)
    variant = ValueResidualCo4LM(cfg).double().eval()
    assert sum(p.numel() for p in variant.parameters()) - sum(p.numel() for p in control.parameters()) == 2 * (cfg.n_layers - 1)
    for block in variant.blocks[1:]:
        block.attn.mix_first.data.zero_()
        block.attn.mix_current.data.fill_(1)
    tokens = torch.randint(0, cfg.vocab_size, (2, 10))
    actual, actual_loss = variant(tokens, tokens)
    expected, expected_loss = control(tokens, tokens)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-8)
    assert torch.allclose(actual_loss, expected_loss, atol=1e-10, rtol=1e-8)
    assert variant.blocks[0].attn.v_proj.weight.shape == (16, 48)


def test_value_residual_causality_gradients_and_checkpoint(tmp_path) -> None:
    torch.manual_seed(27)
    model = build_posthack_model(config(), "value_residual").double()
    tokens = torch.randint(0, 64, (2, 10))
    targets = torch.randint(0, 64, (2, 10))
    logits, loss = model(tokens, targets)
    changed = tokens.clone(); changed[:, 7:] = torch.randint(0, 64, (2, 3))
    changed_logits, _ = model(changed)
    assert torch.allclose(logits[:, :7], changed_logits[:, :7], atol=1e-10, rtol=1e-8)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].attn.v_proj.weight.grad is not None
    for block in model.blocks[1:]:
        assert block.attn.v_proj.weight.grad is not None
        assert block.attn.mix_first.grad is not None
        assert block.attn.mix_current.grad is not None
        assert torch.isfinite(block.attn.mix_first.grad)
        assert torch.isfinite(block.attn.mix_current.grad)
    path = tmp_path / "value_residual.pt"
    torch.save(model.state_dict(), path)
    clone = build_posthack_model(config(), "value_residual").double()
    clone.load_state_dict(torch.load(path, weights_only=True))
    reloaded, _ = clone(tokens)
    assert torch.equal(logits, reloaded)
    assert torch.equal(model.generate(tokens[:1], 2), clone.generate(tokens[:1], 2))


def test_zero_gated_long_skip_matches_value_residual_then_receives_gradient() -> None:
    cfg = config()
    torch.manual_seed(93)
    parent = build_posthack_model(cfg, "value_residual").double()
    torch.manual_seed(93)
    candidate = build_posthack_model(cfg, "value_residual_skip").double()
    candidate.load_state_dict(parent.state_dict(), strict=False)
    tokens = torch.randint(0, cfg.vocab_size, (2, 10))
    target = torch.randint(0, cfg.vocab_size, (2, 10))
    expected, expected_loss = parent(tokens, target)
    actual, actual_loss = candidate(tokens, target)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    actual_loss.backward()
    assert candidate.skip_gate.grad is not None
    assert torch.isfinite(candidate.skip_gate.grad)
