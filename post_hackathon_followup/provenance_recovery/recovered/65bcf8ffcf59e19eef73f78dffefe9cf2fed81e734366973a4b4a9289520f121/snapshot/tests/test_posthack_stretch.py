import torch

from latticelm.config import LatticeConfig
from latticelm.model import LatticeLM
from latticelm.posthack_variants import DifferentialCo4LM, HypersphericalCo4LM


def tiny_config():
    return LatticeConfig(vocab_size=128, d_model=64, ffn_hidden=128,
                         n_heads=4, n_kv_heads=2, n_layers=2,
                         context_length=32, architecture="co4_causal",
                         real_gqa=True, qk_norm=True)


def test_nco4_unit_sphere_gradients_and_resume(tmp_path):
    torch.manual_seed(31)
    model = HypersphericalCo4LM(tiny_config())
    x = torch.randint(0, 128, (2, 16))
    y = model(x)[0]
    assert y.shape == (2, 16, 128)
    assert model.last_hidden_norm_max_error < 1e-5
    y.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.ffn_log_gains.grad is not None
    path = tmp_path / "nco4.pt"
    torch.save(model.state_dict(), path)
    clone = HypersphericalCo4LM(tiny_config())
    clone.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(model(x)[0], clone(x)[0])


def test_differential_disabled_matches_base_and_is_causal():
    torch.manual_seed(32)
    base = LatticeLM(tiny_config())
    candidate = DifferentialCo4LM(tiny_config())
    candidate.load_state_dict(base.state_dict(), strict=False)
    for block in candidate.blocks:
        block.attn.disable_differential = True
    x = torch.randint(0, 128, (2, 16))
    torch.testing.assert_close(base(x)[0], candidate(x)[0], atol=1e-6, rtol=0)
    for block in candidate.blocks:
        block.attn.disable_differential = False
    output = candidate(x)[0]
    changed = x.clone()
    changed[:, 8:] = torch.randint(0, 128, changed[:, 8:].shape)
    torch.testing.assert_close(output[:, :8], candidate(changed)[0][:, :8], atol=1e-6, rtol=0)
    output.square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in candidate.parameters())
