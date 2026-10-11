import torch

from latticelm.config import LatticeConfig
from latticelm.model import CausalAttention, LatticeLM, apply_rope
from latticelm.co4_reference import rms_unit
import torch.nn.functional as F


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


def test_modern_dense_control_uses_same_qk_norm_and_gqa_rule() -> None:
    cfg = LatticeConfig(vocab_size=64, d_model=48, n_layers=1, n_heads=6,
                        n_kv_heads=2, ffn_hidden=96, architecture="lattice",
                        qk_norm=True, real_gqa=True)
    torch.manual_seed(7)
    attention = CausalAttention(cfg).double()
    x = torch.randn(2, 7, 48, dtype=torch.double, requires_grad=True)
    actual = attention(x)
    q = attention.q_proj(x).view(2, 7, 6, 8).transpose(1, 2)
    k = attention.k_proj(x).view(2, 7, 2, 8).transpose(1, 2)
    v = attention.v_proj(x).view(2, 7, 2, 8).transpose(1, 2)
    expected = F.scaled_dot_product_attention(
        apply_rope(rms_unit(q)), apply_rope(rms_unit(k)).repeat_interleave(3, 1),
        v.repeat_interleave(3, 1), is_causal=True, dropout_p=0.0)
    expected = attention.o_proj(expected.transpose(1, 2).contiguous().view(2, 7, 48))
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-8)
    changed = x.detach().clone(); changed[:, 5:] += 10
    assert torch.allclose(attention(changed)[:, :5], actual[:, :5], atol=1e-10, rtol=1e-8)
    actual.sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in attention.parameters())
