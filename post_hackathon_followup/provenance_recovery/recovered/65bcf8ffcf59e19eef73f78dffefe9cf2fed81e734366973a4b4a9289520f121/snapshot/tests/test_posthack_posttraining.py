import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from posthack_posttrain import retention_objective


def test_reference_kl_zero_for_identical_teacher_and_positive_when_shifted():
    torch.manual_seed(80)
    logits = torch.randn(4, 8, 16, requires_grad=True)
    targets = torch.randint(0, 16, (4, 8))
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 5:] = True
    same, components = retention_objective(logits, targets, mask,
                                           logits[2:].detach().clone(), .1)
    assert abs(float(components["reference_kl"].detach())) < 1e-6
    shifted = logits[2:].detach().clone()
    shifted[..., 0] += 3
    different, shifted_components = retention_objective(logits, targets,
                                                          mask, shifted, .1)
    assert float(shifted_components["reference_kl"].detach()) > 0
    assert different > same
    different.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_prompt_tokens_are_masked_from_completion_loss():
    torch.manual_seed(81)
    logits = torch.randn(4, 8, 16)
    targets = torch.randint(0, 16, (4, 8))
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 6:] = True
    before, _ = retention_objective(logits, targets, mask, None, 0)
    changed = targets.clone()
    changed[:2, :6] = (changed[:2, :6] + 3) % 16
    after, _ = retention_objective(logits, changed, mask, None, 0)
    torch.testing.assert_close(before, after)
