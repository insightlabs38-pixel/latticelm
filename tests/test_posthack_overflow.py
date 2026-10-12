from __future__ import annotations

from pathlib import Path

import torch

from latticelm.config import LatticeConfig
from latticelm.posthack_variants import build_posthack_model


ROOT = Path(__file__).resolve().parents[1]


def test_registered_tiny_engram_is_parameter_matched_and_causal():
    cfg = LatticeConfig.from_json(ROOT / "post_hackathon_research/configs/8m_engram.json")
    torch.manual_seed(77)
    model = build_posthack_model(cfg).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert abs(parameters - 7_991_800) / 7_991_800 < .001
    tokens = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.inference_mode():
        original, _ = model(tokens)
        changed = tokens.clone()
        changed[:, 8:] = torch.randint(0, cfg.vocab_size, (1, 4))
        altered, _ = model(changed)
    torch.testing.assert_close(original[:, :8], altered[:, :8], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(original).all()
