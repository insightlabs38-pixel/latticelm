from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import posthack_train


def test_source_validation_restores_training_mode(monkeypatch):
    model = torch.nn.Linear(2, 2).train()

    def validation(chosen_model, values, context, batches):
        chosen_model.eval()
        return 1.0, [1.0]

    monkeypatch.setattr(posthack_train, "validation", validation)
    losses, _ = posthack_train.evaluate(model, {"source": [np.arange(8)]}, 2)
    assert losses == {"source": 1.0}
    assert model.training
    model.eval()
    posthack_train.evaluate(model, {"source": [np.arange(8)]}, 2)
    assert not model.training
