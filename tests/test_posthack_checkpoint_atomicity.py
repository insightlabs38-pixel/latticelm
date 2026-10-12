from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import posthack_train


def test_interrupted_checkpoint_publish_preserves_resumable_previous(tmp_path, monkeypatch):
    posthack_train.save_checkpoint(tmp_path, {"value": torch.tensor([1])})
    assert posthack_train.valid_checkpoint(tmp_path / "latest.pt")
    original_replace = posthack_train.os.replace

    def fail_latest_checksum(source, destination):
        if Path(destination) == tmp_path / "latest.sha256":
            raise OSError("simulated publish interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(posthack_train.os, "replace", fail_latest_checksum)
    with pytest.raises(OSError, match="simulated"):
        posthack_train.save_checkpoint(tmp_path, {"value": torch.tensor([2])})
    assert posthack_train.valid_checkpoint(tmp_path / "previous.pt")
    assert not posthack_train.valid_checkpoint(tmp_path / "latest.pt")
    assert int(torch.load(tmp_path / "previous.pt", weights_only=False)["value"][0]) == 1


def test_best_checkpoint_survives_later_rolling_updates(tmp_path):
    first = posthack_train.save_checkpoint(tmp_path, {"value": torch.tensor([1])})
    posthack_train.preserve_best_checkpoint(tmp_path, first)
    posthack_train.save_checkpoint(tmp_path, {"value": torch.tensor([2])})
    posthack_train.save_checkpoint(tmp_path, {"value": torch.tensor([3])})
    assert posthack_train.valid_checkpoint(tmp_path / "best.pt")
    assert int(torch.load(tmp_path / "best.pt", weights_only=False)["value"][0]) == 1
    assert int(torch.load(tmp_path / "latest.pt", weights_only=False)["value"][0]) == 3
