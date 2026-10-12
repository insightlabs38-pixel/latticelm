"""An interrupted calibration must restart without changing its registration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import posthack_calibrate as calibration


def test_phm_calibration_uses_parameter_matched_phm_config():
    from latticelm.config import LatticeConfig
    from latticelm.posthack_variants import build_posthack_model

    for regime in ("16m", "32m", "49m"):
        path = calibration.config_path(regime, "co4", "phm4_ffn")
        assert path.name == f"{regime}_phm4.json"
        cfg = LatticeConfig.from_json(path)
        phm_count = sum(p.numel() for p in build_posthack_model(cfg, "phm4_ffn").parameters())
        base = LatticeConfig.from_json(calibration.config_path(regime, "co4", "baseline"))
        base_count = sum(p.numel() for p in build_posthack_model(base, "baseline").parameters())
        assert abs(phm_count / base_count - 1) < .001


def test_calibration_registration_retry_is_immutable(tmp_path, monkeypatch):
    registry = tmp_path / "registry.jsonl"
    monkeypatch.setattr(calibration, "REGISTRY", registry)
    events = []
    monkeypatch.setattr(calibration, "register", lambda row: registry.write_text(json.dumps(row) + "\n"))
    monkeypatch.setattr(calibration, "event", lambda *args, **kwargs: events.append((args, kwargs)))
    row = {"experiment_id": "CAL-16m-co4-12t", "config_sha256": "config",
           "tokenizer_sha256": "tokenizer", "sampling_manifest_sha256": "manifest",
           "variant": "baseline", "threads": 12, "parameter_count": 16_000_000,
           "active_parameter_count": 16_000_000, "token_budget": 139_264,
           "optimizer_partition": {"weight": "muon"}}
    calibration.ensure_calibration_registration(row)
    original = registry.read_bytes()
    calibration.ensure_calibration_registration(row)
    assert registry.read_bytes() == original
    assert events[-1][0] == (row["experiment_id"], "CALIBRATION_RETRY")

    with pytest.raises(RuntimeError, match="config_sha256"):
        calibration.ensure_calibration_registration({**row, "config_sha256": "changed"})
    assert registry.read_bytes() == original
