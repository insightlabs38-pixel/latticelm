from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import posthack_phase8 as phase8


def test_second_seed_gate_budgets_both_arms_and_protected_time(tmp_path, monkeypatch):
    now = 1_000_000.0
    research = tmp_path / "research"
    research.mkdir()
    state = research / "controller_state.json"
    state.write_text(json.dumps({"archive_start_epoch": now + 7 * 3600}))
    monkeypatch.setattr(phase8, "RESEARCH", research)
    monkeypatch.setattr(phase8, "STATE", state)
    monkeypatch.setattr(phase8.time, "time", lambda: now)
    monkeypatch.setattr(phase8, "remaining_tokens", lambda name, budget: 10_000_000)
    arms = [("co4", 10_000), ("dense", 10_000)]
    fits, _ = phase8.pair_fits(arms, protected_hours=1)
    assert not fits  # both arms need 3.5 reserved hours each, plus protected time
    state.write_text(json.dumps({"archive_start_epoch": now + 10 * 3600}))
    fits, _ = phase8.pair_fits(arms, protected_hours=1)
    assert fits
    (research / "runs/dense").mkdir(parents=True)
    (research / "runs/dense/hf_archive_receipt.json").write_text("{}")
    state.write_text(json.dumps({"archive_start_epoch": now + 5 * 3600}))
    fits, _ = phase8.pair_fits(arms, protected_hours=1)
    assert fits  # archived dense arm no longer consumes forecast time
