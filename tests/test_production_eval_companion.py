import hashlib
import json
from pathlib import Path

import pytest
import torch

import scripts.run_production_checkpoint_evaluation_companion as companion


def payload(tokens=600):
    return {
        "schema": "final-production-checkpoint-v1", "model": {}, "optimizers": {}, "config": {"learning_rate": .0008},
        "config_sha256": "c" * 64, "tokens": tokens, "step": tokens // 8192,
        "manifest_sha256": "m" * 64, "tokenizer_sha256": "t" * 64,
        "next_batch_sha256": "n" * 64, "validation_loss": 2.9, "source_validation": {"fineweb": 2.9},
    }


def write_checkpoint(root: Path, tokens=600):
    run = root / "final_production_run"; run.mkdir()
    path = run / "latest.pt"; torch.save(payload(tokens), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest(); (run / "latest.sha256").write_text(digest + "\n")
    return path, digest


def test_partial_or_bad_sidecar_is_rejected(tmp_path, monkeypatch):
    path, digest = write_checkpoint(tmp_path)
    monkeypatch.setattr(companion, "RUN", path.parent)
    assert companion.latest_checkpoint() is not None
    (path.parent / "latest.sha256").write_text("bad\n")
    assert companion.latest_checkpoint() is None
    (path.parent / "latest.sha256").write_text(digest + "\n")
    path.write_bytes(path.read_bytes() + b"partial")
    assert companion.latest_checkpoint() is None


def test_checkpoint_event_requires_newer_tokens(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(json.dumps(x) for x in [
        {"event": "CHECKPOINT", "at": 10, "tokens": 500, "sha256": "old"},
        {"event": "CHECKPOINT", "at": 20, "tokens": 600, "sha256": "new"},
    ]) + "\n")
    monkeypatch.setattr(companion, "EVENTS", events)
    assert companion.checkpoint_event_after(11, 500)["tokens"] == 600
    assert companion.checkpoint_event_after(21, 500) is None


def test_terminal_state_is_idempotent(tmp_path, monkeypatch):
    state = tmp_path / "state.json"; state.write_text(json.dumps({"status": "COMPLETE"}))
    monkeypatch.setattr(companion, "STATE", state)
    assert companion.load_state()["status"] in companion.TERMINAL


def test_pin_records_immutable_identity(tmp_path, monkeypatch):
    path, digest = write_checkpoint(tmp_path)
    monkeypatch.setattr(companion, "ART", tmp_path)
    state_file = tmp_path / "state.json"; events_file = tmp_path / "events.jsonl"
    monkeypatch.setattr(companion, "STATE", state_file); monkeypatch.setattr(companion, "EVENT_LOG", events_file)
    event = {"event": "CHECKPOINT", "at": 20, "tokens": 600, "sha256": digest, "validation_loss": 2.9}
    state = {"status": "CHECKPOINT_DETECTED"}
    identity = companion.pin_checkpoint(state, event, (path, payload(), digest))
    assert identity["tokens"] == 600
    assert Path(identity["checkpoint"]).is_file()
    assert hashlib.sha256(Path(identity["checkpoint"]).read_bytes()).hexdigest() == digest
    assert json.loads(state_file.read_text())["status"] == "CHECKPOINT_VERIFIED"
