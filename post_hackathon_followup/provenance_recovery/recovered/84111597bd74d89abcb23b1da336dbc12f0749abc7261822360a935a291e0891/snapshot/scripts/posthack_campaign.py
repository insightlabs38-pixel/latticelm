"""Local, append-only control plane for the post-hackathon research campaign.

This module deliberately never invokes git commit or push. It records the
source tree state and stores research output outside the historical artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "post_hackathon_research"
STATE = RESEARCH / "controller_state.json"
REGISTRY = RESEARCH / "RUN_REGISTRY.jsonl"
EVENTS = RESEARCH / "controller_events.jsonl"


def utc(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def init() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    if not (RESEARCH / "FROZEN_HACKATHON_STATE.json").exists():
        raise RuntimeError("frozen hackathon record is required")
    now = time.time()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    state = {
        "schema": "posthack-controller-v1",
        "started_utc": utc(now),
        "started_epoch": now,
        "working_deadline_utc": utc(now + 15 * 86400),
        "working_deadline_epoch": now + 15 * 86400,
        "archival_reserve_hours": 12,
        "archive_start_epoch": now + 15 * 86400 - 12 * 3600,
        "source_commit": commit,
        "frozen_record_sha256": sha256(RESEARCH / "FROZEN_HACKATHON_STATE.json"),
        "active_run": None,
        "queued_runs": [],
        "throughput_calibration": {},
        "measured_evaluation_seconds": {},
        "measured_preprocessing_seconds": {},
        "decisions": [],
    }
    atomic_json(STATE, state)
    append(EVENTS, {"at": utc(now), "event": "CAMPAIGN_STARTED", "working_deadline_utc": state["working_deadline_utc"]})
    return state


def register(record: dict) -> None:
    if not STATE.exists():
        raise RuntimeError("initialize campaign before registering")
    required = {"experiment_id", "hypothesis", "parent_control", "config", "tokenizer_sha256",
                "sampling_manifest_sha256", "code_commit", "seed", "parameter_count",
                "active_parameter_count", "context_length", "optimizer_partition",
                "token_budget", "planned_evaluation_points", "promotion_criterion",
                "abort_criterion", "expected_wall_seconds", "estimate_provenance"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"missing registration fields: {sorted(missing)}")
    if REGISTRY.exists():
        with REGISTRY.open(encoding="utf-8") as handle:
            for line in handle:
                if json.loads(line).get("experiment_id") == record["experiment_id"]:
                    raise ValueError(f"duplicate experiment ID: {record['experiment_id']}")
    append(REGISTRY, {**record, "registered_at": utc(), "status": "REGISTERED"})
    append(EVENTS, {"at": utc(), "event": "RUN_REGISTERED", "experiment_id": record["experiment_id"]})


def event(experiment_id: str, kind: str, **details: object) -> None:
    append(EVENTS, {"at": utc(), "event": kind, "experiment_id": experiment_id, **details})


def set_active(experiment_id: str | None) -> None:
    state = json.loads(STATE.read_text())
    state["active_run"] = experiment_id
    atomic_json(STATE, state)


def record_calibration(result: dict) -> None:
    state = json.loads(STATE.read_text())
    key = f"{result['regime']}/{result['architecture']}/{result['threads']}"
    state["throughput_calibration"][key] = result
    atomic_json(STATE, state)
    event(result["experiment_id"], "CALIBRATION_COMPLETE",
          **{key: value for key, value in result.items() if key != "experiment_id"})


def can_launch(remaining_tokens: int, measured_tps: float, eval_seconds: float,
               checkpoint_seconds: float, safety_seconds: float) -> tuple[bool, float]:
    state = json.loads(STATE.read_text())
    predicted_finish = time.time() + remaining_tokens / measured_tps + eval_seconds + checkpoint_seconds + safety_seconds
    return predicted_finish < state["archive_start_epoch"], predicted_finish


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "status"))
    args = parser.parse_args()
    state = init() if args.action == "init" else json.loads(STATE.read_text())
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
