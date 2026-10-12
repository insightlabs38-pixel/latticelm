"""Recompute the campaign queue against measured throughput and archive reserve."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from posthack_campaign import RESEARCH, STATE, atomic_json, utc


def recent_tps(experiment_id: str) -> float | None:
    path = RESEARCH / "runs" / experiment_id / "metrics.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) < 20:
        return None
    left, right = rows[max(0, len(rows) - 101)], rows[-1]
    elapsed = right["wall_seconds"] - left["wall_seconds"]
    return (right["logical_tokens"] - left["logical_tokens"]) / elapsed if elapsed > 0 else None


def estimate() -> dict:
    state = json.loads(STATE.read_text())
    with (RESEARCH / "RUN_REGISTRY.jsonl").open(encoding="utf-8") as handle:
        records = {row["experiment_id"]: row for line in handle if (row := json.loads(line))}
    ids = ([state["active_run"]] if state["active_run"] else []) + [
        item for item in state["queued_runs"] if item != state["active_run"]]
    now = time.time()
    cursor = now
    projections = []
    for experiment_id in ids:
        if experiment_id not in records:
            continue
        record = records[experiment_id]
        config = record["config"]
        regime = f"{round(record['parameter_count'] / 1_000_000)}m"
        architecture = "co4" if config["architecture"] == "co4_causal" else "dense"
        variant = record.get("variant", "baseline")
        calibration_architecture = variant if variant != "baseline" else architecture
        choices = [item for key, item in state["throughput_calibration"].items()
                   if key.startswith(f"{regime}/{calibration_architecture}/")]
        calibration = max(choices, key=lambda row: row["tokens_per_second"]) if choices else None
        speed = recent_tps(experiment_id) if experiment_id == state["active_run"] else None
        if speed is None:
            speed = (calibration["tokens_per_second"] if calibration
                     else record.get("estimated_throughput_tps"))
        run_dir = RESEARCH / "runs" / experiment_id
        result_path = run_dir / "result.json"
        if result_path.exists() and json.loads(result_path.read_text())["status"] == "COMPLETED":
            remaining = 0
        else:
            metrics = run_dir / "metrics.jsonl"
            if metrics.exists():
                with metrics.open(encoding="utf-8") as handle:
                    last = None
                    for line in handle:
                        last = json.loads(line)
                completed = last["logical_tokens"] if last else 0
            else:
                completed = 0
            remaining = max(0, record["token_budget"] - completed)
        train_seconds = remaining / speed if speed else None
        eval_seconds = state["measured_evaluation_seconds"].get(experiment_id, 3600)
        checkpoint_seconds = 900
        safety_seconds = 1800
        if train_seconds is not None:
            cursor += train_seconds + eval_seconds + checkpoint_seconds + safety_seconds
        projections.append({"experiment_id": experiment_id, "remaining_tokens": remaining,
                            "tokens_per_second_used": speed, "training_seconds": train_seconds,
                            "evaluation_seconds_reserved": eval_seconds,
                            "projected_finish_utc": utc(cursor) if speed else None,
                            "fits_archival_reserve": bool(speed and cursor < state["archive_start_epoch"])})
    return {"at": utc(now), "working_deadline_utc": state["working_deadline_utc"],
            "archive_start_utc": utc(state["archive_start_epoch"]),
            "remaining_working_hours": (state["working_deadline_epoch"] - now) / 3600,
            "remaining_training_hours": (state["archive_start_epoch"] - now) / 3600,
            "active_run": state["active_run"], "queued_runs": state["queued_runs"],
            "projections": projections}


def write() -> dict:
    result = estimate()
    atomic_json(RESEARCH / "LIVE_PROJECTIONS.json", result)
    lines = ["# Live schedule", "", f"Updated: {result['at']}",
             f"Working deadline: {result['working_deadline_utc']}",
             f"Archival reserve starts: {result['archive_start_utc']}",
             f"Remaining training window: {result['remaining_training_hours']:.2f} hours.",
             f"Active run: {result['active_run'] or 'none'}.", "",
             "| Run | Remaining tokens | TPS used | Projected finish UTC | Fits reserve |",
             "|---|---:|---:|---|---|"]
    for row in result["projections"]:
        lines.append(f"| {row['experiment_id']} | {row['remaining_tokens']:,} | "
                     f"{row['tokens_per_second_used'] or 0:,.0f} | "
                     f"{row['projected_finish_utc'] or 'unavailable'} | "
                     f"{'yes' if row['fits_archival_reserve'] else 'no'} |")
    lines.extend(["", "Projections are recomputed from current run progress and measured C4A calibration. They include reserved evaluation, checkpoint, and safety time. Active-run TPS uses the latest 100 steps when available.", ""])
    path = RESEARCH / "LIVE_SCHEDULE.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    os.replace(temporary, path)
    return result


if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
