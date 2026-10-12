"""Keep common D/N checkpoints by hardlink without copying active weights."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from posthack_campaign import EVENTS, RESEARCH, event
from posthack_schedule import write as write_schedule

TARGETS = {10_000_000: "10m", 50_000_000: "50m"}
EXPECTED_IDS = {
    "A8-CO4-S1337", "A8-DENSE-S1337", "A8-CO4-S2026", "A8-DENSE-S2026",
    "A16-CO4-S1337", "A16-DENSE-S1337", "A32-CO4-S1337", "A32-DENSE-S1337",
}


def maybe_preserve(row: dict) -> None:
    experiment_id = row.get("experiment_id")
    if row.get("event") != "CHECKPOINT" or experiment_id not in EXPECTED_IDS:
        return
    count = row.get("logical_tokens")
    if not isinstance(count, int):
        return
    run = RESEARCH / "runs" / experiment_id
    latest = run / "latest.pt"
    if not latest.exists():
        return
    sidecar = latest.with_suffix(".sha256")
    if not sidecar.exists() or sidecar.read_text().strip() != row.get("checkpoint_sha256"):
        return
    for threshold, name in TARGETS.items():
        if count < threshold or count >= threshold + 10_000_000:
            continue
        destination = run / "milestones" / f"{name}.pt"
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(latest, destination)
        destination.with_suffix(".sha256").write_text(sidecar.read_text())
        event(experiment_id, "MILESTONE_PRESERVED", milestone=name,
              logical_tokens=count, checkpoint_sha256=row["checkpoint_sha256"])


def main() -> None:
    position = 0
    while True:
        with EVENTS.open(encoding="utf-8") as handle:
            handle.seek(position)
            while line := handle.readline():
                try:
                    maybe_preserve(json.loads(line))
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            position = handle.tell()
        try:
            write_schedule()
        except Exception as error:
            event("CAMPAIGN", "SCHEDULE_UPDATE_FAILED", error=f"{type(error).__name__}: {error}")
        if all((result := RESEARCH / "runs" / experiment_id / "result.json").exists() and
               json.loads(result.read_text())["status"] == "COMPLETED"
               for experiment_id in EXPECTED_IDS):
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
