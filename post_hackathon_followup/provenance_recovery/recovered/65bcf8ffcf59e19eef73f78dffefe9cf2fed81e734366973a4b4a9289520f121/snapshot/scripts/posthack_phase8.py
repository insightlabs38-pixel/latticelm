"""Use surplus time for one paired second-seed 16M replication."""
from __future__ import annotations

import json
import sys
import time

from posthack_archive import upload
from posthack_campaign import RESEARCH, ROOT, STATE, event, sha256, utc
from posthack_phase1 import invoke
from posthack_phase5 import endpoint
from posthack_phase6 import mechanism
from posthack_single_run import register_screen, run_screen

CONFIG = RESEARCH / "configs/16m_co4_seed2026.json"
CONTROL = "R16-S2026-CONTROL"
CANDIDATE = "R16-S2026-REPLICATION"


def main() -> None:
    selection_path = RESEARCH / "promotion_selection.json"
    if not selection_path.exists():
        raise RuntimeError("promotion selection prerequisite missing")
    state = json.loads(STATE.read_text())
    hours = (state["archive_start_epoch"] - time.time()) / 3600
    selection = json.loads(selection_path.read_text())["independent_winners"]
    parent = selection.get("architecture") or selection.get("data")
    if not parent or hours < 30:
        reason = "no promotable mechanism" if not parent else f"only {hours:.1f} hours remain before reserve"
        (RESEARCH / "SECOND_SEED_REPLICATION.md").write_text(
            f"# Second-seed replication\n\nNot launched: {reason}.\n")
        event(CANDIDATE, "RUN_REJECTED", reason=reason)
        return
    if not CONFIG.exists():
        source = json.loads((RESEARCH / "configs/16m_co4.json").read_text())
        source["seed"] = 2026
        CONFIG.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    calibrated = state["throughput_calibration"]["16m/co4/16"]
    register_screen(experiment_id=CONTROL, config_path=CONFIG, variant="baseline",
                    budget=150_000_000, parent_control="A16-CO4-S1337",
                    hypothesis="Independent seed-2026 control for the strongest 16M mechanism",
                    throughput_tps=calibrated["tokens_per_second"] * .9,
                    estimate_provenance=calibrated["experiment_id"])
    run_screen(CONTROL, CONFIG, 16, calibrated["tokens_per_second"] * .9)
    variant, extra = mechanism(parent, 150_000_000)
    mixture = extra.pop("mixture_weights", None)
    architecture = variant if variant != "baseline" else "co4"
    measured = state["throughput_calibration"].get(f"16m/{architecture}/16", calibrated)
    register_screen(experiment_id=CANDIDATE, config_path=CONFIG, variant=variant,
                    budget=150_000_000, parent_control=CONTROL,
                    hypothesis=f"Replicate {parent} against a matched control under independent seed 2026",
                    throughput_tps=measured["tokens_per_second"] * .9,
                    estimate_provenance=measured["experiment_id"],
                    mixture_weights=mixture, extra_fields=extra)
    run_screen(CANDIDATE, CONFIG, 16, measured["tokens_per_second"] * .9)
    base = endpoint(CONTROL)
    new = endpoint(CANDIDATE)
    (RESEARCH / "SECOND_SEED_REPLICATION.md").write_text(
        "# Second-seed replication\n\n"
        f"The 16M mechanism `{parent}` was replicated at seed 2026 with a newly trained matched Co4 control. "
        "Both use 150M logical tokens and the same tokenizer, source corpus and evaluation surface.\n\n"
        "| Arm | Held-out loss | WikiText BPB | Four-task raw accuracy |\n"
        "|---|---:|---:|---:|\n"
        f"| Control | {base[0]:.5f} | {base[1]:.5f} | {base[2]:.4f} |\n"
        f"| Replication | {new[0]:.5f} | {new[1]:.5f} | {new[2]:.4f} |\n\n"
        "The first-seed and second-seed signs should be read together; a one-point capability difference remains noisy without paired uncertainty.\n")
    upload([RESEARCH / "SECOND_SEED_REPLICATION.md"], "SECOND-SEED-REPORT")


if __name__ == "__main__":
    main()
