"""Time-gated paired 16M Co4/dense and selected-mechanism replications."""
from __future__ import annotations

import json
import time

from posthack_archive import upload
from posthack_campaign import RESEARCH, STATE, atomic_json, event, utc
from posthack_phase5 import endpoint
from posthack_phase6 import mechanism
from posthack_single_run import register_screen, remaining_tokens, run_screen

CONFIG = RESEARCH / "configs/16m_co4_seed2026.json"
DENSE_CONFIG = RESEARCH / "configs/16m_dense_seed2026.json"
PHM_CONFIG = RESEARCH / "configs/16m_phm4_seed2026.json"
BUDGET = 150_000_000
CONTROL = "R16-S2026-CONTROL"
CANDIDATE = "R16-S2026-REPLICATION"
MATCHED_DENSE = "A16-DENSE-S2026"


def seed_config(source: str, target) -> None:
    if target.exists():
        return
    config = json.loads((RESEARCH / "configs" / source).read_text())
    config["seed"] = 2026
    target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def calibrated(state: dict, architecture: str) -> dict:
    rows = [state["throughput_calibration"][f"16m/{architecture}/{threads}"]
            for threads in (12, 16)]
    return max(rows, key=lambda row: row["tokens_per_second"])


def pair_fits(arms: list[tuple[str, float]], protected_hours: float) -> tuple[bool, float]:
    state = json.loads(STATE.read_text())
    seconds = protected_hours * 3600
    for name, tps in arms:
        if (RESEARCH / "runs" / name / "hf_archive_receipt.json").exists():
            continue
        seconds += remaining_tokens(name, BUDGET) / tps + 7200 + 1800 + 3600
    finish = time.time() + seconds
    return finish < state["archive_start_epoch"], finish


def freeze_decision(path, name: str, evidence: bool, evidence_reason: str,
                    arms: list[tuple[str, float]], protected_hours: float) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    fits, finish = pair_fits(arms, protected_hours)
    result = {"created_utc": utc(), "run": bool(evidence and fits),
              "evidence_passed": bool(evidence), "estimated_tps": dict(arms),
              "predicted_finish_including_protected_time_utc": utc(finish),
              "protected_downstream_hours": protected_hours,
              "reason": "evidence and measured-time gates passed" if evidence and fits else
                        evidence_reason if not evidence else
                        "paired endpoints and downstream reserve do not fit"}
    atomic_json(path, result)
    event(name, "OPTIONAL_RUN_DECISION", **result)
    return result


def matched_replication(state: dict) -> dict:
    path = RESEARCH / "matched_16m_second_seed_decision.json"
    analysis = json.loads((RESEARCH / "matched_co4_dense_analysis.json").read_text())
    row = next(row for row in analysis["comparisons"] if row["pair"] == "16M seed 1337")
    control_loss = analysis["runs"][row["dense"]]["weighted_heldout_loss"]
    relative = abs(row["heldout_loss_delta_co4_minus_dense"] / control_loss)
    interval = row["paired_window_bootstrap"].get("ci95")
    uncertain = bool(interval and interval[0] <= 0 <= interval[1])
    evidence = uncertain or relative < .005
    co4 = calibrated(state, "co4")
    dense = calibrated(state, "dense")
    rates = [(CONTROL, co4["tokens_per_second"] * .9),
             (MATCHED_DENSE, dense["tokens_per_second"] * .9)]
    decision = freeze_decision(path, "A16-MATCHED-SECOND-SEED", evidence,
                               "first 16M comparison was resolved beyond the registered ambiguity band",
                               rates, 24)
    if decision["run"]:
        seed_config("16m_co4.json", CONFIG)
        seed_config("16m_dense.json", DENSE_CONFIG)
        for name, config, row in ((CONTROL, CONFIG, co4),
                                  (MATCHED_DENSE, DENSE_CONFIG, dense)):
            register_screen(experiment_id=name, config_path=config,
                            variant="baseline", budget=BUDGET,
                            parent_control="A16-CO4-S1337" if name == CONTROL else CONTROL,
                            hypothesis="Independent seed-2026 modern Co4/dense comparison at 16M resolves first-seed ambiguity",
                            throughput_tps=row["tokens_per_second"] * .9,
                            estimate_provenance=row["experiment_id"],
                            extra_fields={"fairness_axes": ["equal_parameters", "equal_tokens",
                                                            "equal_data_order", "wall_clock"],
                                          "second_seed_decision_path": str(path.relative_to(RESEARCH))})
            run_screen(name, config, row["threads"], row["tokens_per_second"] * .9)
    return decision


def mechanism_replication(state: dict) -> tuple[dict, str | None]:
    selection = json.loads((RESEARCH / "promotion_selection.json").read_text())["independent_winners"]
    parent = selection.get("architecture") or selection.get("data")
    base = calibrated(state, "co4")
    variant, extra = mechanism(parent, BUDGET) if parent else ("baseline", {})
    architecture = "co4" if variant == "baseline" else variant
    candidate = calibrated(state, architecture) if parent else base
    rates = [(CONTROL, base["tokens_per_second"] * .9),
             (CANDIDATE, candidate["tokens_per_second"] * .9)]
    decision = freeze_decision(RESEARCH / "mechanism_second_seed_decision.json",
                               CANDIDATE, parent is not None,
                               "no promotable 16M mechanism", rates, 12)
    if decision["run"]:
        seed_config("16m_co4.json", CONFIG)
        candidate_config = CONFIG
        if variant == "phm4_ffn":
            seed_config("16m_phm4.json", PHM_CONFIG)
            candidate_config = PHM_CONFIG
        register_screen(experiment_id=CONTROL, config_path=CONFIG,
                        variant="baseline", budget=BUDGET,
                        parent_control="A16-CO4-S1337",
                        hypothesis="Independent seed-2026 control for the strongest 16M mechanism",
                        throughput_tps=base["tokens_per_second"] * .9,
                        estimate_provenance=base["experiment_id"])
        run_screen(CONTROL, CONFIG, base["threads"], base["tokens_per_second"] * .9)
        mixture = extra.pop("mixture_weights", None)
        register_screen(experiment_id=CANDIDATE, config_path=candidate_config,
                        variant=variant, budget=BUDGET, parent_control=CONTROL,
                        hypothesis=f"Replicate {parent} against a matched control under independent seed 2026",
                        throughput_tps=candidate["tokens_per_second"] * .9,
                        estimate_provenance=candidate["experiment_id"],
                        mixture_weights=mixture, extra_fields=extra)
        run_screen(CANDIDATE, candidate_config, candidate["threads"],
                   candidate["tokens_per_second"] * .9)
    return decision, parent


def main() -> None:
    if not (RESEARCH / "promotion_selection.json").exists():
        raise RuntimeError("promotion selection prerequisite missing")
    state = json.loads(STATE.read_text())
    matched = matched_replication(state)
    mechanism_decision, parent = mechanism_replication(state)
    lines = ["# Second-seed replication decisions", "",
             f"The 16M Co4/dense pair was {'run' if matched['run'] else 'gated out'}: {matched['reason']}.",
             f"The selected mechanism pair was {'run' if mechanism_decision['run'] else 'gated out'}: {mechanism_decision['reason']}.",
             "Each launched pair uses seed 2026, 150M logical tokens per arm, a fresh WSD trajectory, the same DATA-D corpus and order, and the common evaluator. The controller budgets both endpoints and protects later named stretch work before launch.", ""]
    if matched["run"]:
        co4, dense = endpoint(CONTROL), endpoint(MATCHED_DENSE)
        lines += [f"At 150M, second-seed Co4 minus dense held-out loss was {co4[0]-dense[0]:+.5f}, BPB {co4[1]-dense[1]:+.5f}, and four-task raw accuracy {co4[2]-dense[2]:+.4f}.", ""]
    if mechanism_decision["run"]:
        base, new = endpoint(CONTROL), endpoint(CANDIDATE)
        lines += [f"The selected 16M mechanism `{parent}` changed second-seed held-out loss by {new[0]-base[0]:+.5f}, BPB by {new[1]-base[1]:+.5f}, and raw accuracy by {new[2]-base[2]:+.4f}.", ""]
    report = RESEARCH / "SECOND_SEED_REPLICATION.md"
    report.write_text("\n".join(lines))
    upload([report, RESEARCH / "matched_16m_second_seed_decision.json",
            RESEARCH / "mechanism_second_seed_decision.json"], "SECOND-SEED-REPORT")


if __name__ == "__main__":
    main()
