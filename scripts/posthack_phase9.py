"""Evidence- and time-gated overflow after mandatory and named stretch work."""
from __future__ import annotations

import json
import sys

from posthack_archive import upload
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, can_launch, event, sha256, utc
from posthack_phase1 import invoke
from posthack_phase5 import endpoint
from posthack_phase7 import baseline_50m
from posthack_single_run import TASKS, register_screen, remaining_tokens, run_screen

CONFIG8 = RESEARCH / "configs/8m_co4.json"
CONFIG16 = RESEARCH / "configs/16m_co4.json"
ENGRAM_CONFIG = RESEARCH / "configs/8m_engram.json"
ENGRAM_MAP = RESEARCH / "configs/engram_4k_identity_map.json"
INDEX = RESEARCH / "research_d/documents.jsonl"
LATER_MIXTURE = {"fineweb_edu": .40, "fineweb": .20, "wikipedia": .20, "finemath": .20}


def frozen_decision(name: str, *, evidence: bool, reason: str,
                    budget: int, tps: float) -> dict:
    path = RESEARCH / f"overflow_{name}_decision.json"
    if path.exists():
        return json.loads(path.read_text())
    fits, finish = can_launch(budget, tps, 7200, 1800, 3600)
    result = {"created_utc": utc(), "run": bool(evidence and fits),
              "evidence_passed": evidence, "estimated_tps": tps,
              "predicted_finish_utc": utc(finish),
              "reason": "evidence and time gate passed" if evidence and fits else
                        reason if not evidence else "predicted endpoint crosses archival reserve"}
    atomic_json(path, result)
    event(name, "OPTIONAL_RUN_DECISION", **result)
    return result


def measured_gate(name: str, budget: int, tps: float, decision: dict) -> bool:
    if (RESEARCH / "runs" / name / "hf_archive_receipt.json").exists():
        return True
    fits, finish = can_launch(remaining_tokens(name, budget), tps, 7200, 1800, 3600)
    if fits:
        return True
    decision["run"] = False
    decision["reason"] = "measured throughput cannot finish before archival reserve"
    decision["measured_finish_utc"] = utc(finish)
    atomic_json(RESEARCH / f"overflow_{name}_decision.json", decision)
    event(name, "OPTIONAL_RUN_REJECTED_AFTER_CALIBRATION", **decision)
    return False


def adaptive_evidence() -> tuple[bool, str]:
    analysis = RESEARCH / "developmental_experts_analysis.json"
    if not analysis.exists():
        return False, "four-arm developmental-expert evidence is missing"
    rows = {row["name"]: row for row in json.loads(analysis.read_text())}
    fixed = rows["F0-8M-FIXED-EXPERTS"]
    naive = rows["F1-8M-NAIVE-BIRTH"]
    catchup = rows["F3-8M-CLONE-CATCHUP"]
    quality = (catchup["wiki_bpb"] <= min(fixed["wiki_bpb"], naive["wiki_bpb"]) * .99 or
               catchup["heldout_loss"] <= min(fixed["heldout_loss"], naive["heldout_loss"]) * .99)
    specialization = catchup["domain_expert_mi_nats"] >= naive["domain_expert_mi_nats"] + .005
    raw = endpoint("F3-8M-CLONE-CATCHUP")[2] - endpoint("F0-8M-FIXED-EXPERTS")[2]
    eligible = quality and specialization and raw >= -.01
    return eligible, ("F3 cleared quality, specialization, and broad-capability gates" if eligible else
                      "F3 did not clear the pre-registered quality, MI, and capability gates")


def maybe_adaptive_birth() -> dict:
    name = "F4-8M-ADAPTIVE-BIRTH"
    evidence, reason = adaptive_evidence()
    state = json.loads(STATE.read_text())
    rows = [state["throughput_calibration"][f"8m/moe_growth/{threads}"] for threads in (12, 16)]
    best = max(rows, key=lambda row: row["tokens_per_second"])
    decision = frozen_decision(name, evidence=evidence, reason=reason,
                               budget=100_000_000, tps=best["tokens_per_second"] * .8)
    if decision["run"]:
        rule = {"transition_at_fraction": .25, "later_mixture": LATER_MIXTURE,
                "birth_strategy": "adaptive_clone", "specialty_catchup_tokens": 5_000_000,
                "adaptive_birth": {"window_steps": 64, "consecutive_windows": 3,
                                   "min_max_load": .31, "min_domain_expert_mi": .01,
                                   "min_finemath_loss_gap": .03,
                                   "min_finemath_examples": 64,
                                   "latest_birth_fraction": .60},
                "parent_rule": "most FineMath assignments among initial four; lower index breaks ties",
                "optimizer_policy": "old state retained; cloned child weights inherit no moments"}
        register_screen(experiment_id=name, config_path=CONFIG8, variant="moe_growth",
                        budget=100_000_000, parent_control="F3-8M-CLONE-CATCHUP",
                        hypothesis="Sustained router utilization, domain specialization, and FineMath loss gap can time expert birth better than a fixed curriculum boundary",
                        throughput_tps=decision["estimated_tps"],
                        estimate_provenance=best["experiment_id"],
                        extra_fields={"sampling_mode": "curriculum",
                                      "sampling_rule": {"curriculum_fraction": .25,
                                                        "curriculum_strength": .8},
                                      "research_d_index_path": str(INDEX.relative_to(ROOT)),
                                      "research_d_index_sha256": sha256(INDEX),
                                      "moe_rule": rule})
        run_screen(name, CONFIG8, best["threads"], decision["estimated_tps"])
    return decision


def maybe_value_skip() -> dict:
    name = "B16-VALUE-RESIDUAL-SKIP"
    state = json.loads(STATE.read_text())
    positive = any(row.get("experiment_id") == "B16-VALUE-RESIDUAL-S1337"
                   and row.get("classification") == "PROVISIONAL_POSITIVE"
                   for row in state["decisions"])
    base = state["throughput_calibration"]["16m/value_residual/16"]
    decision = frozen_decision(name, evidence=positive,
                               reason="first value-residual screen was not provisionally positive",
                               budget=150_000_000, tps=base["tokens_per_second"] * .85)
    if decision["run"]:
        key = "16m/value_residual_skip"
        if not all(f"{key}/{threads}" in json.loads(STATE.read_text())["throughput_calibration"]
                   for threads in (12, 16)):
            invoke([sys.executable, "scripts/posthack_calibrate.py", "--regimes", "16m",
                    "--architectures", "co4", "--threads", "12", "16",
                    "--variant", "value_residual_skip"],
                   RESEARCH / "value_skip_calibration.log", "value-residual skip calibration")
        state = json.loads(STATE.read_text())
        best = max((state["throughput_calibration"][f"{key}/{threads}"] for threads in (12, 16)),
                   key=lambda row: row["tokens_per_second"])
        if not measured_gate(name, 150_000_000, best["tokens_per_second"], decision):
            return decision
        register_screen(experiment_id=name, config_path=CONFIG16,
                        variant="value_residual_skip", budget=150_000_000,
                        parent_control="B16-VALUE-RESIDUAL-S1337",
                        hypothesis="One zero-gated embedding-to-final-state scalar skip adds value beyond post-MOD value residual alone",
                        throughput_tps=best["tokens_per_second"],
                        estimate_provenance=best["experiment_id"])
        run_screen(name, CONFIG16, best["threads"], best["tokens_per_second"])
    return decision


def maybe_engram() -> dict:
    name = "O8-ENGRAM-REVISIT"
    state = json.loads(STATE.read_text())
    base = state["throughput_calibration"]["8m/co4/16"]
    decision = frozen_decision(name, evidence=True, reason="",
                               budget=50_000_000, tps=base["tokens_per_second"] * .55)
    if decision["run"]:
        upload([ENGRAM_MAP], "ENGRAM-TOKEN-MAP")
        key = "8m/engram"
        if not all(f"{key}/{threads}" in json.loads(STATE.read_text())["throughput_calibration"]
                   for threads in (12, 16)):
            invoke([sys.executable, "scripts/posthack_calibrate.py", "--regimes", "8m",
                    "--architectures", "engram", "--threads", "12", "16"],
                   RESEARCH / "engram_calibration.log", "tiny Engram calibration")
        state = json.loads(STATE.read_text())
        best = max((state["throughput_calibration"][f"{key}/{threads}"] for threads in (12, 16)),
                   key=lambda row: row["tokens_per_second"])
        if not measured_gate(name, 50_000_000, best["tokens_per_second"], decision):
            return decision
        register_screen(experiment_id=name, config_path=ENGRAM_CONFIG,
                        variant="baseline", budget=50_000_000,
                        parent_control="A8-CO4-S1337/50M milestone",
                        hypothesis="A small causal n-gram memory improves Co4 at matched trainable parameters and tokens",
                        throughput_tps=best["tokens_per_second"],
                        estimate_provenance=best["experiment_id"],
                        extra_fields={"fairness_axes": ["equal_parameters", "equal_tokens", "wall_clock"]})
        run_screen(name, ENGRAM_CONFIG, best["threads"], best["tokens_per_second"])
    return decision


def report(decisions: dict) -> None:
    lines = ["# Evidence-gated overflow", "",
             "This phase runs only after the mandatory tracks, paired second seed, and both named stretch pilots have completed or been gated out. Every option has a frozen decision record before training; the archival reserve always wins.", ""]
    for name, decision in decisions.items():
        lines.append(f"- {name}: {'launched' if decision['run'] else 'gated out'} — {decision['reason']}.")
    name = "F4-8M-ADAPTIVE-BIRTH"
    if (RESEARCH / "runs" / name / "hf_archive_receipt.json").exists():
        result = json.loads((RESEARCH / "runs" / name / "result.json").read_text())
        parent = endpoint("F3-8M-CLONE-CATCHUP")
        candidate = endpoint(name)
        lines.extend(["", f"Adaptive expert birth occurred at {result['moe_birth_tokens']} logical tokens (null means the registered trigger never fired).",
                      f"F4 minus fixed-birth F3: held-out loss {candidate[0]-parent[0]:+.5f}, WikiText BPB {candidate[1]-parent[1]:+.5f}, four-task raw accuracy {candidate[2]-parent[2]:+.4f}. Routing windows and exact trigger values are in the event log."])
    name = "B16-VALUE-RESIDUAL-SKIP"
    if (RESEARCH / "runs" / name / "hf_archive_receipt.json").exists():
        parent, candidate = endpoint("B16-VALUE-RESIDUAL-S1337"), endpoint(name)
        lines.extend(["", f"The single scalar skip changed BPB by {candidate[1]-parent[1]:+.5f} and four-task raw accuracy by {candidate[2]-parent[2]:+.4f} versus value residual alone."])
    name = "O8-ENGRAM-REVISIT"
    if (RESEARCH / "runs" / name / "hf_archive_receipt.json").exists():
        base_loss, base_bpb = baseline_50m()
        candidate = endpoint(name)
        lines.extend(["", f"The tiny Engram screen versus the 8M Co4 50M milestone changed held-out loss by {candidate[0]-base_loss:+.5f} and WikiText BPB by {candidate[1]-base_bpb:+.5f}. This is a bounded revisit, not a full memory architecture campaign."])
    path = RESEARCH / "OVERFLOW_OPTIONS.md"
    path.write_text("\n".join(lines) + "\n")
    upload([path, *(RESEARCH / f"overflow_{name}_decision.json" for name in decisions)],
           "OVERFLOW-OPTIONS-REPORT")


def main() -> None:
    if not (RESEARCH / "STRETCH_NCO4.md").exists() or not (RESEARCH / "STRETCH_DIFFCO4.md").exists():
        raise RuntimeError("named stretch goals must complete or be gated out before overflow")
    decisions = {"F4-8M-ADAPTIVE-BIRTH": maybe_adaptive_birth(),
                 "B16-VALUE-RESIDUAL-SKIP": maybe_value_skip(),
                 "O8-ENGRAM-REVISIT": maybe_engram()}
    report(decisions)


if __name__ == "__main__":
    main()
