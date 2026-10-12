"""Evidence-gated 32M promotion and conditional fresh 49M trajectory."""
from __future__ import annotations

import json
import sys
import time

import torch

from latticelm.config import LatticeConfig
from latticelm.posthack_variants import build_posthack_model
from posthack_archive import upload
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, event, sha256, utc
from posthack_phase1 import invoke
from posthack_phase5 import endpoint
from posthack_single_run import register_screen, run_screen
from train_final_production import load_arrays
from posthack_train import evaluate
from posthack_uncertainty import paired_capability

CONFIG32 = RESEARCH / "configs/32m_co4.json"
CONFIG32_PHM = RESEARCH / "configs/32m_phm4.json"
CONFIG49 = RESEARCH / "configs/49m_co4.json"
CONFIG49_PHM = RESEARCH / "configs/49m_phm4.json"
INDEX = RESEARCH / "research_d/documents.jsonl"
CAP_INDEX = RESEARCH / "research_d/capability_documents.jsonl"
CONTROL16 = "A16-CO4-S1337"
CONTROL32 = "R32-CO4-CONTROL"
DATA_ARMS = ("D1-16M-REBALANCED-S1337", "D2-16M-CAPABILITY",
             "D3-16M-CURRICULUM", "D4-16M-ANNEAL", "D5-16M-COMBINED")
ARCH_ARMS = ("B16-VALUE-RESIDUAL-S1337", "E2-16M-GROWTH-CURRICULUM",
             "G-16M-PHM4-FFN")


def eligible(name: str, control: tuple) -> tuple[bool, dict]:
    row = endpoint(name)
    heldout_change = row[0] / control[0] - 1
    bpb_change = row[1] / control[1] - 1
    capability_change = row[2] - control[2]
    source_regression = max(row[3][source] / control[3][source] - 1 for source in control[3])
    paired = paired_capability(RESEARCH / "runs" / name / "capability_samples.json",
                               RESEARCH / "runs" / CONTROL16 / "capability_samples.json")
    supported_capability_gain = paired["four_task_equal_weight_ci95"][0] > 0
    quality = heldout_change <= -.01 and bpb_change <= -.005 and capability_change >= -.01
    capability = (capability_change >= .01 and supported_capability_gain and
                  heldout_change <= .02 and source_regression <= .05)
    wall_ratio = (json.loads((RESEARCH / "runs" / name / "result.json").read_text())["wall_seconds"] /
                  json.loads((RESEARCH / "runs" / CONTROL16 / "result.json").read_text())["wall_seconds"])
    if name == "G-16M-PHM4-FFN" and wall_ratio > 1.1:
        quality = capability = False
    return quality or capability, {"heldout_relative": heldout_change,
                                   "bpb_relative": bpb_change,
                                   "capability_absolute": capability_change,
                                   "worst_source_relative": source_regression,
                                   "paired_capability": paired,
                                   "supported_capability_gain": supported_capability_gain,
                                   "wall_ratio_to_control": wall_ratio,
                                   "selection_basis": "quality" if quality else "capability" if capability else "none"}


def select_winners() -> dict:
    control = endpoint(CONTROL16)
    evidence = {}
    for name in (*DATA_ARMS, *ARCH_ARMS):
        if (RESEARCH / "runs" / name / "hf_archive_receipt.json").exists():
            evidence[name] = dict(zip(("eligible", "measures"), eligible(name, control)))
    ranking = lambda name: (evidence[name]["measures"]["bpb_relative"] +
                            evidence[name]["measures"]["heldout_relative"],
                            -evidence[name]["measures"]["capability_absolute"])
    data = [name for name in DATA_ARMS if evidence.get(name, {}).get("eligible")]
    architecture = [name for name in ARCH_ARMS if evidence.get(name, {}).get("eligible")]
    data_winner = min(data, key=ranking) if data else None
    arch_winner = min(architecture, key=ranking) if architecture else None
    if arch_winner == "E2-16M-GROWTH-CURRICULUM" and data_winner in ("D3-16M-CURRICULUM", "D5-16M-COMBINED"):
        # E2 contains a curriculum; an independent second winner must differ.
        alternative = [name for name in data if name not in ("D3-16M-CURRICULUM", "D5-16M-COMBINED")]
        data_winner = min(alternative, key=ranking) if alternative else None
    result = {"created_utc": utc(), "control": CONTROL16,
              "independent_winners": {"architecture": arch_winner, "data": data_winner},
              "evidence": evidence,
              "threshold": "at least 1% held-out and 0.5% BPB improvement with <=1 point capability regression; or >=1 point broad capability gain with paired item-bootstrap interval above zero, <=2% held-out and <=5% worst-source regression",
              "phm_policy": "PHM-4 may promote only with the registered LM/BPB/capability signal and <=10% wall-time overhead; 32M/49M PHM configs are parameter matched within 0.01%",
              "tokenizer_policy": "8K 16M subset result is reported separately; full DATA-D-v4 8K retokenization is outside the incremental data budget"}
    atomic_json(RESEARCH / "promotion_selection.json", result)
    return result


def calibration(regime: str, variant: str = "baseline") -> dict:
    state = json.loads(STATE.read_text())
    arch = "co4" if variant == "baseline" else variant
    if not all(f"{regime}/{arch}/{threads}" in state["throughput_calibration"] for threads in (12, 16)):
        command = [sys.executable, "scripts/posthack_calibrate.py", "--regimes", regime,
                   "--architectures", "phm4" if variant == "phm4_ffn" else "co4",
                   "--threads", "12", "16"]
        if variant != "baseline":
            command += ["--variant", variant]
        invoke(command, RESEARCH / f"{regime}_{arch}_calibration.log", f"{regime} {arch} calibration")
    state = json.loads(STATE.read_text())
    return max((state["throughput_calibration"][f"{regime}/{arch}/{threads}"] for threads in (12, 16)),
               key=lambda row: row["tokens_per_second"])


def mechanism(name: str | None, budget: int, final_depth: int = 10) -> tuple[str, dict]:
    if name is None:
        return "baseline", {}
    if name.startswith("B16-"):
        return "value_residual", {}
    if name.startswith("E2-"):
        return "progressive_depth", {
            "sampling_mode": "curriculum",
            "sampling_rule": {"curriculum_fraction": .25, "curriculum_strength": .8},
            "research_d_index_path": str(INDEX.relative_to(ROOT)),
            "research_d_index_sha256": sha256(INDEX),
            "growth_schedule": [{"at_tokens": budget // 8, "depth": round(final_depth * .75)},
                                {"at_tokens": budget // 4, "depth": final_depth}]}
    if name.startswith("G-"):
        return "phm4_ffn", {"phm_dimension": 4, "phm_scope": "FFN projections only"}
    if name.startswith("D1-"):
        return "baseline", {"mixture_weights": {"fineweb_edu": .50,
                                                  "fineweb": .30, "wikipedia": .10,
                                                  "finemath": .10}}
    if name.startswith("D2-"):
        return "baseline", {"sampling_mode": "capability",
                            "research_d_index_path": str(CAP_INDEX.relative_to(ROOT)),
                            "research_d_index_sha256": sha256(CAP_INDEX),
                            "sampling_rule": {"capability_floor": .90, "capability_fraction": .125}}
    if name.startswith(("D3-", "D4-", "D5-")):
        mode = ("curriculum" if name.startswith("D3-") else
                "quality_anneal" if name.startswith("D4-") else "curriculum_anneal")
        extra = {"sampling_mode": mode,
                 "sampling_rule": {"curriculum_fraction": .25,
                                   "curriculum_strength": .8, "quality_floor": .70},
                 "research_d_index_path": str(INDEX.relative_to(ROOT)),
                 "research_d_index_sha256": sha256(INDEX)}
        if mode in ("quality_anneal", "curriculum_anneal"):
            extra["anneal_rule"] = {"at_fraction": .85, "quality_floor": .70,
                                    "mixture_weights": {"fineweb_edu": .60,
                                                        "finemath": .20,
                                                        "fineweb": .10,
                                                        "wikipedia": .10}}
        return "baseline", extra
    raise ValueError(name)


def launch(name: str, config, parent: str, variant: str, extra: dict, budget: int,
           calibration_row: dict) -> None:
    mix = extra.pop("mixture_weights", None)
    register_screen(experiment_id=name, config_path=config, variant=variant,
                    budget=budget, parent_control=parent,
                    hypothesis=f"Promote {parent} to the larger parameter regime at matched token and evaluation budgets",
                    throughput_tps=calibration_row["tokens_per_second"] * .9,
                    estimate_provenance=calibration_row["experiment_id"],
                    mixture_weights=mix, extra_fields=extra)
    run_screen(name, config, calibration_row["threads"],
               calibration_row["tokens_per_second"] * .9)


def compare32(selection: dict) -> dict:
    control = endpoint(CONTROL32)
    rows = {CONTROL32: {"heldout_loss": control[0], "wiki_bpb": control[1],
                        "four_task_accuracy": control[2]}}
    for kind, parent in selection["independent_winners"].items():
        if parent is None:
            continue
        name = f"R32-{kind.upper()}-PROMOTION"
        heldout, bpb, accuracy, _ = endpoint(name)
        rows[name] = {"parent_16m": parent, "heldout_loss": heldout,
                      "wiki_bpb": bpb, "four_task_accuracy": accuracy,
                      "heldout_relative_to_control": heldout / control[0] - 1,
                      "bpb_relative_to_control": bpb / control[1] - 1,
                      "accuracy_absolute_to_control": accuracy - control[2]}
    result = {"created_utc": utc(), "rows": rows,
              "selection": selection["independent_winners"]}
    atomic_json(RESEARCH / "promotion_32m_results.json", result)
    lines = ["# 32M promotion tournament", "",
             "The 32M control and up to two independently selected mechanisms use fresh 300M-token trajectories, the same frozen 4K tokenizer, 32×256 scientific batch, modern Muon/AdamW recipe, and common evaluator. Mechanisms were selected from 16M evidence before 32M training. Raw-token, 6NT active-parameter FLOP proxy, and wall-clock views are separate.", "",
             "| Arm | 16M parent mechanism | Held-out loss | WikiText BPB | Four-task raw accuracy |",
             "|---|---|---:|---:|---:|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {row.get('parent_16m', 'control')} | {row['heldout_loss']:.5f} | {row['wiki_bpb']:.5f} | {row['four_task_accuracy']:.4f} |")
    lines += ["", "The full registry, token curves, source-wise losses, raw capability samples, checkpoint hashes, and measured wall times are archived with each arm. Small capability movements require paired uncertainty analysis before any promotion claim.", ""]
    (RESEARCH / "PROMOTION_32M.md").write_text("\n".join(lines))
    return result


def maybe_49m(promotion: dict) -> None:
    state = json.loads(STATE.read_text())
    remaining_hours = (state["archive_start_epoch"] - time.time()) / 3600
    control = promotion["rows"][CONTROL32]
    candidates = [(name, row) for name, row in promotion["rows"].items() if name != CONTROL32
                  and (row["heldout_relative_to_control"] <= -.01 or
                       row["bpb_relative_to_control"] <= -.01)
                  and row["accuracy_absolute_to_control"] >= -.01]
    if not candidates or remaining_hours < 30:
        reason = "no supported 32M improvement" if not candidates else "less than 30 hours before archival reserve"
        (RESEARCH / "NEXT_49M_TRAJECTORY.md").write_text(
            f"# Near-49M research trajectory\n\nNo fresh 49M run was launched: {reason}. "
            "The frozen 48.6M hackathon checkpoint remains immutable.\n")
        event("R49-RESEARCH", "RUN_REJECTED", reason=reason, remaining_hours=remaining_hours)
        return
    selected_name, _ = min(candidates, key=lambda item: item[1]["wiki_bpb"])
    parent16 = promotion["rows"][selected_name]["parent_16m"]
    variant, _ = mechanism(parent16, 300_000_000, final_depth=12)
    measured = calibration("49m", variant)
    decision_path = RESEARCH / "49m_budget_decision.json"
    if decision_path.exists():
        budget_decision = json.loads(decision_path.read_text())
    else:
        # Preserve time for a paired second seed and the named stretch pilots.
        available_seconds = state["archive_start_epoch"] - time.time() - 36 * 3600
        measured_tps = measured["tokens_per_second"] * .9
        feasible = [budget for budget in (300_000_000, 400_000_000, 500_000_000)
                    if budget / measured_tps + 7200 + 1800 + 3600 < available_seconds]
        budget = max(feasible) if feasible else None
        budget_decision = {"created_utc": utc(), "token_budget": budget,
                           "measured_tps": measured["tokens_per_second"],
                           "selected_32m_parent": selected_name,
                           "criterion": "largest of 300M/400M/500M fitting archival reserve while leaving 36 hours for second-seed and named stretch screens",
                           "reason": "measured throughput supports expanded trajectory" if budget and budget > 300_000_000 else
                                     "300M trajectory fits" if budget else "300M trajectory cannot fit with protected downstream time"}
        atomic_json(decision_path, budget_decision)
        event("R49-RESEARCH", "OPTIONAL_BUDGET_DECISION", **budget_decision)
    budget = budget_decision["token_budget"]
    if budget is None:
        (RESEARCH / "NEXT_49M_TRAJECTORY.md").write_text(
            "# Near-49M research trajectory\n\nNo fresh 49M run was launched: "
            + budget_decision["reason"] + ".\n")
        return
    variant, extra = mechanism(parent16, budget, final_depth=12)
    extra["extra_checkpoint_tokens"] = {"10m": 10_000_000}
    research_config = CONFIG49_PHM if variant == "phm4_ffn" else CONFIG49
    launch("R49-RESEARCH", research_config, selected_name, variant, extra, budget, measured)
    milestone = RESEARCH / "runs/R49-RESEARCH/milestones/10m.pt"
    checkpoint = torch.load(milestone, map_location="cpu", weights_only=False)
    cfg = LatticeConfig(**checkpoint["config"])
    model = build_posthack_model(cfg, checkpoint.get("variant", "baseline"))
    model.load_state_dict(checkpoint["model"])
    torch.set_num_threads(4)
    _, _, valid = load_arrays(ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json")
    losses, _ = evaluate(model, valid, cfg.context_length)
    historical = torch.load(ROOT / "artifacts/final_production_run/early-health.pt",
                            map_location="cpu", weights_only=False)
    if historical["tokens"] != 10_002_432:
        raise RuntimeError("frozen historical 10M checkpoint identity changed")
    result = {"selected_32m_parent": selected_name, "mechanism": parent16,
              "new_10m_source_losses": losses,
              "new_10m_aggregate_loss": sum(losses.values()) / len(losses),
              "frozen_10m_validation_loss": historical["validation_loss"],
              "frozen_10m_checkpoint_sha256": sha256(ROOT / "artifacts/final_production_run/early-health.pt"),
              "new_10m_checkpoint_sha256": sha256(milestone),
              "comparison_limit": "historical production validation aggregation must be matched before claiming a numerical win; 300M is not evidence for 2B outcome"}
    atomic_json(RESEARCH / "next_49m_trajectory.json", result)
    (RESEARCH / "NEXT_49M_TRAJECTORY.md").write_text(
        "# Near-49M research trajectory\n\n"
        f"A fresh 48.6M research model promoted `{parent16}` from the 32M tournament, "
        f"targeting {budget/1e6:.0f}M tokens. The original 1.9B-token hackathon checkpoint was not modified.\n\n"
        f"At roughly 10M tokens, the new source-average held-out loss was {result['new_10m_aggregate_loss']:.5f}; "
        f"the frozen production early-health checkpoint records {result['frozen_10m_validation_loss']:.5f}. "
        "Their validation aggregation may differ, so this is a trajectory anchor, not a numerical architecture claim. "
        f"The new {budget/1e6:.0f}M endpoint remains far short of the frozen 1.9B endpoint.\n")
    upload([RESEARCH / "NEXT_49M_TRAJECTORY.md",
            RESEARCH / "next_49m_trajectory.json", decision_path], "49M-TRAJECTORY-REPORT")


def main() -> None:
    if not (RESEARCH / "POSTTRAINING_TRANSFER.md").exists():
        raise RuntimeError("posttraining report prerequisite missing")
    selection_path = RESEARCH / "promotion_selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else select_winners()
    base = calibration("32m")
    launch(CONTROL32, CONFIG32, "A32-CO4-S1337", "baseline", {}, 300_000_000, base)
    for kind, parent in selection["independent_winners"].items():
        if parent is None:
            continue
        variant, extra = mechanism(parent, 300_000_000)
        row = calibration("32m", variant)
        candidate_config = CONFIG32_PHM if variant == "phm4_ffn" else CONFIG32
        launch(f"R32-{kind.upper()}-PROMOTION", candidate_config, parent,
               variant, extra, 300_000_000, row)
    promotion = compare32(selection)
    upload([selection_path, RESEARCH / "promotion_32m_results.json",
            RESEARCH / "PROMOTION_32M.md"], "32M-PROMOTION-REPORT")
    maybe_49m(promotion)


if __name__ == "__main__":
    main()
