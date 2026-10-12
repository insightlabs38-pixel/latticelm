"""Evidence-gated Phase 2: value residual, tokenizer audit, and D1 mixture."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from posthack_analyze_phase1 import PAIRS, analyze
from posthack_archive import compress_index, upload
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, can_launch, event, sha256, utc
from posthack_phase1 import invoke
from posthack_single_run import register_screen, remaining_tokens, run_screen

CONFIG = RESEARCH / "configs/16m_co4.json"
CONTEXT_CONFIG = RESEARCH / "configs/16m_context512.json"


def maybe_extend_8m() -> None:
    analysis = json.loads((RESEARCH / "matched_co4_dense_analysis.json").read_text())
    decision_path = RESEARCH / "8m_extension_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
    else:
        comparisons = [row for row in analysis["comparisons"] if row["pair"].startswith("8M")]
        signs_disagree = comparisons[0]["heldout_loss_delta_co4_minus_dense"] * comparisons[1]["heldout_loss_delta_co4_minus_dense"] <= 0
        intervals = [row["paired_window_bootstrap"].get("ci95") for row in comparisons]
        both_uncertain = len(intervals) == 2 and all(
            interval is not None and interval[0] <= 0 <= interval[1]
            for interval in intervals)
        mean_delta = sum(row["heldout_loss_delta_co4_minus_dense"] for row in comparisons) / 2
        mean_control_loss = sum(analysis["runs"][row["dense"]]["weighted_heldout_loss"]
                                for row in comparisons) / 2
        ambiguous = signs_disagree or both_uncertain or abs(mean_delta / mean_control_loss) < .005
        state = json.loads(STATE.read_text())
        speeds = {kind: max(state["throughput_calibration"][f"8m/{kind}/{threads}"]["tokens_per_second"]
                            for threads in (12, 16)) for kind in ("co4", "dense")}
        expected = sum(200_000_000 / speeds[kind] + 7200 + 1800 + 3600
                       for kind in ("co4", "dense"))
        remaining = state["archive_start_epoch"] - time.time()
        protected = 180 * 3600
        eligible = ambiguous and min(speeds.values()) >= 9000 and expected + protected < remaining
        decision = {"created_utc": utc(), "run": eligible,
                    "ambiguous_8m_effect": ambiguous,
                    "mean_relative_heldout_delta": mean_delta / mean_control_loss,
                    "measured_tps": speeds, "predicted_pair_seconds": expected,
                    "protected_mandatory_seconds": protected,
                    "criterion": "two-seed sign disagreement, both paired intervals include zero, or <0.5% mean loss separation; both TPS >=9000; fresh paired 200M runs leave 180h for mandatory work",
                    "reason": "ambiguity, speed, and time gates passed" if eligible else
                              "8M effect not ambiguous" if not ambiguous else
                              "throughput or protected mandatory time gate failed"}
        atomic_json(decision_path, decision)
        event("A8-EXTENSION", "OPTIONAL_RUN_DECISION", **decision)
    if decision["run"]:
        state = json.loads(STATE.read_text())
        for architecture in ("co4", "dense"):
            best = max((state["throughput_calibration"][f"8m/{architecture}/{threads}"]
                        for threads in (12, 16)), key=lambda row: row["tokens_per_second"])
            name = f"A8-{architecture.upper()}-EXT200"
            config = RESEARCH / "configs" / f"8m_{architecture}.json"
            register_screen(experiment_id=name, config_path=config, variant="baseline",
                            budget=200_000_000, parent_control=f"A8-{architecture.upper()}-S1337",
                            hypothesis="A fresh longer matched WSD trajectory resolves an ambiguous two-seed 8M Co4/dense effect",
                            throughput_tps=best["tokens_per_second"],
                            estimate_provenance=best["experiment_id"],
                            extra_fields={"fairness_axes": ["equal_parameters", "equal_tokens", "equal_data_order", "wall_clock"],
                                          "extension_policy": "fresh 200M paired WSD runs; earlier 100M WSD checkpoints are not relabeled as 200M continuations"})
            run_screen(name, config, best["threads"], best["tokens_per_second"])
    lines = ["# Optional 8M matched extension", "",
             f"Decision: {'run a fresh 200M Co4/dense pair' if decision['run'] else 'do not extend'} — {decision['reason']}.",
             "The two arms use the same seed, data order, final recipe and 200M-token WSD horizon. They are new registered trajectories; the completed 100M endpoints retain their own schedule and identity.", ""]
    if (RESEARCH / "runs/A8-CO4-EXT200/hf_archive_receipt.json").exists() and (RESEARCH / "runs/A8-DENSE-EXT200/hf_archive_receipt.json").exists():
        co4, dense = endpoint("A8-CO4-EXT200"), endpoint("A8-DENSE-EXT200")
        lines.append(f"At 200M, Co4 minus dense WikiText BPB was {co4[1]['bits_per_byte']-dense[1]['bits_per_byte']:+.5f}; source-wise losses and capability samples are in the run archives.")
    report = RESEARCH / "8M_EXTENSION.md"
    report.write_text("\n".join(lines) + "\n")
    upload([decision_path, report], "8M-EXTENSION-DECISION")


def context_gate() -> None:
    audit_path = RESEARCH / "prompt_truncation_audit.json"
    if not audit_path.exists():
        invoke([sys.executable, "scripts/posthack_prompt_audit.py"],
               RESEARCH / "prompt_truncation_audit.log", "full eight-task prompt audit")
    audit = json.loads(audit_path.read_text())
    decision_path = RESEARCH / "context512_decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
    else:
        state = json.loads(STATE.read_text())
        tps = state["throughput_calibration"]["16m/co4/16"]["tokens_per_second"] * .55
        fits, finish = can_launch(150_000_000, tps, 7200, 1800, 3600)
        decision = {"created_utc": utc(), "run": bool(audit["train_context512"] and fits),
                    "max_task_truncation_fraction": audit["max_task_meaningful_truncation_fraction_at_256"],
                    "predicted_finish_utc": utc(finish),
                    "criterion": "full eight-task 4K audit >=1.5% meaningful truncation in any task, and 150M-token matched-context run fits reserve",
                    "reason": "audit and time gate passed" if audit["train_context512"] and fits else
                              "prompt audit below context trigger" if not audit["train_context512"] else
                              "context-512 run would cross archival reserve"}
        atomic_json(decision_path, decision)
        event("C16-CONTEXT512", "OPTIONAL_RUN_DECISION", **decision)
    if decision["run"]:
        state = json.loads(STATE.read_text())
        if not all(f"16m/context512/{threads}" in state["throughput_calibration"] for threads in (12, 16)):
            invoke([sys.executable, "scripts/posthack_calibrate.py", "--regimes", "16m",
                    "--architectures", "context512", "--threads", "12", "16"],
                   RESEARCH / "context512_calibration.log", "context-512 calibration")
        state = json.loads(STATE.read_text())
        best = max((state["throughput_calibration"][f"16m/context512/{threads}"] for threads in (12, 16)),
                   key=lambda row: row["tokens_per_second"])
        fits, finish = can_launch(remaining_tokens("C16-CONTEXT512", 150_000_000),
                                  best["tokens_per_second"], 7200, 1800, 3600)
        if not fits:
            decision["run"] = False
            decision["reason"] = "measured context-512 throughput crosses archival reserve"
            decision["measured_finish_utc"] = utc(finish)
            atomic_json(decision_path, decision)
        else:
            register_screen(experiment_id="C16-CONTEXT512", config_path=CONTEXT_CONFIG,
                            variant="baseline", budget=150_000_000,
                            parent_control="A16-CO4-S1337",
                            hypothesis="Context 512 improves materially truncated held-out capability at matched 16M parameters and 8192-token effective batches",
                            throughput_tps=best["tokens_per_second"],
                            estimate_provenance=best["experiment_id"],
                            extra_fields={"prompt_audit_sha256": sha256(audit_path),
                                          "fairness_axes": ["equal_parameters", "equal_tokens", "equal_effective_batch_tokens", "wall_clock"]})
            run_screen("C16-CONTEXT512", CONTEXT_CONFIG, best["threads"], best["tokens_per_second"])
    lines = ["# Context-length gate", "",
             f"The full eight-task 4K prompt audit found a maximum meaningful truncation fraction of {audit['max_task_meaningful_truncation_fraction_at_256']:.2%} at context 256.",
             f"Decision: {'run 512-token context' if decision['run'] else 'do not run 512-token context'} — {decision['reason']}.",
             "The 512 configuration uses batch 16 × context 512, preserving the 8192-token effective optimization batch of the 256 control (batch 32 × context 256). Tokenized document chunks differ by context length, as intended.", ""]
    if (RESEARCH / "runs/C16-CONTEXT512/hf_archive_receipt.json").exists():
        control, candidate = endpoint("A16-CO4-S1337"), endpoint("C16-CONTEXT512")
        lines.append(f"WikiText BPB changed from {control[1]['bits_per_byte']:.5f} to {candidate[1]['bits_per_byte']:.5f}; source-wise held-out losses and eight-task capability samples are in the run archive.")
    report = RESEARCH / "CONTEXT_512.md"
    report.write_text("\n".join(lines) + "\n")
    upload([audit_path, RESEARCH / "prompt_request_lengths.jsonl",
            decision_path, report], "CONTEXT-GATE-REPORT")


def endpoint(experiment_id: str) -> tuple[dict, dict]:
    root = RESEARCH / "runs" / experiment_id
    validation = [json.loads(line) for line in (root / "evaluations.jsonl").open()][-1]
    wiki = json.loads((root / "wikitext103.json").read_text())
    return validation, wiki


def decision(experiment_id: str, parent: str, mechanism: str) -> dict:
    state = json.loads(STATE.read_text())
    prior = next((row for row in state["decisions"] if row.get("experiment_id") == experiment_id
                  and row.get("mechanism") == mechanism), None)
    if prior is not None:
        return prior
    candidate, wiki_candidate = endpoint(experiment_id)
    control, wiki_control = endpoint(parent)
    frozen_mix = json.loads((ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json").read_text())["mixture_definition"]
    weighted = lambda row: sum(frozen_mix[source] * row["source_losses"][source] for source in frozen_mix)
    delta = weighted(candidate) / weighted(control) - 1
    bpb_delta = wiki_candidate["bits_per_byte"] / wiki_control["bits_per_byte"] - 1
    if delta <= -.01 and bpb_delta <= -.005:
        classification = "PROVISIONAL_POSITIVE"
        reason = "held-out loss and BPB improved by registered screening amounts; capability confirmation remains"
    elif delta >= .05 or bpb_delta >= .05:
        classification = "PROVISIONAL_NEGATIVE"
        reason = "held-out loss or BPB regressed by at least 5%"
    else:
        classification = "AMBIGUOUS"
        reason = "screening changes do not clear the quality threshold on both LM measures"
    result = {"at": utc(), "experiment_id": experiment_id, "parent": parent,
              "mechanism": mechanism, "classification": classification,
              "heldout_relative_change": delta, "bpb_relative_change": bpb_delta,
              "reason": reason}
    state["decisions"].append(result)
    atomic_json(STATE, state)
    event(experiment_id, "SCREEN_DECISION", **{k: v for k, v in result.items() if k != "experiment_id"})
    return result


def value_residual_report() -> None:
    parent = "A16-CO4-S1337"
    candidate = "B16-VALUE-RESIDUAL-S1337"
    control_validation, control_wiki = endpoint(parent)
    candidate_validation, candidate_wiki = endpoint(candidate)
    mix = json.loads((ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json").read_text())["mixture_definition"]
    weighted = lambda row: sum(mix[source] * row["source_losses"][source] for source in mix)
    from posthack_uncertainty import paired_capability
    paired = paired_capability(RESEARCH / "runs" / candidate / "capability_samples.json",
                               RESEARCH / "runs" / parent / "capability_samples.json")
    def raw_accuracy(name: str) -> float:
        result = json.loads((RESEARCH / "runs" / name / "capability_screen.json").read_text())
        return sum(float(result["results"][task]["acc,none"])
                   for task in ("hellaswag", "arc_easy", "piqa", "winogrande")) / 4
    control_accuracy, candidate_accuracy = raw_accuracy(parent), raw_accuracy(candidate)
    lines = ["# Post-MOD value residual Co4", "",
             "The bounded 16M screen mixes the first block's post-MOD V with each later block's post-MOD V, initially at 0.5/0.5. The causal GQA attention backbone, tokenizer, data order, optimizer and 150M-token WSD schedule match the modern Co4 control. No cross-layer skip or other architecture change is in this first screen.", "",
             "| Measure | Co4 control | Value residual | Change |", "|---|---:|---:|---:|",
             f"| Frozen-mixture held-out loss | {weighted(control_validation):.5f} | {weighted(candidate_validation):.5f} | {weighted(candidate_validation)/weighted(control_validation)-1:+.2%} |",
             f"| WikiText-103 BPB | {control_wiki['bits_per_byte']:.5f} | {candidate_wiki['bits_per_byte']:.5f} | {candidate_wiki['bits_per_byte']/control_wiki['bits_per_byte']-1:+.2%} |",
             f"| Four-task raw accuracy | {control_accuracy:.4f} | {candidate_accuracy:.4f} | {paired['four_task_equal_weight_delta']:+.4f} |",
             "", f"The four-task paired item-bootstrap 95% interval for value residual minus control is {paired['four_task_equal_weight_ci95']}; it does not include training-seed uncertainty. Source-wise losses, normalized task variants, raw samples and checkpoint hashes are in the run archives. A one-scalar long-skip follow-up is separately evidence/time gated after the mandatory and named stretch work.", ""]
    report = RESEARCH / "VALUE_RESIDUAL.md"
    report.write_text("\n".join(lines))
    atomic_json(RESEARCH / "value_residual_paired_capability.json", paired)
    upload([report, RESEARCH / "value_residual_paired_capability.json"], "VALUE-RESIDUAL-REPORT")


def main() -> None:
    ids = {item for _, left, right in PAIRS for item in (left, right)}
    missing = [item for item in ids if not (RESEARCH / "runs" / item / "hf_archive_receipt.json").exists()]
    if missing:
        raise RuntimeError(f"Phase 1 endpoints are not complete and archived: {sorted(missing)}")
    if not (RESEARCH / "matched_co4_dense_analysis.json").exists():
        analyze()
        upload([RESEARCH / "matched_co4_dense_analysis.json", RESEARCH / "MATCHED_CO4_DENSE.md"], "PHASE1-ANALYSIS")
    maybe_extend_8m()
    milestone_receipt = RESEARCH / "phase1_milestone_archive_receipt.json"
    if not milestone_receipt.exists():
        paths = []
        for experiment_id in sorted(ids):
            run = RESEARCH / "runs" / experiment_id / "milestones"
            if (run / "50m.pt").exists():
                paths.append(run / "50m.pt")
            if experiment_id.startswith("A8-") and (run / "10m.pt").exists():
                paths.append(run / "10m.pt")
        if paths:
            atomic_json(milestone_receipt, {"at": utc(), "files": upload(paths, "PHASE1-MILESTONES")})
    state = json.loads(STATE.read_text())
    if "16m/value_residual/12" not in state["throughput_calibration"] or "16m/value_residual/16" not in state["throughput_calibration"]:
        invoke([sys.executable, "scripts/posthack_calibrate.py", "--regimes", "16m",
                "--architectures", "co4", "--threads", "12", "16",
                "--variant", "value_residual"], RESEARCH / "value_residual_calibration.log", "value-residual calibration")
    state = json.loads(STATE.read_text())
    calibrated = max((state["throughput_calibration"][f"16m/value_residual/{threads}"] for threads in (12, 16)),
                     key=lambda item: item["tokens_per_second"])
    register_screen(experiment_id="B16-VALUE-RESIDUAL-S1337", config_path=CONFIG,
                    variant="value_residual", budget=150_000_000,
                    parent_control="A16-CO4-S1337",
                    hypothesis="Post-MOD first-layer value residual improves Co4 quality or efficiency",
                    throughput_tps=calibrated["tokens_per_second"],
                    estimate_provenance=calibrated["experiment_id"])
    run_screen("B16-VALUE-RESIDUAL-S1337", CONFIG,
               calibrated["threads"], calibrated["tokens_per_second"])
    decision("B16-VALUE-RESIDUAL-S1337", "A16-CO4-S1337", "post-MOD value residual")
    value_residual_report()
    tokenizer_report = RESEARCH / "tokenizers/16k_report.json"
    if not tokenizer_report.exists():
        invoke([sys.executable, "scripts/posthack_tokenizer_audit.py"],
               RESEARCH / "tokenizer16k.log", "16K common-sample tokenizer")
        upload([tokenizer_report, RESEARCH / "tokenizers/final_corpus_16k.json"], "TOKENIZER-16K")
    context_gate()
    if not (RESEARCH / "retokenized_subset/comparison.json").exists():
        invoke([sys.executable, "scripts/posthack_tokenizer_study.py"],
               RESEARCH / "retokenized_subset/study.log", "matched 4K/8K tokenizer study")
    index_report = RESEARCH / "research_d/index_report.json"
    if not index_report.exists():
        invoke([sys.executable, "scripts/posthack_research_d_index.py"],
               RESEARCH / "research_d/index_build.log", "Research-D metadata index")
    index_receipt = RESEARCH / "research_d/index_hf_archive_receipt.json"
    if not index_receipt.exists():
        index_archive, index_identity = compress_index(RESEARCH / "research_d/documents.jsonl")
        atomic_json(index_receipt, {"at": utc(), "files": upload(
            [index_report, index_archive, index_identity], "RESEARCH-D-INDEX")})
    baseline = state["throughput_calibration"]["16m/co4/16"]
    mixture = {"fineweb_edu": 0.50, "fineweb": 0.30, "wikipedia": 0.10, "finemath": 0.10}
    register_screen(experiment_id="D1-16M-REBALANCED-S1337", config_path=CONFIG,
                    variant="baseline", budget=150_000_000,
                    parent_control="A16-CO4-S1337",
                    hypothesis="Reducing Wikipedia allocation and increasing broad FineWeb improves generalization",
                    throughput_tps=baseline["tokens_per_second"],
                    estimate_provenance=baseline["experiment_id"], mixture_weights=mixture)
    run_screen("D1-16M-REBALANCED-S1337", CONFIG,
               baseline["threads"], baseline["tokens_per_second"])
    decision("D1-16M-REBALANCED-S1337", "A16-CO4-S1337", "D1 broad mixture")


if __name__ == "__main__":
    main()
