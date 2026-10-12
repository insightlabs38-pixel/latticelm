"""Evidence-gated Phase 2: value residual, tokenizer audit, and D1 mixture."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from posthack_analyze_phase1 import PAIRS, analyze
from posthack_archive import compress_index, upload
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, event, utc
from posthack_phase1 import invoke
from posthack_single_run import register_screen, run_screen

CONFIG = RESEARCH / "configs/16m_co4.json"


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


def main() -> None:
    ids = {item for _, left, right in PAIRS for item in (left, right)}
    missing = [item for item in ids if not (RESEARCH / "runs" / item / "hf_archive_receipt.json").exists()]
    if missing:
        raise RuntimeError(f"Phase 1 endpoints are not complete and archived: {sorted(missing)}")
    if not (RESEARCH / "matched_co4_dense_analysis.json").exists():
        analyze()
        upload([RESEARCH / "matched_co4_dense_analysis.json", RESEARCH / "MATCHED_CO4_DENSE.md"], "PHASE1-ANALYSIS")
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
    tokenizer_report = RESEARCH / "tokenizers/16k_report.json"
    if not tokenizer_report.exists():
        invoke([sys.executable, "scripts/posthack_tokenizer_audit.py"],
               RESEARCH / "tokenizer16k.log", "16K common-sample tokenizer")
        upload([tokenizer_report, RESEARCH / "tokenizers/final_corpus_16k.json"], "TOKENIZER-16K")
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
