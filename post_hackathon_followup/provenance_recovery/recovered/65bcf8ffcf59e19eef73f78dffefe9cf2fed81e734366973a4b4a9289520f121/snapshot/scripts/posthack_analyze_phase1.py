"""Compare registered Co4/dense controls on paired quality and cost axes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from posthack_campaign import RESEARCH, ROOT, atomic_json, event, utc

PAIRS = [
    ("8M seed 1337", "A8-CO4-S1337", "A8-DENSE-S1337"),
    ("8M seed 2026", "A8-CO4-S2026", "A8-DENSE-S2026"),
    ("16M seed 1337", "A16-CO4-S1337", "A16-DENSE-S1337"),
    ("32M seed 1337", "A32-CO4-S1337", "A32-DENSE-S1337"),
]
COMPETITION = ("hellaswag", "arc_easy", "piqa", "winogrande")
TRANSFER = ("arc_challenge", "openbookqa", "boolq", "lambada_openai")


def last_jsonl(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if not rows:
        raise RuntimeError(f"empty evidence file: {path}")
    return rows[-1]


def source_mix() -> dict[str, float]:
    manifest = json.loads((ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json").read_text())
    return manifest["mixture_definition"]


def source_bytes_per_token() -> dict[str, float]:
    manifest = json.loads((ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json").read_text())
    totals = {name: [0, 0] for name in manifest["mixture_definition"]}
    base = ROOT / "artifacts/data/data_d_v4/canonical-2250m"
    for child in manifest["shards"]:
        metadata = json.loads((base / child["manifest_path"]).read_text())
        values = totals[metadata["source"]]
        values[0] += metadata["raw_byte_count"]
        values[1] += metadata["token_count"]
    return {name: values[0] / values[1] for name, values in totals.items()}


def paired_interval(candidate: list[float], control: list[float], seed: int = 2026) -> dict:
    if len(candidate) != len(control) or not candidate:
        return {"status": "unavailable"}
    differences = np.asarray(candidate, dtype=float) - np.asarray(control, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
    return {"status": "paired_bootstrap", "n": len(differences),
            "candidate_minus_control": float(differences.mean()),
            "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]}


def capability(result: dict, tasks: tuple[str, ...]) -> dict[str, dict]:
    answer = {}
    for task in tasks:
        row = result["results"][task]
        answer[task] = {"raw_accuracy": row.get("acc,none"),
                        "normalized_accuracy": row.get("acc_norm,none"),
                        "sample_count": result.get("n-samples", {}).get(task, {}).get("effective")}
    return answer


def summarize(experiment_id: str, mix: dict[str, float], bytes_per_token: dict[str, float]) -> dict:
    run = RESEARCH / "runs" / experiment_id
    result = json.loads((run / "result.json").read_text())
    if result["status"] != "COMPLETED":
        raise RuntimeError(f"incomplete run: {experiment_id}")
    evaluation = last_jsonl(run / "evaluations.jsonl")
    wiki = json.loads((run / "wikitext103.json").read_text())
    screen = json.loads((run / "capability_screen.json").read_text())
    weighted_loss = sum(mix[source] * evaluation["source_losses"][source] for source in mix)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").open(encoding="utf-8")]
    sampled_counts = {source: 0 for source in mix}
    for row in metrics:
        for source in row["source_labels"]:
            sampled_counts[source] += row["context_length"]
    raw_bytes_estimate = sum(count * bytes_per_token[source] for source, count in sampled_counts.items())
    return {"experiment_id": experiment_id, "logical_tokens": result["logical_tokens"],
            "processed_tokens": result["processed_tokens"], "steps": result["steps"],
            "training_wall_seconds": result["wall_seconds"],
            "effective_tokens_per_second": result["logical_tokens"] / result["wall_seconds"],
            "source_losses": evaluation["source_losses"],
            "paired_source_losses": evaluation["paired_losses"],
            "weighted_heldout_loss": weighted_loss,
            "wikitext_bpb": wiki["bits_per_byte"], "wikitext_ppl": wiki["perplexity"],
            "wikitext_nll_nats": wiki["negative_log_likelihood_nats"],
            "wikitext_bytes_scored": wiki["utf8_bytes"],
            "wikitext_windowing_policy": wiki["windowing_policy"],
            "capability_competition": capability(screen, COMPETITION),
            "capability_transfer": capability(screen, TRANSFER),
            "estimated_raw_bytes_seen": raw_bytes_estimate,
            "estimated_raw_bytes_policy": "sampled source tokens times certified source-level raw-byte/token ratio",
            "sampled_source_tokens": sampled_counts,
            "checkpoint_sha256": result["checkpoint_sha256"]}


def analyze() -> dict:
    mix = source_mix()
    ratios = source_bytes_per_token()
    ids = {item for _, left, right in PAIRS for item in (left, right)}
    runs = {experiment_id: summarize(experiment_id, mix, ratios) for experiment_id in sorted(ids)}
    comparisons = []
    lines = ["# Modern matched Co4 versus dense", "",
             "All runs use the final 4K tokenizer, certified DATA-D-v4, matched 32×256 batch, QK-Norm, GQA, RoPE, SwiGLU, untied embeddings, Muon/AdamW, and the 2/83/15 WSD schedule. Co4 and dense configurations at each scale differ by less than 0.1% in trainable parameters. Held-out aggregate loss uses the frozen DATA-D source weights. Raw-byte exposure is estimated from certified source-level ratios; it is not an exact per-document count.", "",
             "| Pair | Co4 held-out | Dense held-out | Co4 BPB | Dense BPB | Co4 train h | Dense train h | Co4 est. raw GB | Dense est. raw GB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, co4_id, dense_id in PAIRS:
        a, b = runs[co4_id], runs[dense_id]
        candidate = [x for source in sorted(mix) for x in a["paired_source_losses"][source]]
        control = [x for source in sorted(mix) for x in b["paired_source_losses"][source]]
        interval = paired_interval(candidate, control)
        comparison = {"pair": label, "co4": co4_id, "dense": dense_id,
                      "heldout_loss_delta_co4_minus_dense": a["weighted_heldout_loss"] - b["weighted_heldout_loss"],
                      "wikitext_bpb_delta_co4_minus_dense": a["wikitext_bpb"] - b["wikitext_bpb"],
                      "training_wall_ratio_co4_over_dense": a["training_wall_seconds"] / b["training_wall_seconds"],
                      "four_task_raw_mean_co4": float(np.mean([a["capability_competition"][task]["raw_accuracy"] for task in COMPETITION])),
                      "four_task_raw_mean_dense": float(np.mean([b["capability_competition"][task]["raw_accuracy"] for task in COMPETITION])),
                      "transfer_raw_mean_co4": float(np.mean([a["capability_transfer"][task]["raw_accuracy"] for task in TRANSFER])),
                      "transfer_raw_mean_dense": float(np.mean([b["capability_transfer"][task]["raw_accuracy"] for task in TRANSFER])),
                      "paired_window_bootstrap": interval}
        comparisons.append(comparison)
        lines.append(f"| {label} | {a['weighted_heldout_loss']:.4f} | {b['weighted_heldout_loss']:.4f} | {a['wikitext_bpb']:.4f} | {b['wikitext_bpb']:.4f} | {a['training_wall_seconds']/3600:.2f} | {b['training_wall_seconds']/3600:.2f} | {a['estimated_raw_bytes_seen']/1e9:.3f} | {b['estimated_raw_bytes_seen']/1e9:.3f} |")
    lines += ["", "The paired bootstrap uses the same 32 source-validation windows for each pair. Its interval covers that fixed window sample, not model-seed uncertainty. The 8M pair has two seeds; 16M and 32M each have one. Capability screens use fixed first 1,000 examples per task and require a full-suite confirmation before a small point-estimate gain is treated as real.", "",
              "| Pair | Co4 four-task raw mean | Dense four-task raw mean | Co4 transfer raw mean | Dense transfer raw mean |",
              "|---|---:|---:|---:|---:|"]
    for comparison in comparisons:
        lines.append(f"| {comparison['pair']} | {comparison['four_task_raw_mean_co4']:.4f} | {comparison['four_task_raw_mean_dense']:.4f} | {comparison['transfer_raw_mean_co4']:.4f} | {comparison['transfer_raw_mean_dense']:.4f} |")
    for comparison in comparisons:
        interval = comparison["paired_window_bootstrap"]
        lines.append(f"- **{comparison['pair']}**: Co4 − dense held-out loss {comparison['heldout_loss_delta_co4_minus_dense']:+.4f}; BPB {comparison['wikitext_bpb_delta_co4_minus_dense']:+.4f}; paired-window 95% interval {interval.get('ci95')}; wall-time ratio {comparison['training_wall_ratio_co4_over_dense']:.3f}.")
    with (RESEARCH / "RUN_REGISTRY.jsonl").open() as handle:
        parameters = {row["experiment_id"]: row["parameter_count"] for line in handle
                      if (row := json.loads(line)) and row["experiment_id"] in runs}
    trends = {}
    for architecture, selected in (("co4", ["A8-CO4-S1337", "A16-CO4-S1337", "A32-CO4-S1337"]),
                                   ("dense", ["A8-DENSE-S1337", "A16-DENSE-S1337", "A32-DENSE-S1337"])):
        xs = np.log([parameters[name] for name in selected])
        ys = [runs[name]["weighted_heldout_loss"] for name in selected]
        slope, intercept = np.polyfit(xs, ys, 1)
        trends[architecture] = {"heldout_loss_per_ln_parameter_slope": float(slope),
                                "intercept": float(intercept),
                                "points": selected}
    lines += ["", "Local three-point fits of held-out loss against log trainable parameters use the seed-1337 8M, 16M and 32M endpoints: "
              + "; ".join(f"{arch} slope {row['heldout_loss_per_ln_parameter_slope']:+.4f}" for arch, row in trends.items())
              + ". Different scales have different token budgets, so these are descriptive local trends, not a scaling-law extrapolation.", "",
              "Token-matched and parameter-matched conclusions follow directly from the table. The 6NT proxy is close for these models but does not include Co4 MOD arithmetic or measured hardware utilization; a precise FLOP-efficiency conclusion requires an operator-count audit. Raw task means above remain screening estimates, with normalized task variants in the JSON evidence.", ""]
    report = {"schema": "posthack-matched-control-analysis-v1", "created_utc": utc(),
              "runs": runs, "comparisons": comparisons, "local_scaling_trends": trends,
              "limitations": ["8M has only two seeds", "16M/32M have one seed", "capability evaluations are 1000-example screens", "raw bytes and FLOPs are estimated"]}
    atomic_json(RESEARCH / "matched_co4_dense_analysis.json", report)
    (RESEARCH / "MATCHED_CO4_DENSE.md").write_text("\n".join(lines))
    for comparison in comparisons:
        event(comparison["co4"], "PAIRED_CONTROL_ANALYZED", comparison=comparison)
    return report


def main() -> None:
    report = analyze()
    print(json.dumps({"pairs": len(report["comparisons"]), "report": str(RESEARCH / "MATCHED_CO4_DENSE.md")}, indent=2))


if __name__ == "__main__":
    main()
