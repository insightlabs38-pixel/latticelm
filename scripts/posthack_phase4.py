"""Controlled developmental-expert birth experiment with one heavy trainer."""
from __future__ import annotations

import json
import math
import sys

from posthack_archive import upload
from posthack_campaign import RESEARCH, ROOT, STATE, event, sha256
from posthack_phase1 import invoke
from posthack_phase2 import decision
from posthack_single_run import register_screen, run_screen

CONFIG = RESEARCH / "configs/8m_co4.json"
INDEX = RESEARCH / "research_d/documents.jsonl"
ARMS = (
    ("F0-8M-FIXED-EXPERTS", "moe_fixed", "none", 0),
    ("F1-8M-NAIVE-BIRTH", "moe_growth", "random", 0),
    ("F2-8M-CLONE-BIRTH", "moe_growth", "clone", 0),
    ("F3-8M-CLONE-CATCHUP", "moe_growth", "clone", 5_000_000),
)
BUDGET = 100_000_000
LATER_MIXTURE = {"fineweb_edu": .40, "fineweb": .20, "wikipedia": .20, "finemath": .20}


def calibrate() -> None:
    state = json.loads(STATE.read_text())
    for variant in ("moe_fixed", "moe_growth"):
        if all(f"8m/{variant}/{threads}" in state["throughput_calibration"] for threads in (12, 16)):
            continue
        invoke([sys.executable, "scripts/posthack_calibrate.py", "--regimes", "8m",
                "--architectures", "co4", "--threads", "12", "16",
                "--variant", variant], RESEARCH / f"{variant}_calibration.log",
               f"{variant} calibration")


def registration(arm: tuple[str, str, str, int]) -> tuple[int, float]:
    name, variant, strategy, catchup = arm
    state = json.loads(STATE.read_text())
    rows = [state["throughput_calibration"][f"8m/{variant}/{threads}"] for threads in (12, 16)]
    best = max(rows, key=lambda row: row["tokens_per_second"])
    rule = {"transition_at_fraction": .25, "later_mixture": LATER_MIXTURE,
            "birth_strategy": strategy, "specialty_catchup_tokens": catchup,
            "parent_rule": "maximum accumulated FineMath routing among initial four; lower index breaks ties",
            "child_rule": "preferred parent first, then each remaining parent once; deterministic alternate routing between cloned parent/child",
            "optimizer_policy": "old parameter state retained; newborn expert state initialized lazily; cloned weights inherit no optimizer moments"}
    register_screen(experiment_id=name, config_path=CONFIG, variant=variant, budget=BUDGET,
                    parent_control="F0-8M-FIXED-EXPERTS" if name != "F0-8M-FIXED-EXPERTS" else "A8-CO4-S1337",
                    hypothesis={"none": "Fixed experts provide the specialist-capacity control",
                                "random": "Random newborn experts measure the benefit of late capacity alone",
                                "clone": "Cloned late experts inherit mature structure and can specialize"}[strategy]
                    + ("; targeted FineMath catch-up adds a specialty" if catchup else ""),
                    throughput_tps=best["tokens_per_second"] * .8,
                    estimate_provenance=f"{best['experiment_id']} with 20% Research-D sampler allowance",
                    promotion_criterion="F3 beats F0/F1 on held-out LM or broad capability at matched active FLOPs, with specialization evidence",
                    extra_fields={"sampling_mode": "curriculum",
                                  "sampling_rule": {"curriculum_fraction": .25,
                                                    "curriculum_strength": .8},
                                  "research_d_index_path": str(INDEX.relative_to(ROOT)),
                                  "research_d_index_sha256": sha256(INDEX),
                                  "moe_rule": rule,
                                  "active_flop_policy": "one FFN expert per token and layer; router and load balance included"})
    return best["threads"], best["tokens_per_second"] * .8


def entropy_and_mi(domain_counts: list[list[int]]) -> tuple[float, float]:
    total = sum(sum(row) for row in domain_counts)
    if not total:
        return 0.0, 0.0
    expert = [sum(row[col] for row in domain_counts) for col in range(8)]
    domains = [sum(row) for row in domain_counts]
    entropy = -sum((count / total) * math.log(count / total)
                   for count in expert if count)
    mi = sum((domain_counts[d][e] / total) *
             math.log(domain_counts[d][e] * total / (domains[d] * expert[e]))
             for d in range(4) for e in range(8) if domain_counts[d][e])
    return entropy, mi


def report() -> None:
    mixture = json.loads((ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json").read_text())["mixture_definition"]
    rows = []
    for name, _, strategy, catchup in ARMS:
        path = RESEARCH / "runs" / name
        result = json.loads((path / "result.json").read_text())
        validation = [json.loads(line) for line in (path / "evaluations.jsonl").open()][-1]
        wiki = json.loads((path / "wikitext103.json").read_text())
        counts = [[0] * 8 for _ in range(4)]
        entropy = []
        with (path / "routing.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                for layer in row["routing"]["domain_counts_by_layer"]:
                    for domain in range(4):
                        for expert in range(8):
                            counts[domain][expert] += layer[domain][expert]
                entropy.extend(row["routing"]["router_entropy_by_layer"])
        utilization_entropy, mutual_information = entropy_and_mi(counts)
        rows.append({"name": name, "strategy": strategy, "catchup_tokens": catchup,
                     "tokens": result["logical_tokens"], "wall_hours": result["wall_seconds"] / 3600,
                     "heldout_loss": sum(mixture[source] * validation["source_losses"][source]
                                         for source in mixture), "wiki_bpb": wiki["bits_per_byte"],
                     "domain_expert_counts": counts, "utilization_entropy_nats": utilization_entropy,
                     "domain_expert_mi_nats": mutual_information,
                     "mean_router_entropy_nats": sum(entropy) / len(entropy)})
    (RESEARCH / "developmental_experts_analysis.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Developmental experts", "",
             "All arms use the 4K tokenizer, eight-block Co4 backbone, top-one routing, one active FFN expert per token and block, 100M logical tokens, the same 25% difficulty-curriculum transition, and a registered FineMath source increase from about 10% to 20%. F0 starts with eight experts; F1–F3 start with four and expand to eight at the transition. F2/F3 clone mature parents, and F3 directs FineMath tokens toward the children for 5M tokens within the fixed budget.", "",
             "| Arm | Held-out loss | WikiText BPB | Hours | Domain/expert MI (nats) | Expert utilization entropy |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['heldout_loss']:.5f} | {row['wiki_bpb']:.5f} | {row['wall_hours']:.2f} | {row['domain_expert_mi_nats']:.4f} | {row['utilization_entropy_nats']:.4f} |")
    lines += ["", "The routing log contains per-step token fractions by expert and domain, router entropy, load-balancing loss, and catch-up status. Domain/expert mutual information describes attribution; a positive quality difference is required before claiming that specialization helped. The active-parameter count excludes dormant experts, while the registered total count includes all eight. The 6NT active-parameter proxy omits routing and irregular-dispatch overhead, so wall time remains a separate fairness axis.", ""]
    diagnostics = json.loads((RESEARCH / "runs/F3-8M-CLONE-CATCHUP/expert_diagnostics.json").read_text())
    similarities = diagnostics["parent_child_representation_and_parameter_distance"]
    if similarities:
        mean_cosine = sum(row["output_cosine_on_finemath_activation"] for row in similarities) / len(similarities)
        mean_distance = sum(row["relative_parameter_distance"] for row in similarities) / len(similarities)
        lines += [f"F3 parent/child mean output cosine on actual FineMath layer activations: {mean_cosine:.4f}; mean relative parameter distance: {mean_distance:.4f}. The first-router FineWeb-Edu versus FineMath gradient cosine was {diagnostics['fineweb_edu_vs_finemath_first_router_gradient_cosine']}. These are fixed-window endpoint diagnostics, not population estimates.", ""]
    lines += ["For every arm, `expert_diagnostics.json` records source-wise forced-expert loss, one-expert ablation effects, parent/child route-swap counterfactuals, layer-level representation similarity, parameter distance, and one domain gradient-conflict estimate.", ""]
    (RESEARCH / "DEVELOPMENTAL_EXPERTS.md").write_text("\n".join(lines))
    event("F3-8M-CLONE-CATCHUP", "DEVELOPMENTAL_EXPERTS_ANALYZED",
          rows=len(rows), report_sha256=sha256(RESEARCH / "DEVELOPMENTAL_EXPERTS.md"))


def main() -> None:
    if not (RESEARCH / "runs/G-16M-PHM4-FFN/hf_archive_receipt.json").exists():
        raise RuntimeError("Phase 3 PHM-4 screen must finish before developmental experts")
    if not INDEX.exists():
        raise RuntimeError("Research-D index is required")
    calibrate()
    for arm in ARMS:
        threads, tps = registration(arm)
        run_screen(arm[0], CONFIG, threads, tps)
        run = RESEARCH / "runs" / arm[0]
        diagnostic = run / "expert_diagnostics.json"
        if not diagnostic.exists():
            invoke([sys.executable, "scripts/posthack_moe_diagnostics.py",
                    "--checkpoint", str(run / "latest.pt"),
                    "--output", str(diagnostic)], run / "diagnostics.log",
                   "expert counterfactual diagnostics")
            upload([diagnostic], f"{arm[0]}-DIAGNOSTICS")
        if arm[0] != "F0-8M-FIXED-EXPERTS":
            decision(arm[0], "F0-8M-FIXED-EXPERTS", "developmental expert birth")
    report()
    upload([RESEARCH / "DEVELOPMENTAL_EXPERTS.md",
            RESEARCH / "developmental_experts_analysis.json"], "DEVELOPMENTAL-EXPERTS-REPORT")


if __name__ == "__main__":
    main()
