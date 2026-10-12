"""Sequential D3/D4, curriculum-growth, and one PHM-4 screen."""
from __future__ import annotations

import json
import sys
import time

from posthack_archive import compress_index, upload
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, event, sha256, utc
from posthack_phase1 import invoke
from posthack_phase2 import decision, endpoint
from posthack_single_run import register_screen, run_screen
from latticelm.config import LatticeConfig
from latticelm.model import build_model

CONFIG = RESEARCH / "configs/16m_co4.json"
PHM_CONFIG = RESEARCH / "configs/16m_phm4.json"
INDEX = RESEARCH / "research_d/documents.jsonl"
CAPABILITY_INDEX = RESEARCH / "research_d/capability_documents.jsonl"
PARENT = "A16-CO4-S1337"


def weighted_loss(experiment_id: str) -> float:
    validation, _ = endpoint(experiment_id)
    mix = json.loads((ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json").read_text())["mixture_definition"]
    return sum(mix[source] * validation["source_losses"][source] for source in mix)


def competition_accuracy(experiment_id: str) -> float:
    result = json.loads((RESEARCH / "runs" / experiment_id / "capability_screen.json").read_text())
    tasks = ("hellaswag", "arc_easy", "piqa", "winogrande")
    return sum(float(result["results"][task]["acc,none"]) for task in tasks) / len(tasks)


def no_conflicting_failure(name: str) -> tuple[bool, dict]:
    base_validation, _ = endpoint(PARENT)
    candidate_validation, _ = endpoint(name)
    worst_source = max(candidate_validation["source_losses"][source] /
                       base_validation["source_losses"][source] - 1
                       for source in base_validation["source_losses"])
    accuracy_change = competition_accuracy(name) - competition_accuracy(PARENT)
    return accuracy_change >= -.01 and worst_source <= .05, {
        "four_task_raw_accuracy_change": accuracy_change,
        "worst_source_loss_relative_change": worst_source}


def cost_proxy(experiment_id: str) -> tuple[float, float]:
    cfg = LatticeConfig.from_json(CONFIG)
    model = build_model(cfg)
    full = sum(parameter.numel() for parameter in model.parameters())
    block = sum(parameter.numel() for parameter in model.blocks[0].parameters())
    fixed = full - cfg.n_layers * block
    previous = 0
    flops = 0.0
    with (RESEARCH / "runs" / experiment_id / "metrics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            logical = int(row["logical_tokens"])
            active = fixed + block * int(row["active_depth"])
            flops += 6 * active * (logical - previous)
            previous = logical
    result = json.loads((RESEARCH / "runs" / experiment_id / "result.json").read_text())
    return flops, result["wall_seconds"]


def screen_data(experiment_id: str, mode: str, hypothesis: str,
                *, anneal: bool = False, growth: bool = False) -> dict:
    state = json.loads(STATE.read_text())
    if growth:
        calibration = max((state["throughput_calibration"][f"16m/progressive_depth/{threads}"]
                           for threads in (12, 16)), key=lambda row: row["tokens_per_second"])
    else:
        calibration = state["throughput_calibration"]["16m/co4/16"]
    budget = 150_000_000
    rule = {"curriculum_fraction": .25, "curriculum_strength": .8, "quality_floor": .70}
    extra: dict = {"sampling_mode": mode, "sampling_rule": rule,
                   "research_d_index_path": str(INDEX.relative_to(ROOT)),
                   "research_d_index_sha256": sha256(INDEX)}
    if anneal:
        extra["anneal_rule"] = {"at_fraction": .85, "quality_floor": .70,
                                "mixture_weights": {"fineweb_edu": .60, "finemath": .20,
                                                    "fineweb": .10, "wikipedia": .10}}
    if growth:
        extra["growth_schedule"] = [{"at_tokens": 18_750_000, "depth": 7},
                                    {"at_tokens": 37_500_000, "depth": 10}]
    variant = "progressive_depth" if growth else "baseline"
    parent = "E1-16M-GROWTH-STATIC" if growth else PARENT
    register_screen(experiment_id=experiment_id, config_path=CONFIG, variant=variant,
                    budget=budget, parent_control=parent, hypothesis=hypothesis,
                    throughput_tps=calibration["tokens_per_second"] * (.8 if growth else .9),
                    estimate_provenance=f"{calibration['experiment_id']} with {'20' if growth else '10'}% sampler allowance",
                    extra_fields=extra)
    result = run_screen(experiment_id, CONFIG, calibration["threads"],
                        calibration["tokens_per_second"] * (.8 if growth else .9))
    return result


def screen_growth_static() -> dict:
    state = json.loads(STATE.read_text())
    if not all(f"16m/progressive_depth/{threads}" in state["throughput_calibration"] for threads in (12, 16)):
        invoke([sys.executable, "scripts/posthack_calibrate.py", "--regimes", "16m",
                "--architectures", "co4", "--threads", "12", "16",
                "--variant", "progressive_depth"], RESEARCH / "growth_calibration.log", "progressive-depth calibration")
    state = json.loads(STATE.read_text())
    calibrated = max((state["throughput_calibration"][f"16m/progressive_depth/{threads}"] for threads in (12, 16)),
                     key=lambda item: item["tokens_per_second"])
    schedule = [{"at_tokens": 18_750_000, "depth": 7},
                {"at_tokens": 37_500_000, "depth": 10}]
    register_screen(experiment_id="E1-16M-GROWTH-STATIC", config_path=CONFIG,
                    variant="progressive_depth", budget=150_000_000,
                    parent_control=PARENT,
                    hypothesis="Function-preserving depth growth improves compute efficiency on static D0 data",
                    throughput_tps=calibrated["tokens_per_second"],
                    estimate_provenance=calibrated["experiment_id"],
                    extra_fields={"growth_schedule": schedule,
                                  "growth_optimizer_policy": "preallocated dormant blocks; old Muon/AdamW state retained; new block state initialized lazily on first gradient"})
    return run_screen("E1-16M-GROWTH-STATIC", CONFIG,
                      calibrated["threads"], calibrated["tokens_per_second"] * .8)


def screen_phm() -> dict:
    state = json.loads(STATE.read_text())
    if not all(f"16m/phm4_ffn/{threads}" in state["throughput_calibration"] for threads in (12, 16)):
        invoke([sys.executable, "scripts/posthack_calibrate.py", "--regimes", "16m",
                "--architectures", "phm4", "--threads", "12", "16",
                "--variant", "phm4_ffn"], RESEARCH / "phm4_calibration.log", "PHM-4 calibration")
    state = json.loads(STATE.read_text())
    calibrated = max((state["throughput_calibration"][f"16m/phm4_ffn/{threads}"] for threads in (12, 16)),
                     key=lambda item: item["tokens_per_second"])
    register_screen(experiment_id="G-16M-PHM4-FFN", config_path=PHM_CONFIG,
                    variant="phm4_ffn", budget=150_000_000,
                    parent_control=PARENT,
                    hypothesis="Learned PHM-4 FFN algebra merits future work at a matched 16M parameter budget",
                    throughput_tps=calibrated["tokens_per_second"],
                    estimate_provenance=calibrated["experiment_id"],
                    promotion_criterion="Meaningful quality per parameter or per wall/FLOP signal, not parameter novelty alone",
                    extra_fields={"phm_dimension": 4, "phm_scope": "FFN projections only"})
    return run_screen("G-16M-PHM4-FFN", PHM_CONFIG,
                      calibrated["threads"], calibrated["tokens_per_second"])


def interaction_report() -> None:
    ids = [PARENT, "D3-16M-CURRICULUM", "E1-16M-GROWTH-STATIC", "E2-16M-GROWTH-CURRICULUM"]
    losses = {name: weighted_loss(name) for name in ids}
    interaction = (losses["E2-16M-GROWTH-CURRICULUM"] - losses["E1-16M-GROWTH-STATIC"]
                   - losses["D3-16M-CURRICULUM"] + losses[PARENT])
    capabilities = {name: competition_accuracy(name) for name in ids}
    costs = {name: cost_proxy(name) for name in ids}
    capability_interaction = (capabilities["E2-16M-GROWTH-CURRICULUM"] - capabilities["E1-16M-GROWTH-STATIC"]
                              - capabilities["D3-16M-CURRICULUM"] + capabilities[PARENT])
    lines = ["# Curriculum-guided Co4 depth growth", "",
             "The 2×2 design compares full/static D0, full/curriculum D3, progressive/static E1, and progressive/curriculum E2. Progressive models begin at five of ten blocks, grow to seven at 18.75M tokens, and to ten at 37.5M. Newly active blocks clone the preceding block and begin with a zero output gate; the trainer verifies immediate logit parity at each event. Existing optimizer state is retained, and dormant new parameters acquire state on their first gradient.", "",
             "| Arm | Weighted held-out loss | Four-task raw accuracy screen | 6NT FLOP proxy (10¹⁵) | Training hours |", "|---|---:|---:|---:|---:|"]
    for name in ids:
        lines.append(f"| {name} | {losses[name]:.5f} | {capabilities[name]:.4f} | {costs[name][0]/1e15:.3f} | {costs[name][1]/3600:.2f} |")
    lines += ["", f"Loss interaction `E2 − E1 − D3 + D0` = **{interaction:+.5f}**; a negative value favors a synergistic curriculum/growth interaction.",
              f"Raw capability interaction on the same four tasks = **{capability_interaction:+.4f}**; a positive value favors synergy.",
              "", "The FLOP column uses a 6NT proxy with the measured active layer count at every step. It includes embedding/head parameters and excludes sampler, optimizer, and Co4 elementwise overhead. Token-matched endpoints have different active-layer FLOPs; capability-screen uncertainty limits claims from small percentage-point changes.", ""]
    (RESEARCH / "CGLS_CO4.md").write_text("\n".join(lines))
    event("E2-16M-GROWTH-CURRICULUM", "INTERACTION_ANALYZED",
          weighted_heldout_loss_interaction=interaction,
          raw_capability_interaction=capability_interaction)


def phm_report() -> None:
    experiment_id = "G-16M-PHM4-FFN"
    candidate = json.loads((RESEARCH / "runs" / experiment_id / "result.json").read_text())
    control = json.loads((RESEARCH / "runs" / PARENT / "result.json").read_text())
    _, wiki_candidate = endpoint(experiment_id)
    _, wiki_control = endpoint(PARENT)
    state = json.loads(STATE.read_text())
    rows = [state["throughput_calibration"][f"16m/phm4_ffn/{threads}"] for threads in (12, 16)]
    phm_cfg = LatticeConfig.from_json(PHM_CONFIG)
    control_cfg = LatticeConfig.from_json(CONFIG)
    with (RESEARCH / "RUN_REGISTRY.jsonl").open() as handle:
        parameters = {row["experiment_id"]: row["parameter_count"] for line in handle
                      if (row := json.loads(line)) and row["experiment_id"] in (experiment_id, PARENT)}
    phm_macs = 3 * phm_cfg.d_model * phm_cfg.ffn_hidden * phm_cfg.n_layers
    control_macs = 3 * control_cfg.d_model * control_cfg.ffn_hidden * control_cfg.n_layers
    lines = ["# PHM-4 FFN screen", "",
             "One PHM-4 FFN-only model was trained at a matched 16M parameter budget. The learned algebra uses four 4×4 matrices and four lower-dimensional transform blocks per linear map; attention and embeddings remain Co4. The 4096-wide PHM FFN reinvests parameter savings, so arithmetic is not reduced fourfold.", "",
             "| Quantity | PHM-4 | Co4 control |", "|---|---:|---:|",
             f"| Trainable parameters | {parameters[experiment_id]:,} | {parameters[PARENT]:,} |",
             f"| Dense-equivalent FFN matrix shapes per layer | ({2*phm_cfg.ffn_hidden}×{phm_cfg.d_model}), ({phm_cfg.d_model}×{phm_cfg.ffn_hidden}) | ({2*control_cfg.ffn_hidden}×{control_cfg.d_model}), ({control_cfg.d_model}×{control_cfg.ffn_hidden}) |",
             f"| FFN matrix multiply-adds per token | {phm_macs:,} | {control_macs:,} |",
             f"| Training tokens | {candidate['logical_tokens']:,} | {control['logical_tokens']:,} |",
             f"| Weighted held-out loss | {weighted_loss(experiment_id):.5f} | {weighted_loss(PARENT):.5f} |",
             f"| WikiText-103 BPB | {wiki_candidate['bits_per_byte']:.5f} | {wiki_control['bits_per_byte']:.5f} |",
             f"| Training hours | {candidate['wall_seconds']/3600:.2f} | {control['wall_seconds']/3600:.2f} |",
             f"| 12-thread calibration tokens/s | {rows[0]['tokens_per_second']:.0f} | {state['throughput_calibration']['16m/co4/12']['tokens_per_second']:.0f} |",
             f"| 16-thread calibration tokens/s | {rows[1]['tokens_per_second']:.0f} | {state['throughput_calibration']['16m/co4/16']['tokens_per_second']:.0f} |",
             f"| Peak calibration RSS (GiB) | {max(row['rss_bytes'] for row in rows)/2**30:.2f} | {max(state['throughput_calibration'][f'16m/co4/{threads}']['rss_bytes'] for threads in (12,16))/2**30:.2f} |",
             "", "PHM linear maps materialize dense-equivalent matrices; multiply-add work therefore tracks their 4096-wide FFN shapes rather than the number of free PHM parameters. A positive parameter result without a quality-per-wall signal is insufficient for promotion.", ""]
    (RESEARCH / "PHM4_SCREEN.md").write_text("\n".join(lines))


def data_report() -> None:
    arms = [PARENT, "D1-16M-REBALANCED-S1337", "D2-16M-CAPABILITY",
            "D3-16M-CURRICULUM", "D4-16M-ANNEAL"]
    if (RESEARCH / "runs/D5-16M-COMBINED/hf_archive_receipt.json").exists():
        arms.append("D5-16M-COMBINED")
    tasks = ("hellaswag", "arc_easy", "piqa", "winogrande")
    transfer = ("arc_challenge", "openbookqa", "boolq", "lambada_openai")
    selector = json.loads((RESEARCH / "research_d/capability_selector_report.json").read_text())
    d5 = json.loads((RESEARCH / "d5_combination_decision.json").read_text())
    lines = ["# Research-D data, curriculum, and WSD annealing", "",
             "All arms reuse certified DATA-D-v4 token shards and the same 16M Co4 backbone, tokenizer, optimizer and 150M-token horizon. D0 is the modern matched Co4 control. D1 changes only source mixture; D2 adds a 12.5% within-source capability-selected component from SciQ/CommonsenseQA/QASC train-only selector features; D3 biases early sampling by the registered two-signal difficulty score; D4 changes the final WSD-decay mixture and FineWeb-Edu quality floor; D5 combines early curriculum and late anneal only if its frozen gate passes.", "",
             "| Arm | Weighted held-out loss | WikiText BPB | Four-task raw | Transfer raw | Worst source loss |",
             "|---|---:|---:|---:|---:|---:|"]
    source_rows = {}
    for name in arms:
        validation, wiki = endpoint(name)
        capability = json.loads((RESEARCH / "runs" / name / "capability_screen.json").read_text())
        mean = lambda names: sum(float(capability["results"][task]["acc,none"])
                                 for task in names) / len(names)
        losses = validation["source_losses"]
        source_rows[name] = losses
        lines.append(f"| {name} | {weighted_loss(name):.5f} | {wiki['bits_per_byte']:.5f} | {mean(tasks):.4f} | {mean(transfer):.4f} | {max(losses.values()):.5f} |")
    lines.extend(["", "## Held-out loss by source", "",
                  "| Arm | FineWeb-Edu | FineWeb | Wikipedia | FineMath |",
                  "|---|---:|---:|---:|---:|"])
    for name in arms:
        values = source_rows[name]
        lines.append(f"| {name} | {values['fineweb_edu']:.5f} | {values['fineweb']:.5f} | {values['wikipedia']:.5f} | {values['finemath']:.5f} |")
    lines.extend(["", f"D5 was {'launched' if d5['run'] else 'gated out'}: {d5['reason']}. The exact D3/D4 quality, capability, source-regression, and protected-time inputs are frozen in `d5_combination_decision.json`.",
                  f"The capability selector scored {selector['scoring']['documents_scored']:,} indexed documents. Within-source score quantiles, selected counts, pinned training revisions, inherited DATA-D contamination registry, and randomly sampled text snippets for inspection are in `research_d/capability_selector_report.json`. Semantic overlap beyond the inherited exact/near-duplicate checks remains a limitation.", "",
                  "## Observed curriculum difficulty", "",
                  "The following values are cumulative accepted difficulty means from the registered Research-D sampler. The complete step series is preserved in the metrics file.", "",
                  "| D3 logical tokens | Accepted difficulty mean |", "|---:|---:|"])
    metrics = RESEARCH / "runs/D3-16M-CURRICULUM/metrics.jsonl"
    targets = (10_000_000, 37_500_000, 75_000_000, 150_000_000)
    next_target = 0
    with metrics.open() as handle:
        for line in handle:
            row = json.loads(line)
            if next_target >= len(targets) or row["logical_tokens"] < targets[next_target]:
                continue
            diagnostics = row["sampler_diagnostics"] or {}
            accepted = sum(value["accepted"] for value in diagnostics.values())
            difficulty = sum(value["accepted_difficulty_sum"] for value in diagnostics.values())
            lines.append(f"| {row['logical_tokens']:,} | {difficulty/max(1, accepted):.4f} |")
            next_target += 1
    lines.extend(["", "The difficulty score is a within-source mean of document length and first-256-token lexical-diversity percentiles, not an external measure of reasoning difficulty. D4's high-quality FineWeb-Edu threshold was registered before evaluation; source distribution and token exposure remain visible in the run registry and sampler metrics.", ""])
    report = RESEARCH / "DATA_CURRICULUM.md"
    report.write_text("\n".join(lines))
    upload([report, RESEARCH / "d5_combination_decision.json"], "DATA-CURRICULUM-REPORT")


def main() -> None:
    for required in ("B16-VALUE-RESIDUAL-S1337", "D1-16M-REBALANCED-S1337"):
        if not (RESEARCH / "runs" / required / "hf_archive_receipt.json").exists():
            raise RuntimeError(f"Phase 2 prerequisite is missing: {required}")
    if not INDEX.exists():
        raise RuntimeError("Research-D document index is required")
    selector_report = RESEARCH / "research_d/capability_selector_report.json"
    if not selector_report.exists():
        invoke([sys.executable, "scripts/posthack_capability_selector.py"],
               RESEARCH / "research_d/capability_selector.log", "disjoint capability selector")
    selector_receipt = RESEARCH / "research_d/capability_hf_archive_receipt.json"
    if not selector_receipt.exists():
        selector_archive, selector_identity = compress_index(CAPABILITY_INDEX)
        atomic_json(selector_receipt, {"at": utc(), "files": upload(
            [selector_report, selector_archive, selector_identity], "CAPABILITY-SELECTOR-INDEX")})
    baseline = json.loads(STATE.read_text())["throughput_calibration"]["16m/co4/16"]
    register_screen(experiment_id="D2-16M-CAPABILITY", config_path=CONFIG,
                    variant="baseline", budget=150_000_000, parent_control=PARENT,
                    hypothesis="A 12.5% within-source disjoint capability-selected component improves broad held-out capability",
                    throughput_tps=baseline["tokens_per_second"] * .9,
                    estimate_provenance=f"{baseline['experiment_id']} with 10% selector allowance",
                    extra_fields={"sampling_mode": "capability",
                                  "research_d_index_path": str(CAPABILITY_INDEX.relative_to(ROOT)),
                                  "research_d_index_sha256": sha256(CAPABILITY_INDEX),
                                  "sampling_rule": {"capability_floor": .90,
                                                    "capability_fraction": .125,
                                                    "selector_report_sha256": sha256(selector_report)}})
    run_screen("D2-16M-CAPABILITY", CONFIG, 16, baseline["tokens_per_second"] * .9)
    decision("D2-16M-CAPABILITY", PARENT, "D2 disjoint capability selector")
    screen_data("D3-16M-CURRICULUM", "curriculum",
                "An early easy/clean curriculum improves convergence at fixed D0 mixture")
    d3 = decision("D3-16M-CURRICULUM", PARENT, "D3 curriculum warmup")
    screen_data("D4-16M-ANNEAL", "quality_anneal",
                "High-quality FineWeb-Edu and FineMath annealing during WSD decay improves endpoint quality",
                anneal=True)
    d4 = decision("D4-16M-ANNEAL", PARENT, "D4 WSD quality anneal")
    guard3, measures3 = no_conflicting_failure("D3-16M-CURRICULUM")
    guard4, measures4 = no_conflicting_failure("D4-16M-ANNEAL")
    both = d3["classification"] == d4["classification"] == "PROVISIONAL_POSITIVE" and guard3 and guard4
    strongly = any(row["heldout_relative_change"] <= -.02 and row["bpb_relative_change"] <= -.02
                   for row in (d3, d4)) and all(row["classification"] != "PROVISIONAL_NEGATIVE"
                                               for row in (d3, d4)) and guard3 and guard4
    d5_decision_path = RESEARCH / "d5_combination_decision.json"
    if d5_decision_path.exists():
        d5_run = json.loads(d5_decision_path.read_text())["run"]
    else:
        state = json.loads(STATE.read_text())
        co4_tps = state["throughput_calibration"]["16m/co4/16"]["tokens_per_second"]
        # Preserve three mandatory 16M screens (E1, E2, PHM) with a 30%
        # throughput discount and evaluation/checkpoint/safety budgets.
        protected_seconds = 3 * (150_000_000 / (co4_tps * .7) + 7200 + 1800 + 3600)
        d5_seconds = 150_000_000 / (co4_tps * .9) + 7200 + 1800 + 3600
        time_passed = time.time() + protected_seconds + d5_seconds < state["archive_start_epoch"]
        d5_run = bool((both or strongly) and time_passed)
        atomic_json(d5_decision_path, {"created_utc": utc(), "run": d5_run,
                                       "time_gate_passed": time_passed,
                                       "protected_mandatory_seconds": protected_seconds,
                                       "d3": {"screen": d3, "failure_guard": measures3},
                                       "d4": {"screen": d4, "failure_guard": measures4},
                                       "criterion": "both provisional LM positives or one >=2% held-out and BPB positive, with <=1 point raw capability and <=5% worst-source regression in both arms",
                                       "reason": "evidence and time gates passed" if d5_run else
                                                 "D3/D4 evidence or failure guard did not clear" if not (both or strongly) else
                                                 "D5 would consume protected E1/E2/PHM time"})
    if d5_run:
        screen_data("D5-16M-COMBINED", "curriculum_anneal",
                    "Independent early curriculum and late quality anneal gains combine",
                    anneal=True)
        decision("D5-16M-COMBINED", PARENT, "D5 combined curriculum and anneal")
    else:
        event("D5-16M-COMBINED", "RUN_REJECTED", reason="conditional D3/D4 evidence or failure guard did not clear")
    data_report()
    screen_growth_static()
    decision("E1-16M-GROWTH-STATIC", PARENT, "progressive depth on static D0")
    screen_data("E2-16M-GROWTH-CURRICULUM", "curriculum",
                "Synchronizing depth growth with curriculum improves generalization beyond either alone",
                growth=True)
    interaction_report()
    upload([RESEARCH / "CGLS_CO4.md"], "GROWTH-INTERACTION")
    screen_phm()
    decision("G-16M-PHM4-FFN", PARENT, "PHM-4 FFN")
    phm_report()
    upload([RESEARCH / "PHM4_SCREEN.md"], "PHM4-REPORT")


if __name__ == "__main__":
    main()
