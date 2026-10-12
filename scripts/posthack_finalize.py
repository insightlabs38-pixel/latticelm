"""Final evidence plots, explicit synthesis, checksums, and HF-only archive."""
from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from posthack_archive import upload
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, event, sha256, utc
from posthack_analyze_phase1 import source_bytes_per_token


def json_file(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def source_snapshot() -> tuple[Path, Path]:
    paths = sorted({*ROOT.glob("scripts/posthack*.py"),
                    ROOT / "src/latticelm/posthack_variants.py",
                    ROOT / "src/latticelm/research_sampling.py",
                    ROOT / "src/latticelm/model.py",
                    ROOT / "src/latticelm/final_recipe.py",
                    ROOT / "scripts/evaluate_gibc.py",
                    ROOT / "scripts/evaluate_wikitext103.py",
                    *ROOT.glob("tests/test_posthack*.py"),
                    ROOT / "tests/test_co4_causal.py",
                    *RESEARCH.glob("configs/*.json"),
                    *RESEARCH.glob("latticelm-posthack-*.service")})
    paths = [path for path in paths if path.is_file()]
    manifest = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    manifest_path = RESEARCH / "SOURCE_SNAPSHOT_MANIFEST.json"
    atomic_json(manifest_path, {"schema": "posthack-source-snapshot-v1",
                                "created_utc": utc(), "files": manifest,
                                "source_commit": json.loads(STATE.read_text())["source_commit"],
                                "git_policy": "research source preserved on private Hugging Face only; no Git commit or push"})
    archive = RESEARCH / "source-snapshot-final.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for path in paths:
            out.write(path, arcname=str(path.relative_to(ROOT)))
    return manifest_path, archive


def load_run(name: str) -> tuple[dict, dict, dict] | None:
    run = RESEARCH / "runs" / name
    result = json_file(run / "result.json")
    wiki = json_file(run / "wikitext103.json")
    capability = json_file(run / "capability_screen.json")
    if not result or not wiki or not capability or result.get("status") != "COMPLETED":
        return None
    return result, wiki, capability


def plots() -> list[Path]:
    out = RESEARCH / "plots"
    out.mkdir(exist_ok=True)
    registrations = [json.loads(line) for line in (RESEARCH / "RUN_REGISTRY.jsonl").open()]
    names = [row["experiment_id"] for row in registrations
             if row["experiment_id"].startswith(("A8-", "A16-", "A32-", "R32-", "R49-"))]
    axes = {"tokens": [], "raw_gb": [], "wall_hours": [], "flops_proxy": []}
    endpoint = []
    source_bpt = source_bytes_per_token()
    for name in names:
        run = RESEARCH / "runs" / name
        if not (run / "evaluations.jsonl").exists() or not (run / "metrics.jsonl").exists():
            continue
        registration = next(row for row in registrations if row["experiment_id"] == name)
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").open()]
        evaluations = [json.loads(line) for line in (run / "evaluations.jsonl").open()]
        if not metrics or not evaluations:
            continue
        block_parameters = 0
        fixed_parameters = registration["active_parameter_count"]
        if registration.get("variant") == "progressive_depth":
            from latticelm.config import LatticeConfig
            from latticelm.posthack_variants import build_posthack_model
            cfg = LatticeConfig(**registration["config"])
            model = build_posthack_model(cfg, "progressive_depth")
            block_parameters = sum(parameter.numel() for parameter in model.blocks[0].parameters())
            fixed_parameters = registration["parameter_count"] - cfg.n_layers * block_parameters
            del model
        cumulative_bytes = 0.0
        cumulative_flops = 0.0
        previous_tokens = 0
        for metric in metrics:
            cumulative_bytes += sum(source_bpt.get(source, 0.0) * metric["context_length"]
                                    for source in metric.get("source_labels", []))
            metric["estimated_raw_bytes"] = cumulative_bytes
            active_parameters = (fixed_parameters + block_parameters * metric["active_depth"]
                                 if block_parameters else registration["active_parameter_count"])
            cumulative_flops += 6 * active_parameters * (metric["logical_tokens"] - previous_tokens)
            metric["estimated_flops_proxy"] = cumulative_flops
            previous_tokens = metric["logical_tokens"]
        by_token = {row["logical_tokens"]: row for row in metrics}
        x_tokens, x_bytes, x_hours, x_flops, y_loss = [], [], [], [], []
        for evaluation in evaluations:
            token = evaluation["logical_tokens"]
            metric = by_token.get(token)
            if metric is None:
                continue
            x_tokens.append(token / 1e6)
            x_bytes.append(metric["estimated_raw_bytes"] / 1e9)
            x_hours.append(metric["wall_seconds"] / 3600)
            x_flops.append(metric["estimated_flops_proxy"] / 1e15)
            mixture = json_file(ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json")["mixture_definition"]
            y_loss.append(sum(mixture[source] * evaluation["source_losses"][source] for source in mixture))
        if not x_tokens:
            continue
        for key, x in (("tokens", x_tokens), ("raw_gb", x_bytes),
                       ("wall_hours", x_hours), ("flops_proxy", x_flops)):
            axes[key].append((name, x, y_loss))
        complete = load_run(name)
        if complete:
            _, wiki, cap = complete
            four = sum(float(cap["results"][task]["acc,none"])
                       for task in ("hellaswag", "arc_easy", "piqa", "winogrande")) / 4
            endpoint.append((name, x_flops[-1], wiki["bits_per_byte"], four))
    written = []
    for axis, curves in axes.items():
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, x, y in curves:
            ax.plot(x, y, marker=".", markersize=3, label=name)
        ax.set_xlabel({"tokens": "Logical tokens (millions)",
                       "raw_gb": "Estimated UTF-8 bytes seen (billions)",
                       "wall_hours": "Training wall time (hours)",
                       "flops_proxy": "6NT active-parameter proxy (10^15)"}[axis])
        ax.set_ylabel("Frozen-mixture weighted held-out loss")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out / f"heldout_vs_{axis}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    if endpoint:
        fig, panels = plt.subplots(1, 2, figsize=(13, 5))
        for name, flop, bpb, acc in endpoint:
            panels[0].scatter(flop, bpb)
            panels[0].annotate(name, (flop, bpb), fontsize=6)
            panels[1].scatter(flop, acc)
            panels[1].annotate(name, (flop, acc), fontsize=6)
        panels[0].set(xlabel="6NT proxy (10^15)", ylabel="WikiText BPB")
        panels[1].set(xlabel="6NT proxy (10^15)", ylabel="Four-task raw accuracy")
        for ax in panels:
            ax.grid(alpha=.25)
        fig.tight_layout()
        path = out / "bpb_capability_vs_compute.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def calibration_report() -> Path:
    state = json.loads(STATE.read_text())
    rows = sorted(state.get("throughput_calibration", {}).values(),
                  key=lambda row: (row["regime"], row["architecture"], row["threads"]))
    lines = ["# Measured C4A throughput calibration", "",
             "Measurements use each registered configuration's scientific batch semantics (normally 32×256; context-512 uses 16×512), at least five warmup and twelve measured updates, on the 16-core Neoverse V2 C4A VM. Calibration calls are sequential and no second heavy trainer is active. Hours per 100M use measured post-warmup tokens per second. Different objectives and data samplers can add overhead; their live run TPS is separately recorded.", "",
             "| Regime | Architecture/variant | Threads | Parameters | Active parameters | Tokens/s | Hours/100M | Median step s | P10–P90 step s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['regime']} | {row['architecture']} | {row['threads']} | {row['parameter_count']:,} | {row.get('active_parameter_count', row['parameter_count']):,} | {row['tokens_per_second']:,.0f} | {row['hours_per_100m']:.2f} | {row['median_step_seconds']:.3f} | {row['p10_step_seconds']:.3f}–{row['p90_step_seconds']:.3f} |")
    lines += ["", "The historical architecture-tournament cloud measurements were taken on different hardware and are not scheduling floors. A larger microbatch was not substituted for the fixed scientific batch. The 49M regime appears only if the near-49M research trajectory cleared its evidence and time gate.", ""]
    path = RESEARCH / "THROUGHPUT_CALIBRATION.md"
    path.write_text("\n".join(lines))
    return path


def synthesis() -> Path:
    state = json.loads(STATE.read_text())
    master = json_file(RESEARCH / "master_state.json") or {}
    matched = json_file(RESEARCH / "matched_co4_dense_analysis.json")
    promotion = json_file(RESEARCH / "promotion_32m_results.json")
    tokenizer = json_file(RESEARCH / "retokenized_subset/comparison.json")
    expert = json_file(RESEARCH / "developmental_experts_analysis.json")
    posttrain = json_file(RESEARCH / "posttraining/interpolation/report.json")
    lines = ["# Final post-hackathon research synthesis", "",
             f"Prepared {utc()}. The working window began {state['started_utc']} and ends {state['working_deadline_utc']}; the last 12 hours are reserved for evaluation and Hugging Face export. The user's one-day VM lifetime gap remains outside this 15-day working window.", "",
             "The frozen 48.6M hackathon model and its 1.9B-token result are historical artifacts. Post-hackathon model IDs below represent new research trajectories. The research workspace is Git-free; source changes are archived only to the private Hugging Face research repository.", "",
             "## Campaign status", "",
             "| Phase | Status |", "|---|---|"]
    for name, status in (master.get("phase_status") or {}).items():
        lines.append(f"| {name} | {status} |")
    lines += ["", "## Answers to the registered questions", ""]
    if matched:
        lines.append("**Modern Co4 versus dense:**")
        for row in matched["comparisons"]:
            interval = row["paired_window_bootstrap"].get("ci95")
            supported = ("Co4 lower held-out loss" if interval and interval[1] < 0 else
                         "dense lower held-out loss" if interval and interval[0] > 0 else
                         "held-out direction unresolved")
            lines.append(f"- {row['pair']}: {supported}; Co4 − dense loss {row['heldout_loss_delta_co4_minus_dense']:+.4f}, BPB {row['wikitext_bpb_delta_co4_minus_dense']:+.4f}, wall ratio {row['training_wall_ratio_co4_over_dense']:.3f}, paired-window interval {interval}, paired four-task accuracy interval {row.get('paired_capability', {}).get('four_task_equal_weight_ci95')}. A per-token advantage requires the loss and BPB signs to agree; wall advantage also requires comparable quality at no greater wall time.")
        replication = json_file(RESEARCH / "matched_16m_second_seed_decision.json")
        lines.append("The initial matched study has two 8M seeds and one each at 16M and 32M. "
                     + ("A second 16M Co4/dense seed was subsequently run; see SECOND_SEED_REPLICATION.md. "
                        if replication and replication["run"] and
                        (RESEARCH / "runs/A16-DENSE-S2026/hf_archive_receipt.json").exists() else "")
                     + "The 6NT proxy omits Co4 operator overhead, so parameter matching alone does not establish FLOP efficiency.")
    else:
        lines.append("**Modern Co4 versus dense:** the matched analysis was not completed; no winner is asserted.")
    decisions = [row for row in state.get("decisions", []) if row.get("mechanism")]
    best_data = sorted((row for row in decisions if str(row.get("experiment_id", "")).startswith("D")),
                       key=lambda row: row.get("bpb_relative_change", math.inf))
    lines.append("\n**Data composition, selection, curriculum and late annealing:** " +
                 (f"The strongest provisional BPB screen was {best_data[0]['experiment_id']} "
                  f"({best_data[0]['bpb_relative_change']:+.2%} versus its control). "
                  "See source-wise losses and broad capability before treating this as a supported recipe."
                  if best_data else "no complete D-track comparison was available."))
    lines.append("\n**Tokenizer and context:** " +
                 (f"The common-subset 4K/8K comparison measured {tokenizer['4k_bpt_on_common_raw_subset']:.3f} "
                  f"versus {tokenizer['8k_bpt_on_common_raw_subset']:.3f} weighted raw bytes/token. "
                  f"At 150M tokenizer tokens, WikiText BPB was {tokenizer['endpoint_wikitext_bpb']['4096']:.4f} "
                  f"versus {tokenizer['endpoint_wikitext_bpb']['8192']:.4f}; "
                  f"the byte-matched 8K checkpoint BPB was {tokenizer['8k_byte_matched_wikitext_bpb']:.4f}. "
                  f"The 16K screen was {'run' if tokenizer['16k_training_decision']['run_16k'] else 'gated out'} by its frozen evidence/time decision. "
                  "This subset cannot by itself explain the external systems' quality gap."
                  if tokenizer else "the matched 4K/8K training comparison was not completed."))
    prompt_audit = json_file(RESEARCH / "prompt_truncation_audit.json")
    if prompt_audit:
        lines.append(f"The full eight-task prompt audit found a maximum {prompt_audit['max_task_meaningful_truncation_fraction_at_256']:.2%} of requests meaningfully truncated at context 256; the context-512 decision is in CONTEXT_512.md.")
    else:
        lines.append("The full eight-task prompt/truncation audit was not completed.")
    lines.append("\n**Value residual:** " +
                 next((f"{row['classification']}: held-out change {row['heldout_relative_change']:+.2%}, BPB change {row['bpb_relative_change']:+.2%}."
                       for row in decisions if row.get("experiment_id") == "B16-VALUE-RESIDUAL-S1337"),
                      "the 16M screen was not completed."))
    lines.append("\n**Curriculum and capacity growth:** " +
                 ("The registered 2×2 interaction and active-layer FLOP proxy are in CGLS_CO4.md."
                  if (RESEARCH / "CGLS_CO4.md").exists() else "the 2×2 design was not completed."))
    d3 = next((row for row in decisions if row.get("experiment_id") == "D3-16M-CURRICULUM"), None)
    if d3:
        lines.append(f"D3 curriculum's endpoint held-out change was {d3['heldout_relative_change']:+.2%} and BPB change was {d3['bpb_relative_change']:+.2%}; early-versus-late sampled difficulty is recorded in DATA_CURRICULUM.md. A convergence claim requires the token curves, not only this endpoint.")
    if (RESEARCH / "runs/E2-16M-GROWTH-CURRICULUM/hf_archive_receipt.json").exists():
        from posthack_phase3 import weighted_loss, competition_accuracy
        ids = ("A16-CO4-S1337", "D3-16M-CURRICULUM", "E1-16M-GROWTH-STATIC",
               "E2-16M-GROWTH-CURRICULUM")
        values = {name: weighted_loss(name) for name in ids}
        accuracy = {name: competition_accuracy(name) for name in ids}
        loss_interaction = values[ids[3]] - values[ids[2]] - values[ids[1]] + values[ids[0]]
        cap_interaction = accuracy[ids[3]] - accuracy[ids[2]] - accuracy[ids[1]] + accuracy[ids[0]]
        lines.append(f"The E2−E1−D3+D0 interaction was {loss_interaction:+.5f} held-out loss and {cap_interaction:+.4f} four-task raw accuracy; lower loss and higher accuracy indicate favorable interaction. The active-layer FLOP and wall comparisons are in CGLS_CO4.md.")
    lines.append("\n**Developmental experts:** " +
                 ("F0 fixed, F1 naive birth, F2 cloning and F3 clone plus catch-up completed. "
                  "The expert report gives quality, utilization, router entropy and domain/expert mutual information; "
                  "an MI difference alone is not evidence of a useful specialist."
                  if expert else "the four-arm birth comparison was not completed."))
    if expert:
        expert_rows = {row["name"]: row for row in expert}
        fixed = expert_rows["F0-8M-FIXED-EXPERTS"]
        random_birth = expert_rows["F1-8M-NAIVE-BIRTH"]
        catchup = expert_rows["F3-8M-CLONE-CATCHUP"]
        beats_both = (catchup["heldout_loss"] < min(fixed["heldout_loss"], random_birth["heldout_loss"])
                      and catchup["wiki_bpb"] < min(fixed["wiki_bpb"], random_birth["wiki_bpb"]))
        lines.append(f"F3 {'beat' if beats_both else 'did not beat'} both fixed and random-birth controls on held-out loss and BPB: F3 {catchup['heldout_loss']:.5f}/{catchup['wiki_bpb']:.5f}, F0 {fixed['heldout_loss']:.5f}/{fixed['wiki_bpb']:.5f}, F1 {random_birth['heldout_loss']:.5f}/{random_birth['wiki_bpb']:.5f}. F3 domain/expert MI was {catchup['domain_expert_mi_nats']:.4f} nats; attribution and endpoint diagnostics determine whether the quality result reflects specialization.")
    lines.append("\n**PHM-4:** " +
                 ("The single 16M FFN-only screen is reported in PHM4_SCREEN.md; parameter saving is not interpreted as arithmetic saving."
                  if (RESEARCH / "PHM4_SCREEN.md").exists() else "the bounded screen was not completed."))
    if (RESEARCH / "runs/G-16M-PHM4-FFN/hf_archive_receipt.json").exists():
        from posthack_phase5 import endpoint
        base, candidate = endpoint("A16-CO4-S1337"), endpoint("G-16M-PHM4-FFN")
        base_wall = json_file(RESEARCH / "runs/A16-CO4-S1337/result.json")["wall_seconds"]
        new_wall = json_file(RESEARCH / "runs/G-16M-PHM4-FFN/result.json")["wall_seconds"]
        justified = candidate[1] <= base[1] * .99 and new_wall <= base_wall * 1.1
        lines.append(f"PHM-4 {'cleared' if justified else 'did not clear'} the bounded quality-per-wall signal: BPB change {candidate[1]/base[1]-1:+.2%}, wall-time change {new_wall/base_wall-1:+.2%}. Parameter matching alone is insufficient.")
    lines.append("\n**Retention-aware post-training:** " +
                 (f"P0/P1/P2 and interpolation were evaluated. The largest P1 fraction under the retention gate was {posttrain['selected']['alpha']:.2f}. P2 is a task-tuned diagnostic upper bound."
                  if posttrain else "the retention/interpolation screens were not completed."))
    selection = json_file(RESEARCH / "posttraining/parent_selection.json")
    if selection and (RESEARCH / "runs/P1-16M-ALIGNED-REPLAY-KL/hf_archive_receipt.json").exists():
        from posthack_phase5 import endpoint
        base, candidate = endpoint(selection["experiment_id"]), endpoint("P1-16M-ALIGNED-REPLAY-KL")
        skill_base = json_file(RESEARCH / "posttraining/parent_skill_validation.json")
        skill_new = json_file(RESEARCH / "runs/P1-16M-ALIGNED-REPLAY-KL/heldout_skill_validation.json")
        skill_change = (skill_new["aggregate_nll_nats_per_completion_token"] -
                        skill_base["aggregate_nll_nats_per_completion_token"])
        lines.append(f"P1 versus its 16M parent changed four-task raw accuracy by {candidate[2]-base[2]:+.4f}, WikiText BPB by {candidate[1]/base[1]-1:+.2%}, and disjoint held-out skill NLL/token by {skill_change:+.5f}; source-wise retention and paired capability intervals govern transfer claims. The approximately on-policy and full-size decisions are separately recorded.")
    lines.append("\n**32M promotion:** " +
                 (f"The tournament included {len(promotion['rows'])} fresh 300M-token arms. See PROMOTION_32M.md for source-wise, BPB and capability evidence."
                  if promotion else "the fresh 32M tournament was not completed."))
    lines.append("\n**Near-49M trajectory:** " +
                 ("See NEXT_49M_TRAJECTORY.md. A 300–500M-token trajectory does not establish a 1.9B-token outcome."
                  if (RESEARCH / "NEXT_49M_TRAJECTORY.md").exists() else "the evidence/time gate did not yield a completed report."))
    for label, filename in (("nCo4", "STRETCH_NCO4.md"), ("Differential-Co4", "STRETCH_DIFFCO4.md")):
        lines.append(f"\n**{label}:** " + (f"See {filename} for the engineering/scientific gate."
                                            if (RESEARCH / filename).exists() else "not attempted before final reserve."))
    if (RESEARCH / "runs/S1-8M-NCO4/hf_archive_receipt.json").exists():
        from posthack_phase5 import endpoint
        base, candidate = endpoint("S1-8M-ADAMW-CONTROL"), endpoint("S1-8M-NCO4")
        wall_base = json_file(RESEARCH / "runs/S1-8M-ADAMW-CONTROL/result.json")["wall_seconds"]
        wall_new = json_file(RESEARCH / "runs/S1-8M-NCO4/result.json")["wall_seconds"]
        signal = candidate[1] <= base[1] * .99 and wall_new <= wall_base * 1.1
        lines.append(f"nCo4 {'cleared' if signal else 'did not clear'} a bounded quality-per-wall pilot signal: BPB {candidate[1]/base[1]-1:+.2%}, wall time {wall_new/wall_base-1:+.2%} versus its matched AdamW control. A positive single pilot still requires replication.")
    elif (RESEARCH / "STRETCH_NCO4.md").exists():
        lines.append("nCo4 did not produce a completed pilot before its engineering or time gate; no scientific quality verdict is inferred.")
    if (RESEARCH / "runs/S2-8M-DIFFCO4/hf_archive_receipt.json").exists():
        from posthack_phase5 import endpoint
        from posthack_phase7 import baseline_50m
        base_loss, base_bpb = baseline_50m()
        candidate = endpoint("S2-8M-DIFFCO4")
        speed_base = state["throughput_calibration"]["8m/co4/16"]["tokens_per_second"]
        speed_diff = state["throughput_calibration"]["8m/differential_co4/16"]["tokens_per_second"]
        signal = candidate[1] <= base_bpb * .99 and speed_diff >= speed_base * .8
        lines.append(f"Differential-Co4 {'cleared' if signal else 'did not clear'} a bounded quality/CPU-cost pilot signal: BPB {candidate[1]/base_bpb-1:+.2%}, calibrated throughput {speed_diff/speed_base-1:+.2%} versus Co4. The attention operator profile and held-out loss comparison are in STRETCH_DIFFCO4.md.")
    elif (RESEARCH / "STRETCH_DIFFCO4.md").exists():
        lines.append("Differential-Co4 did not produce a completed pilot before its engineering or time gate; no scientific quality verdict is inferred.")
    recommendations = []
    if promotion:
        for name, row in promotion["rows"].items():
            if name == "R32-CO4-CONTROL":
                continue
            if (row["heldout_relative_to_control"] <= -.01 or
                row["bpb_relative_to_control"] <= -.01) and row["accuracy_absolute_to_control"] >= -.01:
                recommendations.append((row["bpb_relative_to_control"] + row["heldout_relative_to_control"],
                                        row["parent_16m"], name))
    recommendations.sort()
    lines += ["", "## Future allocation", ""]
    if recommendations:
        lines.append("The strongest supported 32M promotion signals for future GPU confirmation are " +
                     ", ".join(f"`{parent}` (32M arm `{name}`)" for _, parent, name in recommendations[:2]) +
                     ". These are research priorities, not production claims; repeat-seed and larger-corpus confirmation remain necessary.")
    else:
        lines.append("No 32M candidate cleared the registered quality and broad-capability guard. No substantial GPU allocation is justified by this campaign alone; retain the best control and investigate the most informative unresolved mechanism with a better-powered design.")
    negatives = [row["experiment_id"] for row in decisions
                 if row.get("classification") == "PROVISIONAL_NEGATIVE"]
    if negatives:
        lines.append("Discontinue or redesign the provisionally negative branches " +
                     ", ".join(f"`{name}`" for name in negatives) +
                     " before spending larger compute. A single negative screen is a stop for this campaign, not a universal architecture verdict.")
    else:
        lines.append("No branch is labeled a definitive scientific failure from the current screening evidence; gated-out stretches are time decisions, not negative results.")
    lines += ["",
              "## Limits", "",
              "Capability evaluations are fixed 1,000-example screens unless a run explicitly records a fuller evaluation. Paired validation-window bootstrap does not replace model-seed uncertainty. Raw bytes in DATA-D training curves are source-ratio estimates. The 6NT proxy omits attention and optimizer overhead; wall-clock throughput is reported separately. Missing phase evidence is labeled directly above.", ""]
    path = RESEARCH / "FINAL_RESEARCH_SYNTHESIS.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    manifest, source = source_snapshot()
    calibration_report()
    picture_paths = plots()
    synthesis_path = synthesis()
    mandatory = ("CAMPAIGN_PLAN.md", "LIVE_SCHEDULE.md", "RUN_REGISTRY.jsonl",
                 "THROUGHPUT_CALIBRATION.md", "MATCHED_CO4_DENSE.md",
                 "VALUE_RESIDUAL.md", "TOKENIZER_CONTEXT.md", "DATA_CURRICULUM.md",
                 "CGLS_CO4.md", "DEVELOPMENTAL_EXPERTS.md", "PHM4_SCREEN.md",
                 "POSTTRAINING_TRANSFER.md", "PROMOTION_32M.md",
                 "FINAL_RESEARCH_SYNTHESIS.md")
    missing = [name for name in mandatory if not (RESEARCH / name).exists()]
    report = {"schema": "posthack-final-archive-v1", "created_utc": utc(),
              "mandatory_reports_missing": missing,
              "source_manifest_sha256": sha256(manifest),
              "source_archive_sha256": sha256(source),
              "plots": {path.name: sha256(path) for path in picture_paths},
              "working_deadline_utc": json.loads(STATE.read_text())["working_deadline_utc"],
              "git_policy": "no commit, no Git/GitHub push; only private Hugging Face archive"}
    path = RESEARCH / "FINAL_ARCHIVE_MANIFEST.json"
    atomic_json(path, report)
    archive = [manifest, source, synthesis_path, path, *picture_paths]
    archive.extend(RESEARCH / name for name in mandatory if (RESEARCH / name).exists() and name != "FINAL_RESEARCH_SYNTHESIS.md")
    archive.extend(RESEARCH / name for name in (
        "MASTER_CONTROLLER.md", "GIT_ISOLATION.md", "IMPLEMENTATION_AUDIT.md", "8M_EXTENSION.md", "CONTEXT_512.md", "SECOND_SEED_REPLICATION.md",
        "NEXT_49M_TRAJECTORY.md", "STRETCH_NCO4.md",
        "STRETCH_DIFFCO4.md", "OVERFLOW_OPTIONS.md") if (RESEARCH / name).exists())
    archive.extend(path for path in (
        RESEARCH / "8m_extension_decision.json",
        RESEARCH / "context512_decision.json",
        RESEARCH / "d5_combination_decision.json",
        RESEARCH / "matched_16m_second_seed_decision.json",
        RESEARCH / "mechanism_second_seed_decision.json",
        RESEARCH / "49m_budget_decision.json",
        RESEARCH / "s1-nco4_time_decision.json",
        RESEARCH / "s2-diffco4_time_decision.json",
        RESEARCH / "posttraining/onpolicy_decision.json",
        RESEARCH / "posttraining/fullsize_decision.json",
        RESEARCH / "retokenized_subset/16k_decision.json",
        RESEARCH / "value_residual_paired_capability.json",
        RESEARCH / "differential_attention_profile.json",
    ) if path.exists())
    archive.extend(path for path in RESEARCH.glob("overflow_*_decision.json") if path.exists())
    archive.extend(path for path in (RESEARCH / "controller_state.json",
                                     RESEARCH / "controller_events.jsonl",
                                     RESEARCH / "master_state.json",
                                     RESEARCH / "matched_co4_dense_analysis.json",
                                     RESEARCH / "promotion_32m_results.json",
                                     RESEARCH / "promotion_selection.json") if path.exists())
    upload(archive, "FINAL-RESEARCH-ARCHIVE")
    event("FINAL", "FINAL_ARCHIVE_COMPLETE", files=len(archive), missing_reports=missing,
          synthesis_sha256=sha256(synthesis_path))


if __name__ == "__main__":
    main()
