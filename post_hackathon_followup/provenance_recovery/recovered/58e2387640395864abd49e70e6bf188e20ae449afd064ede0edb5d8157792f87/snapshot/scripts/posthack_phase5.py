"""Retention-aware post-training screens on the strongest suitable 16M parent."""
from __future__ import annotations

import json
import sys

import torch

from latticelm.config import LatticeConfig
from latticelm.posthack_variants import build_posthack_model
from posthack_archive import upload
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, can_launch, event, register, sha256, utc
from posthack_phase1 import invoke
from posthack_single_run import TASKS, registered_ids
from posthack_train import valid_checkpoint
from posthack_uncertainty import paired_capability

MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
DATA = RESEARCH / "posttraining/data"
ARMS = (
    ("P0-16M-GENERIC-REPLAY", "generic_replay", "generic_train.jsonl", 0.0),
    ("P1-16M-ALIGNED-REPLAY-KL", "aligned_replay_kl", "aligned_train.jsonl", .10),
    ("P2-16M-TASK-TUNED-DIAGNOSTIC", "task_tuned_replay", "task_train.jsonl", 0.0),
)
ONPOLICY_ARM = ("P3-16M-ONPOLICY-REPLAY", "onpolicy_replay", "aligned_train.jsonl", .10)
FULLSIZE_ARM = ("P4-49M-ALIGNED-REPLAY-KL", "aligned_replay_kl", "aligned_train.jsonl", .10)
FROZEN = ROOT / "artifacts/final_production_run/final.pt"


def endpoint(name: str) -> tuple[float, float, float, dict]:
    run = RESEARCH / "runs" / name
    validation = [json.loads(line) for line in (run / "evaluations.jsonl").open()][-1]
    wiki = json.loads((run / "wikitext103.json").read_text())
    capability = json.loads((run / "capability_screen.json").read_text())
    aggregate = sum(float(capability["results"][task]["acc,none"])
                    for task in ("hellaswag", "arc_easy", "piqa", "winogrande")) / 4
    mixture = json.loads(MANIFEST.read_text())["mixture_definition"]
    heldout = sum(mixture[source] * validation["source_losses"][source]
                  for source in mixture)
    return heldout, wiki["bits_per_byte"], aggregate, validation["source_losses"]


def choose_parent() -> dict:
    candidates = ["A16-CO4-S1337", "B16-VALUE-RESIDUAL-S1337",
                  "D1-16M-REBALANCED-S1337", "D2-16M-CAPABILITY",
                  "D3-16M-CURRICULUM", "D4-16M-ANNEAL",
                  "D5-16M-COMBINED", "E1-16M-GROWTH-STATIC",
                  "E2-16M-GROWTH-CURRICULUM", "G-16M-PHM4-FFN"]
    control = endpoint("A16-CO4-S1337")
    eligible = []
    for name in candidates:
        run = RESEARCH / "runs" / name
        if not (run / "hf_archive_receipt.json").exists():
            continue
        heldout, bpb, capability, sources = endpoint(name)
        if capability < control[2] - .01:
            continue
        if any(sources[source] > control[3][source] * 1.05 for source in sources):
            continue
        eligible.append((bpb, heldout, name))
    if not eligible:
        eligible = [(control[1], control[0], "A16-CO4-S1337")]
    _, _, name = min(eligible)
    checkpoint = RESEARCH / "runs" / name / "latest.pt"
    result = {"experiment_id": name, "checkpoint_path": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint),
              "selection_rule": "minimum WikiText BPB among 16M 4K models with <=1 point four-task raw capability regression and <=5% source-wise held-out loss regression versus A16 Co4",
              "eligible": [row[2] for row in sorted(eligible)]}
    atomic_json(RESEARCH / "posttraining/parent_selection.json", result)
    return result


def evaluate_skills(checkpoint: Path, output: Path, log: Path) -> None:
    if output.exists():
        return
    invoke([sys.executable, "scripts/posthack_posttrain_eval.py",
            "--checkpoint", str(checkpoint), "--output", str(output),
            "--threads", "4"], log, "held-out skill completion")


def register_arm(arm: tuple, parent: dict, *, budget: int = 1_000_000,
                 throughput_tps: float | None = None) -> None:
    name, objective, data_file, kl_weight = arm
    if name in registered_ids():
        return
    checkpoint = torch.load(ROOT / parent["checkpoint_path"], map_location="cpu", weights_only=False)
    cfg = LatticeConfig(**checkpoint["config"])
    variant = checkpoint.get("variant", "baseline")
    model = build_posthack_model(cfg, variant)
    parameters = sum(p.numel() for p in model.parameters())
    state = json.loads(STATE.read_text())
    base_tps = state["throughput_calibration"]["16m/co4/16"]["tokens_per_second"]
    conservative_tps = throughput_tps or base_tps * (.12 if objective == "onpolicy_replay" else
                                                    .25 if kl_weight else .35)
    data_path = DATA / data_file
    registration = {"experiment_id": name,
                    "hypothesis": {
                        "generic_replay": "Generic SFT with 50% natural replay gains transfer while retaining LM skill",
                        "aligned_replay_kl": "Reference KL on natural replay limits specialization damage without suppressing broad gains",
                        "task_tuned_replay": "Official task training splits provide a diagnostic upper bound on directly learnable benchmark skill",
                        "onpolicy_replay": "Current-model sampled replay continuations improve transfer or retention beyond static natural replay"}[objective],
                    "parent_control": parent["experiment_id"],
                    "config": cfg.to_dict(), "variant": variant,
                    "tokenizer_sha256": sha256(TOKENIZER),
                    "sampling_manifest_sha256": sha256(MANIFEST),
                    "code_commit": state["source_commit"],
                    "source_file_sha256": {"scripts/posthack_posttrain.py": sha256(ROOT / "scripts/posthack_posttrain.py")},
                    "seed": 915123, "parameter_count": parameters,
                    "active_parameter_count": model.active_parameter_count() if hasattr(model, "active_parameter_count") else parameters,
                    "context_length": cfg.context_length,
                    "optimizer_partition": {key: "adamw" for key, _ in model.named_parameters()},
                    "optimizer_policy": "fresh AdamW, constant LR 3e-5, 50% SFT/replay objective weighting",
                    "learning_rate": 3e-5, "token_budget": budget,
                    "planned_evaluation_points": [budget // 4, budget // 2, 3 * budget // 4, budget],
                    "promotion_criterion": "Broad held-out gain with preferably <2% LM/BPB retention loss and no source collapse; P2 diagnostic only",
                    "abort_criterion": "Nonfinite update, severe retention loss, or archival reserve",
                    "expected_wall_seconds": budget / conservative_tps + 7200,
                    "estimated_throughput_tps": conservative_tps,
                    "estimate_provenance": "16m/co4/16 C4A calibration discounted for short SFT batches and optional teacher forward",
                    "parent_checkpoint_path": parent["checkpoint_path"],
                    "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                    "posttraining_data_path": str(data_path.relative_to(ROOT)),
                    "posttraining_data_sha256": sha256(data_path),
                    "posttraining_data_order": "seeded_resource_round_robin_v1",
                    "posttraining_completion_policy": "skip answers plus EOS exceeding the configured context window; left-truncate overlong prompts",
                    "posttraining_objective": objective,
                    "replay_fraction": .5, "reference_kl_weight": kl_weight,
                    "objective_equation": "L=0.5*CE_completion+0.5*CE_natural+0.5*lambda*KL(p_ref||p_theta) on natural replay tokens",
                    "best_checkpoint_metric": "minimum frozen-mixture weighted held-out source loss at a registered checkpoint",
                    "comparison_label": "task-tuned diagnostic upper bound, not pretraining-only" if objective == "task_tuned_replay" else "retention-aware posttraining"}
    if objective == "onpolicy_replay":
        registration.update({"onpolicy_replay_fraction": .25,
                             "onpolicy_continuation_tokens": 8,
                             "onpolicy_policy": "sample eight autoregressive continuation tokens plus final target from current model on four of sixteen natural replay examples per step; remaining replay stays static"})
    register(registration)


def decide_onpolicy() -> dict:
    path = RESEARCH / "posttraining/onpolicy_decision.json"
    if path.exists():
        return json.loads(path.read_text())
    state = json.loads(STATE.read_text())
    tps = state["throughput_calibration"]["16m/co4/16"]["tokens_per_second"] * .12
    fits, finish = can_launch(1_000_000, tps, 7200, 1200, 3600)
    result = {"created_utc": utc(), "run": fits, "estimated_tps": tps,
              "predicted_finish_utc": utc(finish),
              "criterion": "bounded current-model generation on 25% of replay examples; launch only if predicted endpoint and evaluation fit archival reserve",
              "reason": "time gate passed" if fits else "predicted finish crosses archival reserve"}
    atomic_json(path, result)
    event(ONPOLICY_ARM[0], "OPTIONAL_RUN_DECISION", **result)
    return result


def decide_fullsize(parent: dict) -> dict:
    path = RESEARCH / "posttraining/fullsize_decision.json"
    if path.exists():
        return json.loads(path.read_text())
    baseline = endpoint(parent["experiment_id"])
    candidate = endpoint(ARMS[1][0])
    capability_gain = candidate[2] - baseline[2]
    paired = paired_capability(RESEARCH / "runs" / ARMS[1][0] / "capability_samples.json",
                               RESEARCH / "runs" / parent["experiment_id"] / "capability_samples.json")
    bpb_change = candidate[1] / baseline[1] - 1
    worst_source = max(candidate[3][source] / baseline[3][source] - 1 for source in baseline[3])
    evidence = (capability_gain >= .01 and paired["four_task_equal_weight_ci95"][0] > 0
                and bpb_change <= .02 and worst_source <= .05)
    state = json.loads(STATE.read_text())
    calibration = state["throughput_calibration"].get("49m/co4/16")
    tps = (calibration["tokens_per_second"] if calibration else 2800) * .20
    fits, finish = can_launch(500_000, tps, 7200, 1800, 3600)
    result = {"created_utc": utc(), "run": bool(evidence and fits),
              "capability_gain": capability_gain, "bpb_relative_change": bpb_change,
              "paired_capability": paired,
              "worst_source_relative_change": worst_source,
              "estimated_tps": tps, "predicted_finish_utc": utc(finish),
              "criterion": "P1 gains >=1 point four-task raw accuracy with positive paired item-bootstrap lower bound, <=2% BPB and <=5% worst-source loss damage; bounded 49M child must fit reserve",
              "reason": ("evidence and time gate passed" if evidence and fits else
                         "16M retention/transfer evidence insufficient" if not evidence else
                         "49M confirmation would cross archival reserve")}
    atomic_json(path, result)
    event(FULLSIZE_ARM[0], "OPTIONAL_RUN_DECISION", **result)
    return result


def evaluate_frozen_parent() -> None:
    out = RESEARCH / "posttraining/fullsize_parent"
    out.mkdir(parents=True, exist_ok=True)
    wiki = out / "wikitext103.json"
    capability = out / "capability_screen.json"
    samples = out / "capability_samples.json"
    validation = out / "source_validation.json"
    skills = out / "heldout_skill_validation.json"
    if not wiki.exists():
        invoke([sys.executable, "scripts/evaluate_wikitext103.py", "--checkpoint", str(FROZEN),
                "--tokenizer", str(TOKENIZER), "--output", str(wiki), "--threads", "4"],
               out / "evaluate.log", "frozen BASE WikiText")
    if not capability.exists():
        invoke([sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", str(FROZEN),
                "--tokenizer", str(TOKENIZER), "--output", str(capability),
                "--tasks", TASKS, "--limit", "1000", "--threads", "4",
                "--predictions-output", str(samples)], out / "evaluate.log", "frozen BASE capability")
    if not validation.exists():
        from train_final_production import load_arrays
        from posthack_train import evaluate
        checkpoint = torch.load(FROZEN, map_location="cpu", weights_only=False)
        cfg = LatticeConfig(**checkpoint["config"])
        model = build_posthack_model(cfg)
        model.load_state_dict(checkpoint["model"])
        torch.set_num_threads(4)
        _, _, valid = load_arrays(MANIFEST)
        losses, paired = evaluate(model, valid, cfg.context_length)
        atomic_json(validation, {"source_losses": losses, "paired_losses": paired,
                                 "checkpoint_sha256": sha256(FROZEN), "evaluated_utc": utc()})
    evaluate_skills(FROZEN, skills, out / "evaluate.log")
    upload([wiki, capability, samples, validation, skills], "P4-FROZEN-PARENT-EVAL")


def run_arm(arm: tuple) -> None:
    name = arm[0]
    run = RESEARCH / "runs" / name
    receipt = run / "hf_archive_receipt.json"
    if receipt.exists():
        return
    record = next(json.loads(line) for line in (RESEARCH / "RUN_REGISTRY.jsonl").open()
                  if json.loads(line)["experiment_id"] == name)
    result = run / "result.json"
    completed = result.exists() and json.loads(result.read_text())["status"] == "COMPLETED"
    if not completed:
        previous_tokens = 0
        resumable = next((path for path in (run / "latest.pt", run / "previous.pt")
                          if valid_checkpoint(path)), None)
        if resumable is not None:
            previous_tokens = int(torch.load(resumable, map_location="cpu", weights_only=False)["logical_tokens"])
        allowed, predicted = can_launch(max(0, record["token_budget"] - previous_tokens),
                                        record["estimated_throughput_tps"],
                                        7200, 1200, 3600)
        if not allowed:
            event(name, "POSTTRAIN_DEFERRED", predicted_finish_utc=utc(predicted),
                  reason="archival reserve")
            raise RuntimeError("posttraining screen does not fit reserve")
        invoke([sys.executable, "scripts/posthack_posttrain.py", "--experiment-id", name,
                "--threads", "12"], run / "train.log", "posttraining")
    if json.loads(result.read_text())["status"] != "COMPLETED":
        raise RuntimeError(f"posttraining arm stopped before registered endpoint: {name}")
    checkpoint = run / "latest.pt"
    if not (run / "wikitext103.json").exists():
        invoke([sys.executable, "scripts/evaluate_wikitext103.py", "--checkpoint", str(checkpoint),
                "--tokenizer", str(TOKENIZER), "--output", str(run / "wikitext103.json"),
                "--threads", "4"], run / "evaluate.log", "WikiText retention")
    if not (run / "capability_screen.json").exists():
        invoke([sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", str(checkpoint),
                "--tokenizer", str(TOKENIZER), "--output", str(run / "capability_screen.json"),
                "--tasks", TASKS, "--limit", "1000", "--threads", "4",
                "--predictions-output", str(run / "capability_samples.json")],
               run / "evaluate.log", "broad capability screen")
    evaluate_skills(checkpoint, run / "heldout_skill_validation.json",
                    run / "evaluate.log")
    files = [result, run / "metrics.jsonl", run / "evaluations.jsonl",
             run / "wikitext103.json", run / "capability_screen.json",
             run / "capability_samples.json", run / "heldout_skill_validation.json",
             checkpoint]
    if valid_checkpoint(run / "best.pt"):
        files += [run / "best.pt", run / "best.sha256"]
    atomic_json(receipt, {"at": utc(), "files": upload(files, name)})
    state = json.loads(STATE.read_text())
    state["queued_runs"] = [item for item in state["queued_runs"] if item != name]
    atomic_json(STATE, state)


def report(parent: dict) -> None:
    names = [parent["experiment_id"]] + [arm[0] for arm in ARMS]
    if (RESEARCH / "runs" / ONPOLICY_ARM[0] / "hf_archive_receipt.json").exists():
        names.append(ONPOLICY_ARM[0])
    scores = {name: endpoint(name) for name in names}
    baseline = scores[parent["experiment_id"]]
    lines = ["# Retention-aware post-training transfer", "",
             f"The parent is {parent['experiment_id']} (SHA-256 `{parent['checkpoint_sha256']}`). P0 trains on disjoint generic SciQ/QASC reasoning QA with 50% natural DATA-D replay. P1 adds CommonsenseQA training examples for broad capability alignment and a frozen-parent reference KL on the same replay. P2 uses official training splits of the four competition tasks, with exact held-out-prompt hash exclusion; it is a **task-tuned diagnostic upper bound** and is not comparable to pretraining-only systems. All screens use a fresh AdamW optimizer, 3e-5 LR, one million logical target tokens, and the 4K tokenizer.", "",
             "The implemented objective is `0.5 CE_completion + 0.5 CE_natural + 0.5 λ KL(p_ref || p_θ)`; λ is 0 for P0/P2 and 0.10 for P1/P3/P4. KL is evaluated only on natural replay tokens. Completion loss masks prompt and padding tokens. Registered SFT data order shuffles each pinned training source with a fixed seed, then interleaves sources round-robin so a one-million-token screen sees every included skill rather than only the first dataset in the file.", "",
             "| Arm | Held-out loss | WikiText BPB | Four-task raw accuracy | Held-out skill NLL/token | BPB change vs parent |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        heldout, bpb, accuracy, _ = scores[name]
        skill_path = (RESEARCH / "posttraining/parent_skill_validation.json" if name == parent["experiment_id"]
                      else RESEARCH / "runs" / name / "heldout_skill_validation.json")
        skill = json.loads(skill_path.read_text())["aggregate_nll_nats_per_completion_token"]
        lines.append(f"| {name} | {heldout:.5f} | {bpb:.5f} | {accuracy:.4f} | {skill:.5f} | {bpb / baseline[1] - 1:+.2%} |")
    lines += ["", "P2 is labeled task-tuned in the registry, data manifest, checkpoint lineage, and this report. A small raw-accuracy change is not a supported broad gain without paired uncertainty analysis. Weight interpolation is evaluated before any recovery training.", ""]
    onpolicy = json.loads((RESEARCH / "posttraining/onpolicy_decision.json").read_text())
    lines.append(f"The approximately on-policy replay arm was {'run' if onpolicy['run'] else 'gated out'}: {onpolicy['reason']}. When run, four of sixteen natural replay examples per step have eight continuation tokens sampled autoregressively from the current model; generation input tokens are counted separately in `processed_tokens`.")
    interpolation = RESEARCH / "posttraining/interpolation/report.json"
    if interpolation.exists():
        selected = json.loads(interpolation.read_text())["selected"]
        lines.extend([f"Interpolation retained P1 fraction `{selected['alpha']:.2f}` under the pre-registered <=2% WikiText BPB and <=5% source-loss retention gates. The complete interpolation frontier and selected capability screen are in `posttraining/interpolation/report.json`.", ""])
    fullsize = json.loads((RESEARCH / "posttraining/fullsize_decision.json").read_text())
    lines.extend(["", "## Frozen BASE confirmation gate", "",
                  f"The bounded 49M child was {'run' if fullsize['run'] else 'gated out'}: {fullsize['reason']}. The frozen BASE checkpoint itself was not modified."])
    if (RESEARCH / "runs" / FULLSIZE_ARM[0] / "hf_archive_receipt.json").exists():
        child = endpoint(FULLSIZE_ARM[0])
        out = RESEARCH / "posttraining/fullsize_parent"
        wiki = json.loads((out / "wikitext103.json").read_text())
        capability = json.loads((out / "capability_screen.json").read_text())
        raw = sum(float(capability["results"][task]["acc,none"])
                  for task in ("hellaswag", "arc_easy", "piqa", "winogrande")) / 4
        source = json.loads((out / "source_validation.json").read_text())["source_losses"]
        mixture = json.loads(MANIFEST.read_text())["mixture_definition"]
        heldout = sum(mixture[key] * source[key] for key in mixture)
        frozen_skill = json.loads((out / "heldout_skill_validation.json").read_text())["aggregate_nll_nats_per_completion_token"]
        child_skill = json.loads((RESEARCH / "runs" / FULLSIZE_ARM[0] / "heldout_skill_validation.json").read_text())["aggregate_nll_nats_per_completion_token"]
        lines.extend(["", "| 49M arm | Held-out loss | WikiText BPB | Four-task raw accuracy | Held-out skill NLL/token |",
                      "|---|---:|---:|---:|---:|",
                      f"| Frozen BASE | {heldout:.5f} | {wiki['bits_per_byte']:.5f} | {raw:.4f} | {frozen_skill:.5f} |",
                      f"| New P4 child | {child[0]:.5f} | {child[1]:.5f} | {child[2]:.4f} | {child_skill:.5f} |",
                      "", "P4 is post-trained from the immutable frozen BASE checkpoint on the same P1 objective, with a fresh optimizer and 500K logical target tokens. These evaluations use the common post-hackathon evaluator."])
    (RESEARCH / "POSTTRAINING_TRANSFER.md").write_text("\n".join(lines))


def main() -> None:
    if not (RESEARCH / "runs/F3-8M-CLONE-CATCHUP/hf_archive_receipt.json").exists():
        raise RuntimeError("developmental experts phase prerequisite missing")
    if not (DATA / "manifest.json").exists():
        invoke([sys.executable, "scripts/posthack_posttrain_data.py"],
               RESEARCH / "posttraining/data_build.log", "pinned posttraining data")
    data_receipt = DATA / "hf_archive_receipt.json"
    if not data_receipt.exists():
        receipts = upload([DATA / "manifest.json", DATA / "generic_train.jsonl",
                           DATA / "aligned_train.jsonl", DATA / "task_train.jsonl",
                           DATA / "heldout_skill_validation.jsonl"], "POSTTRAIN-DATA")
        atomic_json(data_receipt, {"at": utc(), "files": receipts})
    selection_path = RESEARCH / "posttraining/parent_selection.json"
    parent = json.loads(selection_path.read_text()) if selection_path.exists() else choose_parent()
    evaluate_skills(ROOT / parent["checkpoint_path"],
                    RESEARCH / "posttraining/parent_skill_validation.json",
                    RESEARCH / "posttraining/parent_evaluate.log")
    for arm in ARMS:
        register_arm(arm, parent)
        run_arm(arm)
    onpolicy = decide_onpolicy()
    if onpolicy["run"]:
        register_arm(ONPOLICY_ARM, parent, throughput_tps=onpolicy["estimated_tps"])
        run_arm(ONPOLICY_ARM)
    if not (RESEARCH / "posttraining/interpolation/report.json").exists():
        invoke([sys.executable, "scripts/posthack_interpolate.py"],
               RESEARCH / "posttraining/interpolation.log", "P1 weight interpolation")
    fullsize = decide_fullsize(parent)
    if fullsize["run"]:
        frozen_record = json.loads((RESEARCH / "FROZEN_HACKATHON_STATE.json").read_text())
        if sha256(FROZEN) != frozen_record["artifacts"]["checkpoint"]["sha256"]:
            raise RuntimeError("frozen BASE checkpoint identity changed")
        frozen_parent = {"experiment_id": "FROZEN-HACKATHON-BASE",
                         "checkpoint_path": str(FROZEN.relative_to(ROOT)),
                         "checkpoint_sha256": frozen_record["artifacts"]["checkpoint"]["sha256"]}
        evaluate_frozen_parent()
        register_arm(FULLSIZE_ARM, frozen_parent, budget=500_000,
                     throughput_tps=fullsize["estimated_tps"])
        run_arm(FULLSIZE_ARM)
    report(parent)
    upload([RESEARCH / "POSTTRAINING_TRANSFER.md", selection_path,
            RESEARCH / "posttraining/onpolicy_decision.json",
            RESEARCH / "posttraining/fullsize_decision.json",
            RESEARCH / "posttraining/parent_skill_validation.json"], "POSTTRAIN-REPORT")


if __name__ == "__main__":
    main()
