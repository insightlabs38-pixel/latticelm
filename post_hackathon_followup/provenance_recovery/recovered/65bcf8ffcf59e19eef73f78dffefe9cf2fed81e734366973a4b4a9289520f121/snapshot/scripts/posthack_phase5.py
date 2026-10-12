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

MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
DATA = RESEARCH / "posttraining/data"
ARMS = (
    ("P0-16M-GENERIC-REPLAY", "generic_replay", "generic_train.jsonl", 0.0),
    ("P1-16M-ALIGNED-REPLAY-KL", "aligned_replay_kl", "generic_train.jsonl", .10),
    ("P2-16M-TASK-TUNED-DIAGNOSTIC", "task_tuned_replay", "task_train.jsonl", 0.0),
)


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


def register_arm(arm: tuple, parent: dict) -> None:
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
    conservative_tps = base_tps * (.25 if kl_weight else .35)
    data_path = DATA / data_file
    registration = {"experiment_id": name,
                    "hypothesis": {
                        "generic_replay": "Generic SFT with 50% natural replay gains transfer while retaining LM skill",
                        "aligned_replay_kl": "Reference KL on natural replay limits specialization damage without suppressing broad gains",
                        "task_tuned_replay": "Official task training splits provide a diagnostic upper bound on directly learnable benchmark skill"}[objective],
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
                    "learning_rate": 3e-5, "token_budget": 1_000_000,
                    "planned_evaluation_points": [250_000, 500_000, 750_000, 1_000_000],
                    "promotion_criterion": "Broad held-out gain with preferably <2% LM/BPB retention loss and no source collapse; P2 diagnostic only",
                    "abort_criterion": "Nonfinite update, severe retention loss, or archival reserve",
                    "expected_wall_seconds": 1_000_000 / conservative_tps + 7200,
                    "estimated_throughput_tps": conservative_tps,
                    "estimate_provenance": "16m/co4/16 C4A calibration discounted for short SFT batches and optional teacher forward",
                    "parent_checkpoint_path": parent["checkpoint_path"],
                    "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                    "posttraining_data_path": str(data_path.relative_to(ROOT)),
                    "posttraining_data_sha256": sha256(data_path),
                    "posttraining_objective": objective,
                    "replay_fraction": .5, "reference_kl_weight": kl_weight,
                    "objective_equation": "L=0.5*CE_completion+0.5*CE_natural+0.5*lambda*KL(p_ref||p_theta) on natural replay tokens",
                    "comparison_label": "task-tuned diagnostic upper bound, not pretraining-only" if objective == "task_tuned_replay" else "retention-aware posttraining"}
    register(registration)


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
    files = [result, run / "metrics.jsonl", run / "evaluations.jsonl",
             run / "wikitext103.json", run / "capability_screen.json",
             run / "capability_samples.json", checkpoint]
    atomic_json(receipt, {"at": utc(), "files": upload(files, name)})
    state = json.loads(STATE.read_text())
    state["queued_runs"] = [item for item in state["queued_runs"] if item != name]
    atomic_json(STATE, state)


def report(parent: dict) -> None:
    names = [parent["experiment_id"]] + [arm[0] for arm in ARMS]
    scores = {name: endpoint(name) for name in names}
    baseline = scores[parent["experiment_id"]]
    lines = ["# Retention-aware post-training transfer", "",
             f"The parent is {parent['experiment_id']} (SHA-256 `{parent['checkpoint_sha256']}`). P0 trains on disjoint generic reasoning QA with 50% natural DATA-D replay. P1 uses the same data and replay, plus a frozen-parent reference KL on replay. P2 uses official training splits of the four competition tasks, with exact held-out-prompt hash exclusion; it is a **task-tuned diagnostic upper bound** and is not comparable to pretraining-only systems. All screens use a fresh AdamW optimizer, 3e-5 LR, one million logical target tokens, and the 4K tokenizer.", "",
             "The implemented objective is `0.5 CE_completion + 0.5 CE_natural + 0.5 λ KL(p_ref || p_θ)`; λ is 0 for P0/P2 and 0.10 for P1. KL is evaluated only on natural replay tokens. Completion loss masks prompt and padding tokens.", "",
             "| Arm | Held-out loss | WikiText BPB | Four-task raw accuracy | BPB change vs parent |",
             "|---|---:|---:|---:|---:|"]
    for name in names:
        heldout, bpb, accuracy, _ = scores[name]
        lines.append(f"| {name} | {heldout:.5f} | {bpb:.5f} | {accuracy:.4f} | {bpb / baseline[1] - 1:+.2%} |")
    lines += ["", "P2 is labeled task-tuned in the registry, data manifest, checkpoint lineage, and this report. A small raw-accuracy change is not a supported broad gain without paired uncertainty analysis. Weight interpolation is evaluated before any recovery training.", ""]
    interpolation = RESEARCH / "posttraining/interpolation/report.json"
    if interpolation.exists():
        selected = json.loads(interpolation.read_text())["selected"]
        lines.extend([f"Interpolation retained P1 fraction `{selected['alpha']:.2f}` under the pre-registered <=2% WikiText BPB and <=5% source-loss retention gates. The complete interpolation frontier and selected capability screen are in `posttraining/interpolation/report.json`.", ""])
    (RESEARCH / "POSTTRAINING_TRANSFER.md").write_text("\n".join(lines))


def main() -> None:
    if not (RESEARCH / "runs/F3-8M-CLONE-CATCHUP/hf_archive_receipt.json").exists():
        raise RuntimeError("developmental experts phase prerequisite missing")
    if not (DATA / "manifest.json").exists():
        invoke([sys.executable, "scripts/posthack_posttrain_data.py"],
               RESEARCH / "posttraining/data_build.log", "pinned posttraining data")
        upload([DATA / "manifest.json", DATA / "generic_train.jsonl",
                DATA / "task_train.jsonl"], "POSTTRAIN-DATA")
    selection_path = RESEARCH / "posttraining/parent_selection.json"
    parent = json.loads(selection_path.read_text()) if selection_path.exists() else choose_parent()
    for arm in ARMS:
        register_arm(arm, parent)
        run_arm(arm)
    if not (RESEARCH / "posttraining/interpolation/report.json").exists():
        invoke([sys.executable, "scripts/posthack_interpolate.py"],
               RESEARCH / "posttraining/interpolation.log", "P1 weight interpolation")
    report(parent)
    upload([RESEARCH / "POSTTRAINING_TRANSFER.md", selection_path], "POSTTRAIN-REPORT")


if __name__ == "__main__":
    main()
