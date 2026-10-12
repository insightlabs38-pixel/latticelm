"""Sequential, deadline-aware matched Co4/dense campaign backbone."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from latticelm.config import LatticeConfig
from latticelm.final_recipe import muon_parameter_partition
from latticelm.model import build_model
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, can_launch, event, register, sha256, utc
from posthack_archive import upload
from posthack_train import valid_checkpoint

TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
ARMS = [
    ("A8-CO4-S1337", "8m", "co4", 1337, 100_000_000),
    ("A8-DENSE-S1337", "8m", "dense", 1337, 100_000_000),
    ("A8-CO4-S2026", "8m", "co4", 2026, 100_000_000),
    ("A8-DENSE-S2026", "8m", "dense", 2026, 100_000_000),
    ("A16-CO4-S1337", "16m", "co4", 1337, 150_000_000),
    ("A16-DENSE-S1337", "16m", "dense", 1337, 150_000_000),
    ("A32-CO4-S1337", "32m", "co4", 1337, 100_000_000),
    ("A32-DENSE-S1337", "32m", "dense", 1337, 100_000_000),
]


def config_path(regime: str, architecture: str, seed: int) -> Path:
    suffix = "_seed2026" if seed == 2026 else ""
    return RESEARCH / "configs" / f"{regime}_{architecture}{suffix}.json"


def chosen_calibration(state: dict, regime: str, architecture: str) -> dict:
    rows = [state["throughput_calibration"].get(f"{regime}/{architecture}/{threads}") for threads in (12, 16)]
    if any(row is None for row in rows):
        raise RuntimeError(f"missing required 12/16-thread calibration: {regime}/{architecture}")
    return max(rows, key=lambda row: row["tokens_per_second"])


def register_arms() -> None:
    state = json.loads(STATE.read_text())
    existing = {json.loads(line)["experiment_id"] for line in (RESEARCH / "RUN_REGISTRY.jsonl").open()}
    queued = []
    for experiment_id, regime, architecture, seed, budget in ARMS:
        path = config_path(regime, architecture, seed)
        cfg = LatticeConfig.from_json(path)
        calibration = chosen_calibration(state, regime, architecture)
        model = build_model(cfg)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _, _, partition = muon_parameter_partition(model)
        expected_seconds = budget / calibration["tokens_per_second"] + 3600
        if experiment_id not in existing:
            register({"experiment_id": experiment_id,
                      "hypothesis": "Modern causal Co4 MOD improves quality or efficiency relative to a parameter-matched dense Transformer",
                      "parent_control": f"{regime}/{seed}/{('dense' if architecture == 'co4' else 'co4')}",
                      "config": cfg.to_dict(), "config_sha256": sha256(path),
                      "tokenizer_sha256": sha256(TOKENIZER),
                      "sampling_manifest_sha256": sha256(MANIFEST),
                      "code_commit": state["source_commit"],
                      "source_file_sha256": {name: sha256(ROOT / name) for name in (
                          "src/latticelm/model.py", "src/latticelm/co4_reference.py",
                          "scripts/posthack_train.py", "scripts/posthack_phase1.py")},
                      "seed": seed, "parameter_count": params, "active_parameter_count": params,
                      "context_length": cfg.context_length, "optimizer_partition": partition,
                      "token_budget": budget,
                      "planned_evaluation_points": [min(budget, point) for point in (10_000_000, 25_000_000, 50_000_000, budget)],
                      "promotion_criterion": "Supported >=1% LM/BPB gain or >=10% compute/wall efficiency at comparable quality, without broad capability regression",
                      "abort_criterion": "Nonfinite training, contamination, >5% source-wise held-out regression at 50M, or projected finish after archival reserve",
                      "expected_wall_seconds": expected_seconds,
                      "estimate_provenance": calibration["experiment_id"],
                      "fairness_axes": ["parameters", "tokens", "estimated_FLOPs", "wall_clock"],
                      "initial_evaluation_policy": "Source-wise validation at each checkpoint; WikiText-103 full test and fixed 1000-example/task capability screen at endpoint",
                      "best_checkpoint_metric": "minimum frozen-mixture weighted held-out source loss at a registered checkpoint"})
        receipt = RESEARCH / "runs" / experiment_id / "hf_archive_receipt.json"
        if not receipt.exists():
            queued.append(experiment_id)
    state = json.loads(STATE.read_text())
    state["queued_runs"] = queued
    atomic_json(STATE, state)


def invoke(command: list[str], log: Path, label: str) -> float:
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc()}] {label} start\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        handle.write(f"[{utc()}] {label} completed\n")
    return time.perf_counter() - started


def run_arm(arm: tuple) -> None:
    experiment_id, regime, architecture, seed, budget = arm
    state = json.loads(STATE.read_text())
    calibration = chosen_calibration(state, regime, architecture)
    threads = calibration["threads"]
    run_dir = RESEARCH / "runs" / experiment_id
    result = run_dir / "result.json"
    if (run_dir / "hf_archive_receipt.json").exists():
        return
    already_trained = result.exists() and json.loads(result.read_text())["status"] == "COMPLETED"
    previous_tokens = 0
    resume_path = next((path for path in (run_dir / "latest.pt", run_dir / "previous.pt")
                        if valid_checkpoint(path)), None)
    resume = resume_path is not None
    if resume_path is not None:
        import torch
        previous_tokens = torch.load(resume_path, map_location="cpu", weights_only=False)["logical_tokens"]
    allowed, predicted_finish = can_launch(budget - previous_tokens,
                                           calibration["tokens_per_second"],
                                           eval_seconds=7200, checkpoint_seconds=1800,
                                           safety_seconds=3600)
    if not allowed:
        event(experiment_id, "RUN_REJECTED", reason="predicted finish crosses archival reserve",
              predicted_finish_utc=utc(predicted_finish))
        state = json.loads(STATE.read_text())
        state["decisions"].append({"at": utc(), "experiment_id": experiment_id,
                                   "decision": "DEFER", "reason": "archival reserve"})
        atomic_json(STATE, state)
        return
    event(experiment_id, "RUN_PROMOTED_TO_TRAIN", reason="matched-control backbone",
          predicted_finish_utc=utc(predicted_finish), calibration_id=calibration["experiment_id"])
    train_command = [sys.executable, "scripts/posthack_train.py", "--experiment-id", experiment_id,
                     "--config", str(config_path(regime, architecture, seed)),
                     "--target-tokens", str(budget), "--threads", str(threads),
                     "--checkpoint-tokens", "10000000", "--deadline-epoch", str(state["archive_start_epoch"])]
    if resume:
        train_command.append("--resume")
    train_seconds = 0.0 if already_trained else invoke(train_command, run_dir / "train.log", "training")
    outcome = json.loads(result.read_text())
    if outcome["status"] != "COMPLETED":
        event(experiment_id, "EVALUATION_DEFERRED", reason="training ended before registered endpoint")
        return
    checkpoint = run_dir / "latest.pt"
    eval_seconds = 0.0
    if not (run_dir / "wikitext103.json").exists():
        eval_seconds += invoke([sys.executable, "scripts/evaluate_wikitext103.py", "--checkpoint", str(checkpoint),
                                "--tokenizer", str(TOKENIZER), "--output", str(run_dir / "wikitext103.json"),
                                "--threads", "4"], run_dir / "evaluate.log", "WikiText-103")
    tasks = "hellaswag,arc_easy,piqa,winogrande,arc_challenge,openbookqa,boolq,lambada_openai"
    if not (run_dir / "capability_screen.json").exists():
        eval_seconds += invoke([sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", str(checkpoint),
                                "--tokenizer", str(TOKENIZER), "--output", str(run_dir / "capability_screen.json"),
                                "--tasks", tasks, "--limit", "1000", "--threads", "4",
                                "--predictions-output", str(run_dir / "capability_samples.json")],
                               run_dir / "evaluate.log", "capability screen")
    state = json.loads(STATE.read_text())
    state["measured_evaluation_seconds"][experiment_id] = eval_seconds
    state["queued_runs"] = [item for item in state["queued_runs"] if item != experiment_id]
    state["decisions"].append({"at": utc(), "experiment_id": experiment_id,
                               "decision": "EVALUATE_FOR_PROMOTION", "reason": "registered endpoint complete"})
    atomic_json(STATE, state)
    event(experiment_id, "ENDPOINT_EVALUATED", train_seconds=train_seconds, eval_seconds=eval_seconds)
    archive_paths = [config_path(regime, architecture, seed), result, run_dir / "evaluations.jsonl",
                     run_dir / "metrics.jsonl", run_dir / "wikitext103.json",
                     run_dir / "capability_screen.json", run_dir / "capability_samples.json", checkpoint]
    if valid_checkpoint(run_dir / "best.pt"):
        archive_paths += [run_dir / "best.pt", run_dir / "best.sha256"]
    receipts = upload(archive_paths, experiment_id)
    atomic_json(run_dir / "hf_archive_receipt.json", {"files": receipts, "at": utc()})


def main() -> None:
    register_arms()
    for arm in ARMS:
        run_arm(arm)


if __name__ == "__main__":
    main()
