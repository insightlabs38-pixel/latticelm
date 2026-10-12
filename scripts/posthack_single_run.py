"""Shared registration, launch, evaluation, and archive path for later screens."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from latticelm.config import LatticeConfig
from latticelm.final_recipe import muon_parameter_partition
from latticelm.posthack_variants import build_posthack_model
from posthack_archive import upload
from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, can_launch, event, register, sha256, utc
from posthack_phase1 import invoke
from posthack_train import valid_checkpoint

DEFAULT_MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
DEFAULT_TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
TASKS = "hellaswag,arc_easy,piqa,winogrande,arc_challenge,openbookqa,boolq,lambada_openai"


def registered_ids() -> set[str]:
    with (RESEARCH / "RUN_REGISTRY.jsonl").open(encoding="utf-8") as handle:
        return {json.loads(line)["experiment_id"] for line in handle}


def remaining_tokens(experiment_id: str, budget: int) -> int:
    run_dir = RESEARCH / "runs" / experiment_id
    result = run_dir / "result.json"
    if result.exists() and json.loads(result.read_text()).get("status") == "COMPLETED":
        return 0
    checkpoint = next((path for path in (run_dir / "latest.pt", run_dir / "previous.pt")
                       if valid_checkpoint(path)), None)
    if checkpoint is None:
        return budget
    import torch
    return max(0, budget - int(torch.load(checkpoint, map_location="cpu", weights_only=False)["logical_tokens"]))


def register_screen(*, experiment_id: str, config_path: Path, variant: str,
                    budget: int, parent_control: str, hypothesis: str,
                    throughput_tps: float, estimate_provenance: str,
                    mixture_weights: dict[str, float] | None = None,
                    manifest: Path = DEFAULT_MANIFEST,
                    tokenizer: Path = DEFAULT_TOKENIZER,
                    promotion_criterion: str | None = None,
                    abort_criterion: str | None = None,
                    extra_fields: dict | None = None) -> None:
    if experiment_id in registered_ids():
        return
    cfg = LatticeConfig.from_json(config_path)
    model = build_posthack_model(cfg, variant)
    if cfg.optimizer == "adamw":
        partition = {name: "adamw" for name, _ in model.named_parameters()}
    else:
        _, _, partition = muon_parameter_partition(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state = json.loads(STATE.read_text())
    registration = {
        "experiment_id": experiment_id, "hypothesis": hypothesis,
        "parent_control": parent_control, "config": cfg.to_dict(),
        "config_sha256": sha256(config_path), "variant": variant,
        "manifest_path": str(manifest.relative_to(ROOT)),
        "tokenizer_path": str(tokenizer.relative_to(ROOT)),
        "sampling_manifest_sha256": sha256(manifest),
        "tokenizer_sha256": sha256(tokenizer),
        "code_commit": state["source_commit"],
        "source_file_sha256": {name: sha256(ROOT / name) for name in (
            "src/latticelm/model.py", "src/latticelm/co4_reference.py",
            "src/latticelm/posthack_variants.py", "src/latticelm/final_recipe.py",
            "src/latticelm/research_sampling.py", "scripts/posthack_train.py")},
        "seed": cfg.seed, "parameter_count": params,
        "active_parameter_count": model.active_parameter_count() if hasattr(model, "active_parameter_count") else params,
        "context_length": cfg.context_length, "optimizer_partition": partition,
        "token_budget": budget,
        "planned_evaluation_points": sorted({min(budget, x) for x in (10_000_000, 25_000_000, 50_000_000, budget)}),
        "promotion_criterion": promotion_criterion or "Supported >=1% LM/BPB gain or >=10% wall/FLOP efficiency without capability regression",
        "abort_criterion": abort_criterion or "Nonfinite training, >5% source-wise held-out regression at 50M, or projected finish after reserve",
        "expected_wall_seconds": budget / throughput_tps + 3600,
        "estimated_throughput_tps": throughput_tps,
        "estimate_provenance": estimate_provenance,
        "evaluation_policy": "source-wise validation every checkpoint; endpoint full WikiText-103 and fixed 1000-example/task capability screen",
        "best_checkpoint_metric": "minimum frozen-mixture weighted held-out source loss at a registered checkpoint",
    }
    if mixture_weights is not None:
        registration["mixture_weights"] = mixture_weights
    if cfg.memory_token_map_path:
        registration["memory_token_map_sha256"] = sha256(ROOT / cfg.memory_token_map_path)
    if extra_fields:
        if set(extra_fields) & set(registration):
            raise ValueError("extra registration fields cannot override core identities")
        registration.update(extra_fields)
    register(registration)


def run_screen(experiment_id: str, config_path: Path, threads: int,
               throughput_tps: float, checkpoint_tokens: int = 10_000_000) -> dict:
    state = json.loads(STATE.read_text())
    with (RESEARCH / "RUN_REGISTRY.jsonl").open(encoding="utf-8") as handle:
        matches = [json.loads(line) for line in handle if json.loads(line)["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise RuntimeError("missing or ambiguous registered experiment")
    record = matches[0]
    run_dir = RESEARCH / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "hf_archive_receipt.json"
    if receipt_path.exists():
        return json.loads((run_dir / "result.json").read_text())
    result_path = run_dir / "result.json"
    completed = result_path.exists() and json.loads(result_path.read_text())["status"] == "COMPLETED"
    previous_tokens = 0
    resumable = next((path for path in (run_dir / "latest.pt", run_dir / "previous.pt")
                      if valid_checkpoint(path)), None)
    if resumable is not None and not completed:
        import torch
        previous_tokens = int(torch.load(resumable, map_location="cpu", weights_only=False)["logical_tokens"])
    remaining = 0 if completed else record["token_budget"] - previous_tokens
    allowed, predicted = can_launch(remaining, throughput_tps, 7200, 1800, 3600)
    if not allowed:
        event(experiment_id, "SCREEN_DEFERRED", reason="projected finish after archival reserve",
              predicted_finish_utc=utc(predicted))
        raise RuntimeError("registered screen does not fit before archival reserve")
    if not completed:
        command = [sys.executable, "scripts/posthack_train.py", "--experiment-id", experiment_id,
                   "--config", str(config_path), "--target-tokens", str(record["token_budget"]),
                   "--threads", str(threads), "--checkpoint-tokens", str(checkpoint_tokens),
                   "--deadline-epoch", str(state["archive_start_epoch"])]
        if resumable is not None:
            command.append("--resume")
        invoke(command, run_dir / "train.log", "training")
    result = json.loads(result_path.read_text())
    if result["status"] != "COMPLETED":
        event(experiment_id, "SCREEN_STOPPED", reason=result["status"])
        return result
    checkpoint = run_dir / "latest.pt"
    tokenizer = ROOT / record["tokenizer_path"]
    eval_seconds = 0.0
    if not (run_dir / "wikitext103.json").exists():
        eval_seconds += invoke([sys.executable, "scripts/evaluate_wikitext103.py",
                                "--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer),
                                "--output", str(run_dir / "wikitext103.json"), "--threads", "4"],
                               run_dir / "evaluate.log", "WikiText-103")
    if not (run_dir / "capability_screen.json").exists():
        eval_seconds += invoke([sys.executable, "scripts/evaluate_gibc.py",
                                "--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer),
                                "--output", str(run_dir / "capability_screen.json"),
                                "--tasks", TASKS, "--limit", "1000", "--threads", "4",
                                "--predictions-output", str(run_dir / "capability_samples.json")],
                               run_dir / "evaluate.log", "capability screen")
    state = json.loads(STATE.read_text())
    state["measured_evaluation_seconds"][experiment_id] = eval_seconds
    atomic_json(STATE, state)
    files = [config_path, result_path, run_dir / "metrics.jsonl", run_dir / "evaluations.jsonl",
             run_dir / "wikitext103.json", run_dir / "capability_screen.json",
             run_dir / "capability_samples.json", checkpoint]
    if valid_checkpoint(run_dir / "best.pt"):
        files += [run_dir / "best.pt", run_dir / "best.sha256"]
    if (run_dir / "routing.jsonl").exists():
        files.append(run_dir / "routing.jsonl")
    for name in record.get("extra_checkpoint_tokens", {}):
        milestone = run_dir / "milestones" / f"{name}.pt"
        if milestone.exists():
            files.append(milestone)
    receipts = upload(files, experiment_id)
    atomic_json(receipt_path, {"files": receipts, "at": utc()})
    state = json.loads(STATE.read_text())
    state["queued_runs"] = [name for name in state["queued_runs"] if name != experiment_id]
    atomic_json(STATE, state)
    event(experiment_id, "SCREEN_EVALUATED_AND_ARCHIVED", eval_seconds=eval_seconds)
    return result
