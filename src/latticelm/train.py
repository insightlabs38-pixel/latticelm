from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import subprocess
import resource

import psutil
import torch

from .config import LatticeConfig
from .data import batch_from_tokens, ensure_corpus, make_data
from .model import LatticeLM, build_model
from .tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed); torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def write_environment() -> None:
    artifacts = ROOT / "artifacts"; artifacts.mkdir(exist_ok=True)
    values = {
        "platform": platform.platform(), "python": platform.python_version(), "pytorch": torch.__version__,
        "cpu_count_logical": os.cpu_count(), "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "mkldnn_available": torch.backends.mkldnn.is_available(), "ram_bytes": psutil.virtual_memory().total,
        "triton_importable": _triton_importable(), "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
    }
    (artifacts / "environment.json").write_text(json.dumps(values, indent=2), encoding="utf-8")


def _triton_importable() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def evaluate(model: torch.nn.Module, data: torch.Tensor, config: LatticeConfig) -> float:
    model.eval()
    losses = []
    generator = torch.Generator().manual_seed(424242)
    with torch.no_grad():
        for _ in range(16):
            x, y = batch_from_tokens(data, config.batch_size, config.context_length, generator)
            _, loss = model(x, y)
            losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def append_result(record: dict) -> None:
    artifacts = ROOT / "artifacts"; artifacts.mkdir(exist_ok=True)
    with (artifacts / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    refresh_results_json()
    entries = [json.loads(line) for line in (artifacts / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    fields = list(dict.fromkeys(key for entry in entries for key in entry))
    with (artifacts / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(entries)
    tournament_fields = ["experiment", "architecture_family", "params", "neural_params", "memory_params",
                         "tokens_trained", "wall_seconds", "tokens_per_second", "train_loss", "val_loss",
                         "val_ppl", "peak_rss_bytes", "status", "notes"]
    tournament_path = artifacts / "architecture_tournament.csv"
    write_header = not tournament_path.exists()
    with tournament_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tournament_fields, extrasaction="ignore", lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def refresh_results_json() -> None:
    """Materialize the append-only JSONL ledger as the requested JSON artifact."""
    artifacts = ROOT / "artifacts"
    entries = [json.loads(line) for line in (artifacts / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    (artifacts / "results.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
                    config: LatticeConfig, source: str, generator: torch.Generator,
                    cumulative_wall_seconds: float, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "tokens_seen": step * config.batch_size * config.context_length,
                "config": config.to_dict(), "source": source,
                "torch_rng_state": torch.get_rng_state(), "random_state": random.getstate(),
                "data_generator_state": generator.get_state(),
                "cumulative_wall_seconds": cumulative_wall_seconds,
                "best_val_loss": best_val_loss}, path)
    path.with_suffix(".json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def train(config: LatticeConfig, experiment: str, corpus_path: str | None = None, download: bool = False, resume: str | None = None, tokenizer_path: str | None = None, validation_path: str | None = None, train_tokens_path: str | None = None, validation_tokens_path: str | None = None) -> dict:
    torch.set_num_threads(config.num_threads)
    generator = seed_everything(config.seed)
    write_environment()
    token_path = ROOT / (tokenizer_path or "artifacts/tokenizers/smoke.json")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if train_tokens_path and validation_tokens_path:
        tokenizer = load_tokenizer(token_path)
        def load_tokens(path: str) -> torch.Tensor:
            return (torch.load(path, map_location="cpu", weights_only=True)
                    if Path(path).suffix == ".pt" else torch.from_file(path, dtype=torch.int32)).long()
        train_data = load_tokens(train_tokens_path)
        val_data = load_tokens(validation_tokens_path)
        source = str(Path(train_tokens_path).resolve())
    else:
        text, source = ensure_corpus(corpus_path, download)
        validation_text = Path(validation_path).read_text(encoding="utf-8", errors="replace") if validation_path else None
        tokenizer, train_data, val_data = make_data(text, config.vocab_size, token_path, validation_text)
    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = build_model(config)
    breakdown = model.parameter_breakdown()
    if breakdown["total"] > 50_000_000:
        raise RuntimeError(f"parameter cap exceeded: {breakdown['total']:,}")
    if config.architecture == "mini_engram":
        memory_parameters = list(model.memory.parameters())
        memory_ids = {id(parameter) for parameter in memory_parameters}
        backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in memory_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone_parameters},
            {"params": memory_parameters, "lr": config.learning_rate * config.memory_lr_multiplier},
        ], lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start_step = 0
    previous_wall = 0.0
    best_val_loss = float("inf")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "random_state" in checkpoint:
            random.setstate(checkpoint["random_state"])
        if "data_generator_state" in checkpoint:
            generator.set_state(checkpoint["data_generator_state"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        log_path = ROOT / "artifacts" / "logs" / f"{experiment}.jsonl"
        if log_path.exists():
            previous = [json.loads(line) for line in log_path.read_text().splitlines() if line]
            previous_wall = max(float(checkpoint.get("cumulative_wall_seconds", 0.0)),
                                max((float(row.get("elapsed_wall_seconds", 0.0)) for row in previous), default=0.0))
    if config.hf_persistence_enabled:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            raise RuntimeError("HF persistence enabled but no HF_TOKEN/HUGGING_FACE_HUB_TOKEN is set")
        if not os.environ.get("LATTICELM_HF_REPO"):
            raise RuntimeError("HF persistence enabled but LATTICELM_HF_REPO is not set")
    start = time.perf_counter(); final_train_loss = float("nan"); val_loss = float("nan")
    for step in range(start_step + 1, config.max_steps + 1):
        x, y = batch_from_tokens(train_data, config.batch_size, config.context_length, generator)
        step_start = time.perf_counter()
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip); optimizer.step()
        final_train_loss = float(loss.detach())
        evaluated = step % config.eval_interval == 0 or step == config.max_steps
        improved = False
        if evaluated:
            val_loss = evaluate(model, val_data, config)
            improved = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
        tokens_seen = step * config.batch_size * config.context_length
        latest_due = (tokens_seen % config.hf_upload_interval_tokens == 0 or step == config.max_steps)
        best_upload_due = (config.hf_update_best and improved and
                           (tokens_seen % config.hf_best_upload_interval_tokens == 0 or step == config.max_steps))
        checkpoint_due = step % config.checkpoint_interval == 0 or step == config.max_steps or (config.hf_persistence_enabled and best_upload_due)
        checkpoint_path = ROOT / "artifacts" / "checkpoints" / f"{experiment}_step{step}.pt"
        cumulative_wall = previous_wall + time.perf_counter() - start
        if checkpoint_due:
            save_checkpoint(checkpoint_path, model, optimizer, step, config, source, generator,
                            cumulative_wall, best_val_loss)
        if config.hf_persistence_enabled and evaluated and (latest_due or best_upload_due):
            from .hf_storage import export_checkpoint, qualifies_as_remote_best, upload_checkpoint
            token = os.environ.get("HF_TOKEN") or os.environ["HUGGING_FACE_HUB_TOKEN"]
            metrics_now = {"tokens_trained": tokens_seen, "wall_seconds": cumulative_wall,
                           "val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            roles = ([('latest', 'latest')] if latest_due else [])
            roles += ([('best', 'best')] if best_upload_due and
                      qualifies_as_remote_best(os.environ["LATTICELM_HF_REPO"], token, val_loss) else [])
            if step == config.max_steps and config.hf_named_checkpoint:
                roles.append(('named', f"experiments/{config.hf_named_checkpoint}"))
            for role, remote_path in roles:
                staging = ROOT / "artifacts" / "hf_staging" / remote_path
                export_checkpoint(checkpoint_path, staging, experiment, role,
                                  token_path, token_path.with_suffix(".report.json"),
                                  ROOT / "artifacts" / "dataset_manifest.json", metrics_now)
                upload_checkpoint(staging, os.environ["LATTICELM_HF_REPO"], remote_path, token)
        elapsed_step = time.perf_counter() - step_start
        (ROOT / "artifacts" / "logs").mkdir(parents=True, exist_ok=True)
        with (ROOT / "artifacts" / "logs" / f"{experiment}.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"step": step, "tokens": step * config.batch_size * config.context_length,
                                  "train_loss": final_train_loss, "val_loss": val_loss if evaluated else None,
                                  "elapsed_wall_seconds": previous_wall + time.perf_counter() - start,
                                  "step_seconds": elapsed_step}) + "\n")
    wall = previous_wall + time.perf_counter() - start
    token_count = config.max_steps * config.batch_size * config.context_length
    record = {
        "experiment": experiment,
        "architecture_family": "dense" if config.architecture == "lattice" and not config.memory_enabled else config.architecture,
        "params": breakdown["total"], "neural_params": breakdown["total"] - breakdown["conditional_memory"],
        "memory_params": breakdown["conditional_memory"],
        "context": config.context_length, "tokens_trained": token_count, "wall_seconds": round(wall, 4),
        "tokens_per_second": round(token_count / wall, 3), "train_loss": round(final_train_loss, 6),
        "val_loss": round(val_loss, 6), "val_ppl": round(math.exp(val_loss), 5),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "torch_threads": config.num_threads, "triton_enabled": False, "data_source": source,
        "validation_source": (str(Path(validation_tokens_path).resolve()) if validation_tokens_path else
                              str(Path(validation_path).resolve()) if validation_path else "deterministic 10% tail"),
        "seed": config.seed, "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "config": config.to_dict(), "status": "completed",
        "notes": f"Round {1 if token_count < 1_000_000 else 2 if token_count < 3_000_000 else 3} common-settings architecture comparison",
        "checkpoint": str(ROOT / "artifacts" / "checkpoints" / f"{experiment}_step{config.max_steps}.pt"),
    }
    append_result(record)
    parameter_dir = ROOT / "artifacts" / "parameter_counts"
    parameter_dir.mkdir(exist_ok=True)
    (parameter_dir / f"{experiment}.json").write_text(json.dumps(breakdown, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reproducible LatticeLM experiment")
    parser.add_argument("--config", required=True); parser.add_argument("--experiment", required=True)
    parser.add_argument("--data"); parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume"); parser.add_argument("--tokenizer")
    parser.add_argument("--validation-data")
    parser.add_argument("--train-tokens"); parser.add_argument("--validation-tokens")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = LatticeConfig.from_json(args.config)
    if args.max_steps is not None: config = replace(config, max_steps=args.max_steps)
    print(json.dumps(train(config, args.experiment, args.data, args.download, args.resume, args.tokenizer,
                           args.validation_data, args.train_tokens, args.validation_tokens), indent=2))


if __name__ == "__main__":
    main()
