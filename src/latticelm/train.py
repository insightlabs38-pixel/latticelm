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

import psutil
import torch

from .config import LatticeConfig
from .data import batch_from_tokens, ensure_corpus, make_data
from .model import LatticeLM, build_model

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


def evaluate(model: torch.nn.Module, data: torch.Tensor, config: LatticeConfig, generator: torch.Generator) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(3):
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
    csv_path = artifacts / "results.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if write_header: writer.writeheader()
        writer.writerow(record)


def refresh_results_json() -> None:
    """Materialize the append-only JSONL ledger as the requested JSON artifact."""
    artifacts = ROOT / "artifacts"
    entries = [json.loads(line) for line in (artifacts / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    (artifacts / "results.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, config: LatticeConfig, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": config.to_dict(), "source": source}, path)
    path.with_suffix(".json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def train(config: LatticeConfig, experiment: str, corpus_path: str | None = None, download: bool = False, resume: str | None = None, tokenizer_path: str | None = None) -> dict:
    torch.set_num_threads(config.num_threads)
    generator = seed_everything(config.seed)
    write_environment()
    text, source = ensure_corpus(corpus_path, download)
    token_path = ROOT / (tokenizer_path or "artifacts/tokenizers/smoke.json")
    token_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, train_data, val_data = make_data(text, config.vocab_size, token_path)
    config = replace(config, vocab_size=tokenizer.vocab_size)
    model = build_model(config)
    breakdown = model.parameter_breakdown()
    if breakdown["total"] > 50_000_000:
        raise RuntimeError(f"parameter cap exceeded: {breakdown['total']:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    process = psutil.Process()
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
        if step % config.eval_interval == 0 or step == config.max_steps:
            val_loss = evaluate(model, val_data, config, generator)
        if step % config.checkpoint_interval == 0 or step == config.max_steps:
            save_checkpoint(ROOT / "artifacts" / "checkpoints" / f"{experiment}_step{step}.pt", model, optimizer, step, config, source)
        elapsed_step = time.perf_counter() - step_start
        (ROOT / "artifacts" / "logs").mkdir(parents=True, exist_ok=True)
        with (ROOT / "artifacts" / "logs" / f"{experiment}.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"step": step, "train_loss": final_train_loss, "step_seconds": elapsed_step}) + "\n")
    wall = time.perf_counter() - start
    token_count = (config.max_steps - start_step) * config.batch_size * config.context_length
    record = {
        "experiment": experiment, "params": breakdown["total"], "memory_params": breakdown["conditional_memory"],
        "context": config.context_length, "tokens_trained": token_count, "wall_seconds": round(wall, 4),
        "tokens_per_second": round(token_count / wall, 3), "train_loss": round(final_train_loss, 6),
        "val_loss": round(val_loss, 6), "val_ppl": round(math.exp(val_loss), 5), "peak_rss_bytes": process.memory_info().rss,
        "torch_threads": config.num_threads, "triton_enabled": False, "data_source": source,
        "checkpoint": str(ROOT / "artifacts" / "checkpoints" / f"{experiment}_step{config.max_steps}.pt"),
    }
    append_result(record)
    (ROOT / "artifacts" / "parameter_count.txt").write_text(json.dumps(breakdown, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reproducible LatticeLM experiment")
    parser.add_argument("--config", required=True); parser.add_argument("--experiment", required=True)
    parser.add_argument("--data"); parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume"); parser.add_argument("--tokenizer")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = LatticeConfig.from_json(args.config)
    if args.max_steps is not None: config = replace(config, max_steps=args.max_steps)
    print(json.dumps(train(config, args.experiment, args.data, args.download, args.resume, args.tokenizer), indent=2))


if __name__ == "__main__":
    main()
