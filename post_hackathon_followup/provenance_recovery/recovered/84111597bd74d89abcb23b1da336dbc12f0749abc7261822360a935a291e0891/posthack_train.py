"""Restartable single-run trainer for registered post-hackathon experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.data_d import SourceStream
from latticelm.final_recipe import make_optimizer, wsd_lr_scale
from latticelm.model import build_model
from posthack_campaign import RESEARCH, ROOT, STATE, append, atomic_json, event, set_active, sha256, utc
from train_final_production import WeightedMixture, load_arrays, next_batch_hash, validation

MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
STOP = False


def registered(experiment_id: str) -> dict:
    with (RESEARCH / "RUN_REGISTRY.jsonl").open(encoding="utf-8") as handle:
        matches = [json.loads(line) for line in handle if json.loads(line)["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise RuntimeError("run requires exactly one pre-training registration")
    return matches[0]


def valid_checkpoint(path: Path) -> bool:
    checksum = path.with_suffix(".sha256")
    return path.is_file() and checksum.is_file() and sha256(path) == checksum.read_text().strip()


def save_checkpoint(run_dir: Path, payload: dict) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = run_dir / ".latest.pt.tmp"
    torch.save(payload, temporary)
    digest = sha256(temporary)
    latest = run_dir / "latest.pt"
    previous = run_dir / "previous.pt"
    if latest.exists():
        os.replace(latest, previous)
        os.replace(latest.with_suffix(".sha256"), previous.with_suffix(".sha256"))
    os.replace(temporary, latest)
    latest.with_suffix(".sha256").write_text(digest + "\n")
    return digest


def evaluate(model: torch.nn.Module, valid: dict, context: int) -> tuple[dict, dict]:
    source_losses: dict[str, float] = {}
    paired: dict[str, list[float]] = {}
    for source, arrays in valid.items():
        source_losses[source], paired[source] = validation(model, np.concatenate(arrays), context, batches=8)
    return source_losses, paired


def run(args: argparse.Namespace) -> dict:
    record = registered(args.experiment_id)
    cfg = LatticeConfig.from_json(args.config)
    if json.loads(json.dumps(cfg.to_dict())) != record["config"]:
        raise RuntimeError("registered config differs from training config")
    if args.target_tokens != record["token_budget"]:
        raise RuntimeError("target differs from registered token budget")
    if sha256(MANIFEST) != record["sampling_manifest_sha256"] or sha256(TOKENIZER) != record["tokenizer_sha256"]:
        raise RuntimeError("registered data or tokenizer identity differs")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    top, train, valid = load_arrays(MANIFEST)
    streams = {source: SourceStream(train[source], cfg.context_length, cfg.seed + 17 * index)
               for index, source in enumerate(train)}
    mixture = WeightedMixture(streams, top["mixture_definition"], cfg.batch_size)
    model = build_model(cfg)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != record["parameter_count"]:
        raise RuntimeError("parameter count differs from registration")
    optimizer, partition = make_optimizer(model, cfg.optimizer, cfg.learning_rate,
                                          cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
    if partition != record["optimizer_partition"]:
        raise RuntimeError("optimizer partition differs from registration")
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    run_dir = RESEARCH / "runs" / args.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    step = tokens = 0
    prior_seconds = 0.0
    processed_tokens = 0
    if args.resume:
        checkpoint_path = next((path for path in (run_dir / "latest.pt", run_dir / "previous.pt") if valid_checkpoint(path)), None)
        if checkpoint_path is None:
            raise RuntimeError("no valid checkpoint for resume")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["experiment_id"] != args.experiment_id or checkpoint["target_tokens"] != args.target_tokens:
            raise RuntimeError("resume identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizers"])
        mixture.load_state_dict(checkpoint["mixture"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if next_batch_hash(mixture) != checkpoint["next_batch_sha256"]:
            raise RuntimeError("resume data-order mismatch")
        step, tokens = checkpoint["step"], checkpoint["logical_tokens"]
        processed_tokens = checkpoint["processed_tokens"]
        prior_seconds = checkpoint["wall_seconds"]
        event(args.experiment_id, "RESUMED", step=step, logical_tokens=tokens, checkpoint_sha256=sha256(checkpoint_path))
    elif any((run_dir / p).exists() for p in ("latest.pt", "previous.pt", "result.json")):
        raise RuntimeError("fresh run refuses to overwrite existing state")
    step_tokens = cfg.batch_size * cfg.context_length
    total_steps = math.ceil(args.target_tokens / step_tokens)
    started = time.perf_counter()
    set_active(args.experiment_id)
    event(args.experiment_id, "TRAIN_START", target_tokens=args.target_tokens,
          threads=args.threads, batch_size=cfg.batch_size, context_length=cfg.context_length)
    try:
        while tokens < args.target_tokens:
            state = json.loads(STATE.read_text())
            if time.time() >= state["archive_start_epoch"] or time.time() >= args.deadline_epoch:
                global STOP
                STOP = True
            if STOP and step == 0:
                break
            old_tokens = tokens
            step += 1
            scale = wsd_lr_scale(step, total_steps, cfg.warmup_fraction,
                                 cfg.stable_fraction, cfg.decay_fraction)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * scale
            x, y, labels = mixture.batch()
            increment = min(step_tokens, args.target_tokens - tokens)
            x_tensor = torch.from_numpy(x.astype(np.int64, copy=False))
            y_tensor = torch.from_numpy(y.astype(np.int64, copy=False))
            optimizer.zero_grad()
            logits, _ = model(x_tensor)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size)[:increment], y_tensor.reshape(-1)[:increment])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            tokens += increment
            processed_tokens += step_tokens
            elapsed = prior_seconds + time.perf_counter() - started
            append(metrics_path, {"at": utc(), "step": step, "logical_tokens": tokens,
                                  "processed_tokens": processed_tokens, "train_loss": float(loss.detach()),
                                  "lr": optimizer.param_groups[0]["lr"], "wall_seconds": elapsed,
                                  "tokens_per_second": tokens / max(elapsed, 1e-6),
                                  "batch_size": cfg.batch_size, "context_length": cfg.context_length,
                                  "source_labels": list(labels)})
            checkpoint_due = STOP or tokens >= args.target_tokens or tokens // args.checkpoint_tokens != old_tokens // args.checkpoint_tokens
            if checkpoint_due:
                eval_start = time.perf_counter()
                source_losses, paired = evaluate(model, valid, cfg.context_length)
                eval_seconds = time.perf_counter() - eval_start
                append(run_dir / "evaluations.jsonl", {"at": utc(), "logical_tokens": tokens,
                       "source_losses": source_losses, "paired_losses": paired,
                       "aggregate_loss": sum(source_losses.values()) / len(source_losses),
                       "evaluation_seconds": eval_seconds})
                payload = {"schema": "posthack-checkpoint-v1", "experiment_id": args.experiment_id,
                           "model": model.state_dict(), "optimizers": optimizer.state_dict(),
                           "config": cfg.to_dict(), "step": step, "logical_tokens": tokens,
                           "processed_tokens": processed_tokens, "target_tokens": args.target_tokens,
                           "wall_seconds": elapsed, "mixture": mixture.state_dict(),
                           "next_batch_sha256": next_batch_hash(mixture),
                           "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                           "torch_rng": torch.get_rng_state(), "optimizer_partition": partition,
                           "manifest_sha256": record["sampling_manifest_sha256"],
                           "tokenizer_sha256": record["tokenizer_sha256"]}
                digest = save_checkpoint(run_dir, payload)
                event(args.experiment_id, "CHECKPOINT", logical_tokens=tokens, processed_tokens=processed_tokens,
                      train_loss=float(loss.detach()), source_losses=source_losses,
                      checkpoint_sha256=digest, rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      evaluation_seconds=eval_seconds)
                if os.statvfs(run_dir).f_bavail * os.statvfs(run_dir).f_frsize < 8_000_000_000:
                    raise RuntimeError("disk safety floor crossed")
            if STOP:
                break
        result = {"experiment_id": args.experiment_id,
                  "status": "COMPLETED" if tokens == args.target_tokens else "STOPPED_SAFE",
                  "logical_tokens": tokens, "processed_tokens": processed_tokens,
                  "steps": step, "wall_seconds": prior_seconds + time.perf_counter() - started,
                  "checkpoint": str(run_dir / "latest.pt") if (run_dir / "latest.pt").exists() else None,
                  "checkpoint_sha256": sha256(run_dir / "latest.pt") if (run_dir / "latest.pt").exists() else None,
                  "finished_utc": utc()}
        atomic_json(run_dir / "result.json", result)
        event(args.experiment_id, "TRAIN_END",
              **{key: value for key, value in result.items() if key != "experiment_id"})
        return result
    finally:
        set_active(None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--checkpoint-tokens", type=int, default=10_000_000)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
