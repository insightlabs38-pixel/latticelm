"""Restart-safe final LatticeLM trainer with immutable semantic milestones.

This program is deliberately policy-light: the production master freezes the
recipe and budget, while this child owns exact training state, checkpoint
integrity, telemetry, and deadline projections.  It never increases a target.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import resource
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.data_d import Int32Shard, SourceStream, sha256_file, verify_top_manifest
from latticelm.final_recipe import make_optimizer, wsd_lr_scale
from latticelm.model import build_model

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RUN = ART / "final_production_run"
EVENTS = ART / "final_training_events.jsonl"
CURVE = ART / "final_training_curve.csv"
LOCK = ART / "final_production_training.lock"
STOP = False


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def event(kind: str, **fields) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": time.time(), "at_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush(); os.fsync(handle.fileno())


class WeightedMixture:
    def __init__(self, streams, weights, batch_size: int):
        self.streams = dict(streams); self.counter = 0; self.batch_size = batch_size
        counts = {name: round(weight * 20) for name, weight in weights.items()}
        counts[next(iter(counts))] += 20 - sum(counts.values())
        self.cycle = tuple(name for name in weights for _ in range(counts[name]))

    def batch(self):
        labels = tuple(self.cycle[(self.counter + index) % len(self.cycle)] for index in range(self.batch_size))
        self.counter += self.batch_size
        pairs = [self.streams[name].one() for name in labels]
        return np.stack([x for x, _ in pairs]), np.stack([y for _, y in pairs]), labels

    def state_dict(self):
        return {"counter": self.counter, "draws": {name: stream.draws for name, stream in self.streams.items()}}

    def load_state_dict(self, state):
        self.counter = int(state["counter"])
        for name, stream in self.streams.items(): stream.draws = int(state["draws"][name])


def next_batch_hash(mixture: WeightedMixture) -> str:
    state = mixture.state_dict(); x, y, _ = mixture.batch(); mixture.load_state_dict(state)
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()


def load_arrays(manifest: Path):
    top = json.loads(manifest.read_text()); base = manifest.parent
    train = {source: [] for source in top["mixture_definition"]}
    valid = {source: [] for source in train}
    for key, destination in (("shards", train), ("validation_shards", valid)):
        for child in top[key]:
            meta = json.loads((base / child["manifest_path"]).read_text())
            destination[meta["source"]].append(Int32Shard(base / meta["path"], meta).tokens)
    return top, train, valid


def validation(model, values, context: int, batches: int = 8) -> tuple[float, list[float]]:
    model.eval(); losses: list[float] = []
    with torch.inference_mode():
        for index in range(batches):
            start = index * 4 * context
            chunks = [np.asarray(values[start + i * context:start + (i + 1) * context + 1]) for i in range(4)]
            if any(len(chunk) != context + 1 for chunk in chunks): break
            x = torch.tensor(np.stack([chunk[:-1] for chunk in chunks]), dtype=torch.long)
            y = torch.tensor(np.stack([chunk[1:] for chunk in chunks]), dtype=torch.long)
            _, loss = model(x, y); losses.append(float(loss))
    model.train()
    if not losses: raise RuntimeError("validation corpus too short")
    return sum(losses) / len(losses), losses


def valid_checkpoint(path: Path) -> bool:
    sidecar = path.with_suffix(".sha256")
    return path.is_file() and sidecar.is_file() and sha256_file(path) == sidecar.read_text().strip()


def save_checkpoint(payload: dict, milestone: str | None = None) -> str:
    RUN.mkdir(parents=True, exist_ok=True)
    partial = RUN / ".latest.pt.partial"; partial_sum = RUN / ".latest.sha256.partial"
    torch.save(payload, partial); checksum = sha256_file(partial); partial_sum.write_text(checksum + "\n")
    with partial_sum.open("r+") as handle: handle.flush(); os.fsync(handle.fileno())
    latest = RUN / "latest.pt"; previous = RUN / "previous.pt"
    if latest.exists(): os.replace(latest, previous)
    if latest.with_suffix(".sha256").exists(): os.replace(latest.with_suffix(".sha256"), previous.with_suffix(".sha256"))
    os.replace(partial, latest); os.replace(partial_sum, latest.with_suffix(".sha256"))
    if milestone:
        destination = RUN / f"{milestone}.pt"
        if not destination.exists():
            os.link(latest, destination); destination.with_suffix(".sha256").write_text(checksum + "\n")
    return checksum


def mirror_milestone(path: Path, role: str) -> dict:
    """Best-effort remote durability; publication failures never corrupt training."""
    from huggingface_hub import HfApi
    repo = os.environ.get("LATTICELM_HF_REPO", "insightlabs38-pixel/LatticeLM-research")
    remote = f"final-production/milestones/{role}.pt"; api = HfApi()
    commit = api.upload_file(path_or_fileobj=path, path_in_repo=remote, repo_id=repo,
                             commit_message=f"Mirror final production {role}")
    sibling = next(item for item in api.repo_info(repo, files_metadata=True).siblings if item.rfilename == remote)
    local_hash = sha256_file(path); lfs = getattr(sibling, "lfs", None)
    remote_hash = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if sibling.size != path.stat().st_size or (remote_hash and remote_hash != local_hash):
        raise RuntimeError(f"remote milestone verification failed: {role}")
    return {"repo_id": repo, "path": remote, "revision": commit.oid, "size": sibling.size,
            "local_sha256": local_hash, "lfs_sha256": remote_hash, "verified": True}


def semantic_milestones(target: int) -> dict[str, int]:
    return {
        "early-health": min(10_000_000, target),
        "quarter": round(target * .25),
        "half": round(target * .50),
        "end-stable": round(target * .85),
        "mid-decay": round(target * .925),
        "final": target,
    }


def crossed(old: int, new: int, point: int) -> bool:
    return old < point <= new


def choose_downshift(current_target: int, tokens: int, measured_tps: float, deadline_epoch: float,
                     reserve_hours: float = 50.0, now: float | None = None) -> int:
    """Return the largest non-increasing authorized target preserving reserves."""
    now = time.time() if now is None else now
    usable_seconds = max(0.0, deadline_epoch - now) * 22 / 24 - reserve_hours * 3600
    capacity = tokens + max(0.0, usable_seconds) * measured_tps * .90
    minimum_schedule_target = math.ceil(tokens / .85)
    candidates = [2_250_000_000, 2_200_000_000, 2_100_000_000, 2_000_000_000, 1_900_000_000, 1_800_000_000]
    safe = [value for value in candidates if value <= current_target and value >= minimum_schedule_target and value <= capacity]
    if safe: return max(safe)
    floor = int(capacity // 50_000_000) * 50_000_000
    if floor < minimum_schedule_target: return current_target
    return min(current_target, floor)


def first_batch_certification(cfg, mixture, model, optimizer, manifest_hash: str, tokenizer_hash: str,
                              output: Path) -> dict:
    """One-step reload drill that restores all state before real step zero."""
    initial_model = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    initial_optim = optimizer.state_dict(); initial_mix = mixture.state_dict()
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    expected = next_batch_hash(mixture); x, y, _ = mixture.batch()
    x_tensor = torch.tensor(x, dtype=torch.long); y_tensor = torch.tensor(y, dtype=torch.long)
    optimizer.zero_grad(); logits, _ = model(x_tensor, y_tensor)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y_tensor.reshape(-1)); loss.backward()
    finite = bool(torch.isfinite(loss)) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if not finite: raise FloatingPointError("prelaunch drill produced nonfinite loss or gradients")
    optimizer.step()
    payload = {"model": model.state_dict(), "optimizers": optimizer.state_dict(), "mixture": mixture.state_dict(),
               "manifest_sha256": manifest_hash, "tokenizer_sha256": tokenizer_hash,
               "next_batch_sha256": next_batch_hash(mixture)}
    output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, output)
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    if loaded["next_batch_sha256"] != next_batch_hash(mixture): raise RuntimeError("prelaunch drill reload mismatch")
    model.load_state_dict(initial_model, strict=True); optimizer.load_state_dict(initial_optim); mixture.load_state_dict(initial_mix)
    random.setstate(py_state); np.random.set_state(np_state); torch.set_rng_state(torch_state)
    if next_batch_hash(mixture) != expected: raise RuntimeError("prelaunch drill did not restore batch identity")
    return {"status": "PASS", "first_batch_sha256": expected, "loss": float(loss.detach()), "finite_gradients": True}


def run(args) -> dict:
    global STOP
    cfg = LatticeConfig.from_json(args.config)
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    verify_top_manifest(args.manifest, args.tokenizer)
    manifest_hash, tokenizer_hash = sha256_file(args.manifest), sha256_file(args.tokenizer)
    top, train, valid = load_arrays(args.manifest)
    streams = {source: SourceStream(train[source], cfg.context_length, cfg.seed + 17 * index)
               for index, source in enumerate(train)}
    mixture = WeightedMixture(streams, top["mixture_definition"], cfg.batch_size)
    model = build_model(cfg)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count >= 50_000_000: raise RuntimeError(f"parameter cap exceeded: {parameter_count}")
    optimizer, partition = make_optimizer(model, cfg.optimizer, cfg.learning_rate, cfg.weight_decay,
                                           (cfg.adam_beta1, cfg.adam_beta2))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    config_hash = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()
    target = int(args.target_tokens); original_target = target
    step = tokens = 0; prior_seconds = 0.0; restart_count = 0; last_loss = float("nan"); started_at = time.time()
    if args.fresh and any((RUN / name).exists() for name in ("latest.pt", "previous.pt", "result.json")):
        raise RuntimeError("fresh production launch refused: production state already exists")
    if args.resume:
        checkpoint_path = next((path for path in (RUN / "latest.pt", RUN / "previous.pt") if valid_checkpoint(path)), None)
        if checkpoint_path is None: raise RuntimeError("both latest and previous production checkpoints are invalid")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {"config_sha256": config_hash, "manifest_sha256": manifest_hash, "tokenizer_sha256": tokenizer_hash}
        if any(checkpoint.get(key) != value for key, value in expected.items()): raise RuntimeError("resume identity mismatch")
        model.load_state_dict(checkpoint["model"], strict=True); optimizer.load_state_dict(checkpoint["optimizers"])
        mixture.load_state_dict(checkpoint["mixture"]); random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"]); torch.set_rng_state(checkpoint["torch_rng"])
        if next_batch_hash(mixture) != checkpoint["next_batch_sha256"]: raise RuntimeError("resume next-batch mismatch")
        step, tokens = int(checkpoint["step"]), int(checkpoint["tokens"])
        prior_seconds = float(checkpoint["training_seconds"]); target = int(checkpoint["target_tokens"])
        restart_count = int(checkpoint.get("restart_count", 0)) + 1; last_loss = float(checkpoint.get("train_loss", "nan"))
        started_at = float(checkpoint.get("started_at", started_at - prior_seconds))
        event("RESUME", checkpoint=str(checkpoint_path), tokens=tokens, target_tokens=target, restart_count=restart_count)
    else:
        drill = first_batch_certification(cfg, mixture, model, optimizer, manifest_hash, tokenizer_hash, RUN / "prelaunch-drill.pt")
        atomic_json(ART / "final_prelaunch_certification.json", {**drill, "parameter_count": parameter_count,
                    "seed": cfg.seed, "manifest_sha256": manifest_hash, "tokenizer_sha256": tokenizer_hash,
                    "config_sha256": config_hash, "lr_at_step_zero": 0.0, "optimizer_partition": partition})
        (RUN / "prelaunch-drill.pt").unlink(missing_ok=True)
    if args.certify_only:
        return {"status": "CERTIFIED", "parameter_count": parameter_count, "next_batch_sha256": next_batch_hash(mixture)}
    if args.backend == "compile": model.compile(mode="max-autotune-no-cudagraphs", fullgraph=False)
    started = time.perf_counter(); step_tokens = cfg.batch_size * cfg.context_length
    curve_exists = CURVE.exists(); CURVE.parent.mkdir(parents=True, exist_ok=True)
    curve_handle = CURVE.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(curve_handle, fieldnames=("step", "tokens", "train_loss", "lr", "lr_scale", "elapsed_seconds", "effective_tok_s", "target_tokens"))
    if not curve_exists: writer.writeheader()
    event("TRAINING_START", tokens=tokens, target_tokens=target, parameter_count=parameter_count, first_batch_sha256=next_batch_hash(mixture))
    while tokens < target:
        if time.time() >= args.deadline_epoch: raise RuntimeError("hard internal deadline reached during production")
        old_tokens = tokens; total_steps = math.ceil(target / step_tokens); step += 1
        scale = wsd_lr_scale(step, total_steps, cfg.warmup_fraction, cfg.stable_fraction, cfg.decay_fraction)
        for group, base_lr in zip(optimizer.param_groups, base_lrs): group["lr"] = base_lr * scale
        x, y, _ = mixture.batch(); increment = min(step_tokens, target - tokens)
        x_tensor, y_tensor = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
        optimizer.zero_grad(); logits, _ = model(x_tensor)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size)[:increment], y_tensor.reshape(-1)[:increment]); loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"nonfinite production trajectory at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); optimizer.step()
        tokens += increment; last_loss = float(loss.detach()); elapsed = prior_seconds + time.perf_counter() - started
        effective_tps = tokens / max(elapsed, 1e-9)
        writer.writerow({"step": step, "tokens": tokens, "train_loss": last_loss, "lr": optimizer.param_groups[0]["lr"],
                         "lr_scale": scale, "elapsed_seconds": elapsed, "effective_tok_s": effective_tps, "target_tokens": target})
        curve_handle.flush()
        milestones = semantic_milestones(target)
        checkpoint_due = STOP or tokens == target or tokens // args.checkpoint_tokens != old_tokens // args.checkpoint_tokens
        milestone = next((name for name, point in milestones.items() if crossed(old_tokens, tokens, point)), None)
        if checkpoint_due or milestone:
            source_validation = {}; paired = {}
            for source, arrays in valid.items():
                value, samples = validation(model, np.concatenate(arrays), cfg.context_length)
                source_validation[source], paired[source] = value, samples
            mean_validation = sum(source_validation.values()) / len(source_validation)
            payload = {"schema": "final-production-checkpoint-v1", "model": model.state_dict(), "optimizers": optimizer.state_dict(),
                       "config": cfg.to_dict(), "config_sha256": config_hash, "step": step, "tokens": tokens,
                       "target_tokens": target, "original_target_tokens": original_target, "train_loss": last_loss,
                       "training_seconds": elapsed, "started_at": started_at, "mixture": mixture.state_dict(), "next_batch_sha256": next_batch_hash(mixture),
                       "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                       "manifest_sha256": manifest_hash, "tokenizer_sha256": tokenizer_hash, "restart_count": restart_count,
                       "optimizer_partition": partition, "source_validation": source_validation,
                       "validation_loss": mean_validation, "paired_validation_losses": paired}
            checksum = save_checkpoint(payload, milestone)
            free_bytes = shutil.disk_usage(ROOT).free
            event("CHECKPOINT", tokens=tokens, target_tokens=target, milestone=milestone, sha256=checksum,
                  validation_loss=mean_validation, source_validation=source_validation, effective_tok_s=effective_tps,
                  rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, disk_free_bytes=free_bytes)
            if milestone:
                try: event("MILESTONE_MIRRORED", role=milestone, **mirror_milestone(RUN / f"{milestone}.pt", milestone))
                except Exception as error: event("MILESTONE_MIRROR_RETRY_REQUIRED", role=milestone, error=f"{type(error).__name__}: {error}")
            if free_bytes < args.minimum_free_bytes: raise RuntimeError("disk safety floor crossed")
            if tokens < round(target * .85) and tokens >= 50_000_000:
                revised = choose_downshift(target, tokens, effective_tps, args.deadline_epoch)
                if revised < target:
                    old_target = target; target = revised
                    event("TARGET_DOWNSHIFT", old_target=old_target, new_target=target, tokens=tokens,
                          reason="projected completion threatens protected 50-hour envelope")
                    if target < 1_500_000_000: event("CRITICAL_SCHEDULE_WARNING", target_tokens=target)
        if STOP:
            event("STOPPED_SAFE", tokens=tokens); curve_handle.close(); return {"status": "STOPPED_SAFE", "tokens": tokens}
    curve_handle.close(); elapsed = prior_seconds + time.perf_counter() - started
    final_path = RUN / "final.pt"
    if not final_path.exists(): os.link(RUN / "latest.pt", final_path); final_path.with_suffix(".sha256").write_text(sha256_file(final_path) + "\n")
    result = {"schema": "final-production-result-v1", "status": "TRAINING_COMPLETE", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "config_sha256": config_hash, "manifest_sha256": manifest_hash, "tokenizer_sha256": tokenizer_hash,
              "parameter_count": parameter_count, "tokens": tokens, "original_target_tokens": original_target,
              "final_target_tokens": target, "training_seconds": elapsed, "effective_tokens_per_second": tokens / elapsed,
              "restart_count": restart_count, "train_loss": last_loss, "checkpoint": str(final_path),
              "checkpoint_sha256": sha256_file(final_path), "started_at": started_at, "completed_at": time.time()}
    atomic_json(RUN / "result.json", result); event("TRAINING_COMPLETE", **result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True); parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True); parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--backend", choices=("eager", "compile"), default="compile")
    parser.add_argument("--checkpoint-tokens", type=int, default=50_000_000)
    parser.add_argument("--minimum-free-bytes", type=int, default=15_000_000_000)
    parser.add_argument("--certify-only", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--fresh", action="store_true"); mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True)); signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(run(args), indent=2, sort_keys=True)); return 0
    except BlockingIOError:
        print("another production trainer owns the lock", file=os.sys.stderr); return 73
    except RuntimeError as error:
        if "both latest and previous" in str(error) or "identity mismatch" in str(error): event("PRODUCTION_BLOCKED_CORRUPTION", error=str(error)); return 2
        raise


if __name__ == "__main__": raise SystemExit(main())
