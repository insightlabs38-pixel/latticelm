"""Run the Spot-safe canonical GCP Co4-S lineage to exactly 10M tokens."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import time

from huggingface_hub import CommitOperationAdd, HfApi
import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.final_data import PermutedBlocks
from latticelm.model import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/data/phase7a"
CONFIG_PATH = ROOT / "configs/phase7c_final_10m.json"
MANIFEST_PATH = ROOT / "artifacts/gcp_final_10m_manifest.json"
TARGET_TOKENS = 10_000_000
FULL_STEP_TOKENS = 8 * 128
FULL_STEPS = TARGET_TOKENS // FULL_STEP_TOKENS
FINAL_REMAINDER = TARGET_TOKENS - FULL_STEPS * FULL_STEP_TOKENS
EVALUATIONS = {977: 1_000_000, 2930: 3_000_000, 4883: 5_000_000, 7324: 7_500_000,
               9766: 10_000_000}
REMOTE_INTERVAL_STEPS = 1954
CURVE_FIELDS = ["checkpoint", "nominal_tokens", "training_tokens", "step", "train_loss",
                "common_validation_loss", "common_validation_perplexity", "babylm_validation_loss",
                "finewebedu_validation_loss", "cumulative_training_seconds", "interval_training_seconds",
                "tokens_per_second", "peak_rss_bytes", "learning_rate", "babylm_tokens",
                "finewebedu_tokens", "babylm_sequences_drawn", "finewebedu_sequences_drawn",
                "checkpoint_sha256", "preemptions", "infrastructure_downtime_seconds"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load(name: str) -> torch.Tensor:
    return torch.tensor(np.memmap(DATA / f"{name}.int32", mode="r", dtype=np.int32), dtype=torch.long)


def evaluate(model: torch.nn.Module, tokens: torch.Tensor, batch: int, context: int) -> float:
    generator = torch.Generator().manual_seed(424242)
    losses = []
    model.eval()
    with torch.inference_mode():
        for _ in range(16):
            starts = torch.randint(0, len(tokens) - context - 1, (batch,), generator=generator)
            x = torch.stack([tokens[i:i + context] for i in starts.tolist()])
            y = torch.stack([tokens[i + 1:i + context + 1] for i in starts.tolist()])
            losses.append(float(model(x, y)[1]))
    model.train()
    return sum(losses) / len(losses)


def atomic_checkpoint(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    checksum = digest(temporary)
    temporary.replace(path)
    path.with_suffix(".sha256").write_text(checksum + "\n")
    if digest(path) != checksum:
        raise RuntimeError("checkpoint failed post-write checksum verification")
    return checksum


def roll_checkpoint(root: Path, payload: dict, known_good: bool) -> tuple[Path, str]:
    latest, previous, fallback = root / "latest.pt", root / "previous.pt", root / "fallback.pt"
    if latest.exists():
        shutil.copy2(latest, previous)
        latest.with_suffix(".sha256").replace(previous.with_suffix(".sha256"))
    path_hash = atomic_checkpoint(latest, payload)
    if known_good:
        shutil.copy2(latest, fallback)
        fallback.with_suffix(".sha256").write_text(path_hash + "\n")
    return latest, path_hash


def remote_recovery(path: Path, checksum: str, step: int) -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    repo = os.environ.get("LATTICELM_HF_REPO", "insightlabs38-pixel/LatticeLM-research")
    if not token:
        return None
    operation_checkpoint = CommitOperationAdd(path_in_repo="recovery/final-gcp-10m/latest.pt",
                                               path_or_fileobj=str(path))
    operation_checksum = CommitOperationAdd(path_in_repo="recovery/final-gcp-10m/latest.sha256",
                                             path_or_fileobj=io.BytesIO((checksum + "\n").encode()))
    commit = HfApi(token=token).create_commit(repo_id=repo, repo_type="model",
                                              operations=[operation_checkpoint, operation_checksum],
                                              commit_message=f"Recovery final-gcp-10m step {step}")
    return commit.oid


def make_payload(model, optimizer, scheduler, config, streams, step, training_tokens,
                 baby_tokens, web_tokens, training_seconds, last_eval_seconds, best, preemptions):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "config": config.to_dict(), "source": "DATA-C",
            "lineage": "final-gcp-10m", "step": step, "tokens_seen": training_tokens,
            "baby_tokens": baby_tokens, "finewebedu_tokens": web_tokens,
            "data_stream_state": {key: stream.draws for key, stream in streams.items()},
            "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
            "cumulative_training_seconds": training_seconds, "last_evaluation_seconds": last_eval_seconds,
            "best_validation_loss": best, "preemptions": preemptions,
            "manifest_sha256": digest(MANIFEST_PATH) if MANIFEST_PATH.exists() else "smoke-no-manifest"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    if args.fresh == args.resume:
        parser.error("choose exactly one of --fresh or --resume")
    if not args.smoke and not MANIFEST_PATH.exists():
        parser.error("canonical manifest must exist before token zero")
    config = LatticeConfig.from_json(CONFIG_PATH)
    torch.set_num_threads(config.num_threads); torch.set_num_interop_threads(1)
    random.seed(config.seed); torch.manual_seed(config.seed)
    baby, web = load("babylm_train"), load("finewebedu_train")
    baby_val, web_val, common = load("babylm_validation"), load("finewebedu_validation"), load("common_validation")
    streams = {"baby": PermutedBlocks(baby, 128, config.seed + 11),
               "web": PermutedBlocks(web, 128, config.seed + 29)}
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay,
                                  betas=(config.adam_beta1, config.adam_beta2), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint_root = ROOT / "artifacts/checkpoints" / ("phase7c-smoke" if args.smoke else "final-gcp-10m")
    latest = checkpoint_root / "latest.pt"
    if args.fresh and latest.exists():
        raise RuntimeError("refusing to silently replace an existing lineage; use --resume")
    step = training_tokens = baby_tokens = web_tokens = 0
    previous_training = last_eval_seconds = 0.0
    best = float("inf"); preemptions = 0
    if args.resume:
        if not latest.exists() or not latest.with_suffix(".sha256").exists():
            raise RuntimeError("no local valid checkpoint; run scripts/resume_final_gcp.py to recover remote state")
        expected = latest.with_suffix(".sha256").read_text().strip()
        if digest(latest) != expected:
            raise RuntimeError("latest checkpoint checksum mismatch")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["config"] != config.to_dict() or state["source"] != "DATA-C" or state["lineage"] != "final-gcp-10m":
            raise RuntimeError("checkpoint does not match frozen canonical lineage")
        if not args.smoke and state["manifest_sha256"] != digest(MANIFEST_PATH):
            raise RuntimeError("canonical manifest changed since checkpoint")
        model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); random.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        step, training_tokens = int(state["step"]), int(state["tokens_seen"])
        baby_tokens, web_tokens = int(state["baby_tokens"]), int(state["finewebedu_tokens"])
        previous_training = float(state["cumulative_training_seconds"])
        last_eval_seconds = float(state["last_evaluation_seconds"]); best = float(state["best_validation_loss"])
        preemptions = int(state.get("preemptions", 0)) + (0 if args.smoke else 1)
        for key, draws in state["data_stream_state"].items(): streams[key].draws = int(draws)
    final_step = args.stop_step or (5 if args.smoke else FULL_STEPS + 1)
    session_training = 0.0; train_loss = float("nan")
    curve = ROOT / ("artifacts/gcp_smoke_curve.csv" if args.smoke else "artifacts/gcp_final_training_curve.csv")
    for current in range(step + 1, final_step + 1):
        step_started = time.perf_counter()
        if not args.smoke and current == FULL_STEPS + 1:
            # Exact 75/25 allocation for the final 640 loss-bearing tokens.
            pairs = [streams["baby"].one() for _ in range(4)] + [streams["web"].one() for _ in range(2)]
            x = torch.stack([p[0] for p in pairs]); y = torch.stack([p[1] for p in pairs])
            logits = model(x)[0]
            per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(6, 128)
            mask = torch.zeros_like(per_token); mask[0:3] = 1; mask[3, :96] = 1; mask[4] = 1; mask[5, :32] = 1
            loss = (per_token * mask).sum() / FINAL_REMAINDER
            increment, baby_increment, web_increment = FINAL_REMAINDER, 480, 160
        else:
            pairs = [streams["baby"].one() for _ in range(6)] + [streams["web"].one() for _ in range(2)]
            x = torch.stack([p[0] for p in pairs]); y = torch.stack([p[1] for p in pairs])
            loss = model(x, y)[1]
            increment, baby_increment, web_increment = FULL_STEP_TOKENS, 768, 256
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {current}")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"non-finite gradient at step {current}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip); optimizer.step(); scheduler.step()
        session_training += time.perf_counter() - step_started
        train_loss = float(loss.detach()); training_tokens += increment
        baby_tokens += baby_increment; web_tokens += web_increment
        elapsed = previous_training + session_training
        evaluation = current in EVALUATIONS or current == final_step
        periodic = current % config.checkpoint_interval == 0
        if not (evaluation or periodic): continue
        payload = make_payload(model, optimizer, scheduler, config, streams, current, training_tokens,
                               baby_tokens, web_tokens, elapsed, last_eval_seconds, best, preemptions)
        checkpoint_path, checkpoint_hash = roll_checkpoint(checkpoint_root, payload, evaluation)
        if not evaluation: continue
        baby_loss = evaluate(model, baby_val, config.batch_size, config.context_length)
        web_loss = evaluate(model, web_val, config.batch_size, config.context_length)
        common_loss = evaluate(model, common, config.batch_size, config.context_length)
        best = min(best, common_loss)
        # Resave evaluation metadata as a known-good recovery point.
        elapsed = previous_training + session_training
        payload = make_payload(model, optimizer, scheduler, config, streams, current, training_tokens,
                               baby_tokens, web_tokens, elapsed, elapsed, best, preemptions)
        checkpoint_path, checkpoint_hash = roll_checkpoint(checkpoint_root, payload, True)
        remote_revision = None
        if not args.no_upload and not args.smoke and (current % REMOTE_INTERVAL_STEPS == 0 or current in EVALUATIONS):
            remote_revision = remote_recovery(checkpoint_path, checkpoint_hash, current)
        nominal = EVALUATIONS.get(current, training_tokens)
        row = {"checkpoint": f"final-gcp-{nominal}", "nominal_tokens": nominal,
               "training_tokens": training_tokens, "step": current, "train_loss": train_loss,
               "common_validation_loss": common_loss, "common_validation_perplexity": math.exp(common_loss),
               "babylm_validation_loss": baby_loss, "finewebedu_validation_loss": web_loss,
               "cumulative_training_seconds": elapsed,
               "interval_training_seconds": elapsed - last_eval_seconds,
               "tokens_per_second": (training_tokens - (step and int(state["tokens_seen"]) or 0)) / max(elapsed - previous_training, 1e-9),
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "learning_rate": optimizer.param_groups[0]["lr"], "babylm_tokens": baby_tokens,
               "finewebedu_tokens": web_tokens, "babylm_sequences_drawn": streams["baby"].draws,
               "finewebedu_sequences_drawn": streams["web"].draws, "checkpoint_sha256": checkpoint_hash,
               "preemptions": preemptions, "infrastructure_downtime_seconds": 0.0}
        exists = curve.exists()
        with curve.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS, lineterminator="\n")
            if not exists: writer.writeheader()
            writer.writerow(row)
        raw_log = ROOT / ("artifacts/logs/gcp_smoke.jsonl" if args.smoke else "artifacts/logs/gcp_final_10m.jsonl")
        with raw_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**row, "remote_recovery_revision": remote_revision}) + "\n")
        print(json.dumps({**row, "remote_recovery_revision": remote_revision}), flush=True)
        last_eval_seconds = elapsed
    if not args.smoke and training_tokens != TARGET_TOKENS:
        raise RuntimeError(f"stopped at {training_tokens}, expected exactly {TARGET_TOKENS}")


if __name__ == "__main__":
    main()
