"""Restartable 16M retention-aware SFT, replay, and reference-KL screens."""
from __future__ import annotations

import argparse
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
from tokenizers import Tokenizer

from latticelm.config import LatticeConfig
from latticelm.data_d import SourceStream
from latticelm.final_recipe import make_optimizer
from latticelm.posthack_variants import build_posthack_model
from posthack_campaign import RESEARCH, ROOT, STATE, append, atomic_json, event, set_active, sha256, utc
from posthack_train import evaluate, next_batch_hash, registered, save_checkpoint, valid_checkpoint
from train_final_production import WeightedMixture, load_arrays

MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
STOP = False


def encode_example(tokenizer: Tokenizer, row: dict, context: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    prompt = tokenizer.encode(row["prompt"]).ids
    answer = tokenizer.encode(" " + row["answer"]).ids
    if not answer:
        raise ValueError("SFT answer tokenized to empty text")
    sequence = [1] + prompt + answer + [2]
    sequence = sequence[-(context + 1):]
    answer_length = min(len(answer) + 1, len(sequence) - 1)
    x = np.zeros(context, dtype=np.int64)
    y = np.zeros(context, dtype=np.int64)
    mask = np.zeros(context, dtype=np.bool_)
    used = len(sequence) - 1
    x[:used] = sequence[:-1]
    y[:used] = sequence[1:]
    mask[used-answer_length:used] = True
    return x, y, mask, len((row["prompt"] + row["answer"]).encode("utf-8"))


def load_examples(path: Path, tokenizer: Tokenizer, context: int) -> list[tuple]:
    return [encode_example(tokenizer, json.loads(line), context)
            for line in path.open(encoding="utf-8")]


def retention_objective(logits: torch.Tensor, targets: torch.Tensor,
                        completion_mask: torch.Tensor,
                        reference_replay_logits: torch.Tensor | None,
                        kl_weight: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """0.5 completion CE + 0.5 natural CE + 0.5 λ KL(ref || model)."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("logit/target shape mismatch")
    if logits.shape[0] != 2 * completion_mask.shape[0] or completion_mask.shape[1] != logits.shape[1]:
        raise ValueError("SFT and replay batch sizes or masks mismatch")
    if int(completion_mask.sum()) == 0:
        raise ValueError("no completion target tokens")
    half = completion_mask.shape[0]
    vocab = logits.shape[-1]
    completion_ce = F.cross_entropy(logits[:half].reshape(-1, vocab),
                                    targets[:half].reshape(-1), reduction="none").reshape_as(completion_mask)
    completion_loss = (completion_ce * completion_mask).sum() / completion_mask.sum()
    replay_loss = F.cross_entropy(logits[half:].reshape(-1, vocab),
                                  targets[half:].reshape(-1))
    kl = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if reference_replay_logits is not None:
        if reference_replay_logits.shape != logits[half:].shape:
            raise ValueError("reference replay logits shape mismatch")
        log_student = F.log_softmax(logits[half:], dim=-1)
        reference_probabilities = F.softmax(reference_replay_logits, dim=-1)
        kl = F.kl_div(log_student, reference_probabilities,
                      reduction="batchmean") / logits.shape[1]
    elif kl_weight:
        raise ValueError("reference logits are required for nonzero KL")
    total = .5 * completion_loss + .5 * replay_loss + .5 * kl_weight * kl
    return total, {"sft_loss": completion_loss, "replay_loss": replay_loss,
                   "reference_kl": kl}


def run(experiment_id: str, threads: int) -> dict:
    record = registered(experiment_id)
    if record.get("posttraining_objective") not in {"generic_replay", "aligned_replay_kl", "task_tuned_replay"}:
        raise ValueError("unregistered posttraining objective")
    state = json.loads(STATE.read_text())
    if time.time() >= state["archive_start_epoch"]:
        raise RuntimeError("archival reserve has started")
    parent_path = ROOT / record["parent_checkpoint_path"]
    if sha256(parent_path) != record["parent_checkpoint_sha256"]:
        raise RuntimeError("parent checkpoint hash differs")
    data_path = ROOT / record["posttraining_data_path"]
    if sha256(data_path) != record["posttraining_data_sha256"]:
        raise RuntimeError("SFT data hash differs")
    if sha256(MANIFEST) != record["sampling_manifest_sha256"] or sha256(TOKENIZER) != record["tokenizer_sha256"]:
        raise RuntimeError("replay corpus or tokenizer identity differs")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    random.seed(record["seed"])
    np.random.seed(record["seed"])
    torch.manual_seed(record["seed"])
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=False)
    cfg = LatticeConfig(**checkpoint["config"])
    variant = checkpoint.get("variant", "baseline")
    model = build_posthack_model(cfg, variant)
    model.load_state_dict(checkpoint["model"])
    if sum(p.numel() for p in model.parameters()) != record["parameter_count"]:
        raise RuntimeError("registered parameter count differs")
    reference = None
    kl_weight = float(record["reference_kl_weight"])
    if kl_weight:
        reference = build_posthack_model(cfg, variant)
        reference.load_state_dict(checkpoint["model"])
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
    optimizer, partition = make_optimizer(model, "adamw", record["learning_rate"],
                                          cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
    if partition != record["optimizer_partition"]:
        raise RuntimeError("posttraining optimizer partition differs")
    _, train, valid = load_arrays(MANIFEST)
    top = json.loads(MANIFEST.read_text())
    streams = {source: SourceStream(train[source], cfg.context_length, cfg.seed + 17 * index)
               for index, source in enumerate(train)}
    replay = WeightedMixture(streams, top["mixture_definition"], 16)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    examples = load_examples(data_path, tokenizer, cfg.context_length)
    if not examples:
        raise RuntimeError("no SFT examples")
    run_dir = RESEARCH / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    step = logical = processed = sft_cursor = 0
    sft_raw_bytes = 0
    prior_seconds = 0.0
    latest = run_dir / "latest.pt"
    resume_path = next((path for path in (latest, run_dir / "previous.pt")
                        if valid_checkpoint(path)), None)
    if resume_path is not None:
        if resume_path != latest and latest.exists():
            latest.unlink()
            latest.with_suffix(".sha256").unlink(missing_ok=True)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (checkpoint["experiment_id"] != experiment_id or
                checkpoint["parent_checkpoint_sha256"] != record["parent_checkpoint_sha256"] or
                checkpoint["posttraining_data_sha256"] != record["posttraining_data_sha256"]):
            raise RuntimeError("posttraining resume identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizers"])
        replay.load_state_dict(checkpoint["replay"])
        step, logical, processed = checkpoint["step"], checkpoint["logical_tokens"], checkpoint["processed_tokens"]
        sft_cursor, sft_raw_bytes = checkpoint["sft_cursor"], checkpoint["sft_raw_bytes"]
        prior_seconds = checkpoint["wall_seconds"]
        if checkpoint["next_batch_sha256"] != next_batch_hash(replay):
            raise RuntimeError("replay data order differs at resume")
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
    elif any((run_dir / name).exists() for name in ("latest.pt", "previous.pt", "result.json")):
        raise RuntimeError("existing posttraining state has no verified resumable checkpoint")
    started = time.perf_counter()
    set_active(experiment_id)
    event(experiment_id, "POSTTRAIN_START", objective=record["posttraining_objective"],
          parent_checkpoint_sha256=record["parent_checkpoint_sha256"])
    try:
        while logical < record["token_budget"]:
            if time.time() >= state["archive_start_epoch"]:
                globals()["STOP"] = True
            if STOP and step == 0:
                break
            x_sft, y_sft, masks, byte_counts = zip(*(examples[(sft_cursor + i) % len(examples)]
                                                       for i in range(16)))
            sft_cursor += 16
            x_replay, y_replay, labels = replay.batch()
            x = torch.from_numpy(np.concatenate([np.stack(x_sft), x_replay.astype(np.int64)]))
            y = torch.from_numpy(np.concatenate([np.stack(y_sft), y_replay.astype(np.int64)]))
            mask = torch.from_numpy(np.stack(masks))
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x)
            teacher_logits = None
            if reference is not None:
                with torch.no_grad():
                    teacher_logits, _ = reference(x[16:])
            loss, components = retention_objective(logits, y, mask,
                                                    teacher_logits, kl_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite posttraining objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            old_logical = logical
            logical += int(mask.sum()) + y_replay.size
            processed += x.numel()
            sft_raw_bytes += sum(byte_counts)
            step += 1
            elapsed = prior_seconds + time.perf_counter() - started
            append(run_dir / "metrics.jsonl", {"at": utc(), "step": step,
                   "logical_tokens": logical, "processed_tokens": processed,
                   "sft_raw_bytes": sft_raw_bytes, "wall_seconds": elapsed,
                   "train_loss": float(loss.detach()),
                   "sft_loss": float(components["sft_loss"].detach()),
                   "replay_loss": float(components["replay_loss"].detach()),
                   "reference_kl": float(components["reference_kl"].detach()),
                   "replay_source_labels": list(labels),
                   "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
            checkpoint_due = (STOP or logical >= record["token_budget"] or
                              logical // 250_000 != old_logical // 250_000)
            if checkpoint_due:
                source_losses, paired = evaluate(model, valid, cfg.context_length)
                append(run_dir / "evaluations.jsonl", {"at": utc(), "logical_tokens": logical,
                       "source_losses": source_losses, "paired_losses": paired,
                       "aggregate_loss": sum(source_losses.values()) / len(source_losses)})
                payload = {"schema": "posthack-posttraining-checkpoint-v1",
                           "experiment_id": experiment_id, "variant": variant,
                           "parent_checkpoint_sha256": record["parent_checkpoint_sha256"],
                           "posttraining_data_sha256": record["posttraining_data_sha256"],
                           "model": model.state_dict(), "optimizers": optimizer.state_dict(),
                           "config": cfg.to_dict(), "step": step,
                           "logical_tokens": logical, "processed_tokens": processed,
                           "sft_cursor": sft_cursor, "sft_raw_bytes": sft_raw_bytes,
                           "wall_seconds": elapsed, "replay": replay.state_dict(),
                           "next_batch_sha256": next_batch_hash(replay),
                           "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                           "torch_rng": torch.get_rng_state()}
                digest = save_checkpoint(run_dir, payload)
                event(experiment_id, "POSTTRAIN_CHECKPOINT", logical_tokens=logical,
                      checkpoint_sha256=digest, source_losses=source_losses)
            if STOP:
                break
        result = {"experiment_id": experiment_id,
                  "status": "COMPLETED" if logical >= record["token_budget"] else "STOPPED_SAFE",
                  "logical_tokens": logical, "processed_tokens": processed,
                  "sft_raw_bytes": sft_raw_bytes, "steps": step,
                  "wall_seconds": prior_seconds + time.perf_counter() - started,
                  "checkpoint": str(latest) if latest.exists() else None,
                  "checkpoint_sha256": sha256(latest) if latest.exists() else None,
                  "finished_utc": utc()}
        atomic_json(run_dir / "result.json", result)
        event(experiment_id, "POSTTRAIN_END", **{k: v for k, v in result.items() if k != "experiment_id"})
        return result
    finally:
        set_active(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("STOP", True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    print(json.dumps(run(args.experiment_id, args.threads)), flush=True)
