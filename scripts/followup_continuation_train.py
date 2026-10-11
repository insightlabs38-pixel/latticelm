"""Explicit paired extension phases; parent state is preserved, LR restart is disclosed."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import resource
import signal
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.data_d import SourceStream
from latticelm.final_recipe import make_optimizer, wsd_lr_scale
from latticelm.posthack_variants import build_posthack_model
from latticelm.research_sampling import ResearchSourceStream, load_document_maps
from train_final_production import WeightedMixture, load_arrays, validation
from followup_common import (ROOT, RESEARCH, STATE, atomic_json, atomic_bytes, decision, object_hash,
                             process_identity, registered, sha256, sync_dir, utc, verified_checkpoint)

STOP = False
FLUSH = False


def build_model(record):
    cfg = LatticeConfig(**record["config"])
    if record.get("full_co4"):
        from latticelm.full_co4 import FullCo4LM
        return FullCo4LM(cfg, **record["full_co4"])
    return build_posthack_model(cfg, record.get("variant", "baseline"))


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def diagnostic_state(streams):
    return {name: {key: getattr(stream, key) for key in
            ("accepted", "rejected", "accepted_difficulty_sum", "selected_component_count", "progress")}
            for name, stream in streams.items() if isinstance(stream, ResearchSourceStream)}


def batch_hash(mixture):
    state = mixture.state_dict(); diagnostics = diagnostic_state(mixture.streams)
    x, y, _ = mixture.batch(); mixture.load_state_dict(state)
    for name, values in diagnostics.items():
        for key, value in values.items(): setattr(mixture.streams[name], key, value)
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()


def sampler(record, cfg, top, train):
    mode = record.get("sampling_mode", "baseline")
    if mode == "baseline":
        streams = {name: SourceStream(train[name], cfg.context_length, cfg.seed + 17 * index)
                   for index, name in enumerate(train)}
    else:
        index = ROOT / record["research_d_index_path"]
        if sha256(index) != record["research_d_index_sha256"]: raise ValueError("document index hash mismatch")
        maps = load_document_maps(index); paths = {source: [] for source in train}
        base = (ROOT / record["manifest_path"]).parent
        for item in top["shards"]:
            child = json.loads((base / item["manifest_path"]).read_text()); paths[child["source"]].append(child["path"])
        streams = {name: ResearchSourceStream(train[name], cfg.context_length, cfg.seed + 17 * position,
                   paths[name], maps, name, mode, **record.get("sampling_rule", {})) for position, name in enumerate(train)}
    return WeightedMixture(streams, record.get("mixture_weights", top["mixture_definition"]), cfg.batch_size)


def evaluate_sources(model, valid, context):
    losses, paired = {}, {}; training = model.training
    try:
        for source, arrays in valid.items():
            losses[source], paired[source] = validation(model, np.concatenate(arrays), context, batches=8)
    finally: model.train(training)
    return losses, paired


def append_metric(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())


def publish_checkpoint(run, payload, milestone=False):
    """Unique immutable files plus one atomic manifest avoid checksum-pair tears."""
    path = run / f"checkpoint-{payload['step']:08d}-{uuid.uuid4().hex[:8]}.pt"
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    digest = sha256(temp); os.replace(temp, path); sync_dir(run)
    item = {"file": path.name, "sha256": digest, "size_bytes": path.stat().st_size,
            "step": payload["step"], "logical_tokens": payload["logical_tokens"],
            "processed_tokens": payload["processed_tokens"], "wall_seconds": payload["wall_seconds"], "at": utc()}
    mp = run / "checkpoint_manifest.json"
    previous = json.loads(mp.read_text()) if mp.exists() else {"rolling": [], "milestones": []}
    rolling = [item] + previous["rolling"][:1]
    milestones = previous["milestones"] + ([item] if milestone else [])
    atomic_json(mp, {"schema": "followup-checkpoint-manifest-v1", "rolling": rolling, "milestones": milestones})
    retained = {i["file"] for i in rolling + milestones}
    # Only redundant, committed rolling files are eligible for removal.
    for old in previous["rolling"]:
        if old["file"] not in retained: (run / old["file"]).unlink(missing_ok=True)
    sync_dir(run)
    return item


def train(args):
    global STOP, FLUSH
    record = registered(args.experiment_id); cfg = LatticeConfig(**record["config"])
    run = RESEARCH / "runs" / args.experiment_id; run.mkdir(parents=True, exist_ok=True)
    identity = {**process_identity(os.getpid()), "launch_utc": utc(), "experiment_id": args.experiment_id,
                "working_directory": str(ROOT), "expected_checkpoint_path": str(run / "checkpoint_manifest.json")}
    attempt = uuid.uuid4().hex
    def heartbeat(step=0, tokens=0, phase="INITIALIZING"):
        atomic_json(run / "heartbeat.json", {**identity, "heartbeat_utc": utc(), "heartbeat_epoch": time.time(),
                   "step": step, "logical_tokens": tokens, "phase": phase, "attempt": attempt})
    heartbeat()
    for rel, expected in record.get("source_hashes", {}).items():
        if sha256(ROOT / rel) != expected: raise RuntimeError("registered scientific source changed: " + rel)
    manifest = ROOT / record["manifest_path"]; tokenizer = ROOT / record["tokenizer_path"]
    if sha256(manifest) != record["sampling_manifest_sha256"] or sha256(tokenizer) != record["tokenizer_sha256"]:
        raise ValueError("data/tokenizer hash mismatch")
    torch.set_num_threads(record.get("threads", 16)); torch.set_num_interop_threads(1)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    top, arrays, valid = load_arrays(manifest)
    mixture = sampler(record, cfg, top, arrays)
    if list(arrays) != record["source_order"]: raise ValueError("source ordering mismatch")
    model = build_model(record)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != record["parameter_count"]: raise RuntimeError("parameter count mismatch")
    optimizer, partition = make_optimizer(model, cfg.optimizer, cfg.learning_rate, cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
    if partition != record["optimizer_partition"]: raise RuntimeError("optimizer partition mismatch")
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    if record.get("latent_lr_multiplier", 1) != 1:
        if cfg.optimizer != "muon_hybrid": raise RuntimeError("latent LR multiplier needs an explicit partitioned optimizer")
        # Latents and norms share AdamW in the inherited recipe; do not change all AdamW parameters implicitly.
        raise RuntimeError("latent-specific optimizer groups must be explicitly implemented before registering a multiplier")
    steps = tokens = processed = 0; prior_wall = 0.; origin_commit = record["code_commit"]
    current = verified_checkpoint(run)
    external_parent = False
    if not current and record.get("continuation_checkpoint"):
        parent_path = ROOT / record["continuation_checkpoint"]
        if sha256(parent_path) != record["continuation_checkpoint_sha256"]: raise ValueError("continuation parent hash mismatch")
        ck = torch.load(parent_path, map_location="cpu", weights_only=False)
        if ck["logical_tokens"] != record["initial_logical_tokens"] or ck["step"] != record["schedule_start_step"]: raise ValueError("continuation parent position mismatch")
        if ck.get("full_co4") != record.get("full_co4") or ck.get("variant", "baseline") != record.get("variant", "baseline"): raise ValueError("continuation operator mismatch")
        if ck["manifest_sha256"] != record["sampling_manifest_sha256"] or ck["tokenizer_sha256"] != record["tokenizer_sha256"]: raise ValueError("continuation data/tokenizer mismatch")
        if ck.get('schema')=='posthack-checkpoint-v1':
            from followup_historical import verify_historical_control
            verify_historical_control(ck['experiment_id'],ck,record['continuation_checkpoint_sha256'])
        for rel,digest in ck.get("source_hashes",{}).items():
            if record["source_hashes"].get(rel) != digest: raise RuntimeError("continuation scientific source changed: "+rel)
        decision("EXPLICIT_EXTENSION_PHASE_START",experiment_id=args.experiment_id,parent_sha256=record["continuation_checkpoint_sha256"],
                 parent_step=ck["step"],parent_tokens=ck["logical_tokens"],lr_policy=record["schedule_phase"])
        external_parent = True
        current = {"file": str(parent_path), "sha256": record["continuation_checkpoint_sha256"]}
    if current:
        if not external_parent: ck = torch.load(run / current["file"], map_location="cpu", weights_only=False)
        allowed_schemas=('followup-checkpoint-v1','posthack-checkpoint-v1') if external_parent else ('followup-checkpoint-v1',)
        if (ck["schema"] not in allowed_schemas or (not external_parent and ck["registration_sha256"] != object_hash(record))
            or json.loads(json.dumps(ck["config"])) != record["config"]): raise RuntimeError("resume identity mismatch")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizers"])
        mixture.load_state_dict(ck["mixture"]); mixture.cycle = tuple(ck["sampling_cycle"])
        mixture.annealed = ck["sampling_annealed"]
        for name, values in ck["sampler_diagnostics"].items():
            for key, value in values.items(): setattr(mixture.streams[name], key, value)
        random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"]); torch.set_rng_state(ck["torch_rng"])
        if batch_hash(mixture) != ck["next_batch_sha256"]: raise ValueError("resume data-order mismatch")
        steps, tokens, processed, prior_wall = ck["step"], ck["logical_tokens"], ck["processed_tokens"], ck["wall_seconds"]
        if processed != steps * cfg.batch_size * cfg.context_length: raise ValueError("resume token accounting mismatch")
        decision("EXACT_RESUME", experiment_id=args.experiment_id, step=steps, logical_tokens=tokens, checkpoint_sha256=current["sha256"])
    else:
        if (run / "train_receipt.json").exists(): raise RuntimeError("receipt without resumable checkpoint")
        if any(o["state"] for o in optimizer.state_dict()): raise RuntimeError("optimizer initialized with hidden state")
        initial_hash = batch_hash(mixture)
        if record.get("initial_batch_sha256") and initial_hash != record["initial_batch_sha256"]:
            raise ValueError("initial sampler batch mismatch")
        atomic_json(run / "initialization_receipt.json", {"at": utc(), "seed": cfg.seed, "fresh_initialization": True,
                    "optimizer_states_empty": True, "model_sha256": model_hash(model), "initial_batch_sha256": initial_hash,
                    "sampler_state": mixture.state_dict(), "parameter_count": count, "base_lrs": base_lrs,
                    "registration_sha256": object_hash(record), "config_sha256": object_hash(record["config"])})
    total = record["token_budget"]; step_tokens = cfg.batch_size * cfg.context_length
    schedule_total = math.ceil(record.get("schedule_token_budget", total) / step_tokens)
    started = time.perf_counter(); last_beat = 0.; last_checkpoint = time.monotonic()
    metrics = run / f"metrics-{attempt}.jsonl"
    atomic_json(run / "attempt.json", {**identity, "attempt": attempt, "start_step": steps, "start_tokens": tokens, "at": utc()})
    last_saved = steps if current else -1
    def checkpoint(is_milestone=False):
        nonlocal last_saved, last_checkpoint
        heartbeat(steps, tokens, "CHECKPOINTING")
        # Set the sampler progress used for the NEXT batch, not the prior batch.
        for stream in mixture.streams.values():
            if isinstance(stream, ResearchSourceStream): stream.set_progress(tokens / total)
        losses, paired = evaluate_sources(model, valid, cfg.context_length)
        elapsed = prior_wall + time.perf_counter() - started
        payload = {"schema": "followup-checkpoint-v1", "experiment_id": args.experiment_id,
                   "registration_sha256": object_hash(record), "config_sha256": object_hash(record["config"]),
                   "source_commit": origin_commit, "source_hashes": record.get("source_hashes", {}),
                   "model": model.state_dict(), "optimizers": optimizer.state_dict(), "optimizer_partition": partition,
                   "config": cfg.to_dict(), "variant": record.get("variant", "baseline"), "full_co4": record.get("full_co4"),
                   "step": steps, "logical_tokens": tokens, "processed_tokens": processed, "target_tokens": total,
                   "schedule_total_steps": schedule_total, "base_lrs": base_lrs,
                   "schedule_start_step": record.get("schedule_start_step",0), "schedule_phase": record.get("schedule_phase","explicit-extension"),
                   "continuation_checkpoint_sha256": record.get("continuation_checkpoint_sha256"),
                   "scaler": None, "gradient_accumulation_position": 0, "precision": "float32",
                   "wall_seconds": elapsed, "mixture": mixture.state_dict(), "sampling_cycle": list(mixture.cycle),
                   "sampling_annealed": bool(getattr(mixture, "annealed", False)), "sampler_diagnostics": diagnostic_state(mixture.streams),
                   "next_batch_sha256": batch_hash(mixture), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                   "torch_rng": torch.get_rng_state(), "manifest_sha256": record["sampling_manifest_sha256"],
                   "tokenizer_sha256": record["tokenizer_sha256"], "attempt": attempt}
        if record.get("full_co4"): payload["architecture_state"] = model.architecture_state()
        before = time.perf_counter(); item = publish_checkpoint(run, payload, is_milestone)
        evaluation = {"at": utc(), "step": steps, "logical_tokens": tokens, "source_losses": losses,
                      "paired_losses": paired, "checkpoint_sha256": item["sha256"],
                      "frozen_mixture_weighted_loss": sum(top["mixture_definition"][s] * losses[s] for s in losses),
                      "checkpoint_write_seconds": time.perf_counter()-before}
        atomic_json(run / "latest_validation.json", evaluation)
        append_metric(run / "validation_history.jsonl", evaluation)
        last_saved = steps; last_checkpoint = time.monotonic(); heartbeat(steps, tokens, "TRAINING")
        return item
    if not current or external_parent: current = checkpoint()
    # Test-only rendezvous permits deterministic SIGKILL immediately after a committed checkpoint.
    pause_step = record.get("test_pause_after_checkpoint_step") if steps == record.get("schedule_start_step", 0) else None
    while tokens < total:
        if STOP or time.time() >= args.stop_epoch: break
        for stream in mixture.streams.values():
            if isinstance(stream, ResearchSourceStream): stream.set_progress(tokens / total)
        anneal = record.get("anneal_rule")
        if anneal and tokens / total >= anneal["at_fraction"] and not getattr(mixture, "annealed", False):
            mixture.cycle = WeightedMixture(mixture.streams, anneal["mixture_weights"], cfg.batch_size).cycle; mixture.annealed = True
            decision("DATA_ANNEAL_STARTED", experiment_id=args.experiment_id, logical_tokens=tokens, rule=anneal)
        step = steps + 1
        scale = wsd_lr_scale(step-record.get("schedule_start_step",0), schedule_total, cfg.warmup_fraction, cfg.stable_fraction, cfg.decay_fraction)
        for group, base in zip(optimizer.param_groups, base_lrs): group["lr"] = base * scale
        x, y, labels = mixture.batch(); increment = min(step_tokens, total - tokens)
        optimizer.zero_grad()
        logits, _ = model(torch.from_numpy(x.astype(np.int64, copy=False)))
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size)[:increment], torch.from_numpy(y.astype(np.int64, copy=False)).reshape(-1)[:increment])
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite training loss")
        loss.backward(); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(grad_norm): raise FloatingPointError("nonfinite gradient")
        optimizer.step(); old_tokens = tokens; tokens += increment; processed += step_tokens; steps = step
        elapsed = prior_wall + time.perf_counter() - started
        append_metric(metrics, {"at": utc(), "attempt": attempt, "step": steps, "logical_tokens": tokens,
                               "processed_tokens": processed, "train_loss": float(loss.detach()), "gradient_norm": float(grad_norm),
                               "lr": optimizer.param_groups[0]["lr"], "wall_seconds": elapsed,
                               "tokens_per_second": tokens / max(elapsed, 1e-9), "source_labels": list(labels)})
        if time.monotonic() - last_beat >= 15:
            heartbeat(steps, tokens, "TRAINING"); last_beat = time.monotonic()
        milestone = any(old_tokens < t <= tokens for t in record.get("milestone_tokens", []))
        due = (tokens // record["checkpoint_tokens"] != old_tokens // record["checkpoint_tokens"]
               or time.monotonic()-last_checkpoint >= record.get("max_checkpoint_seconds", 1100)
               or milestone or FLUSH or STOP or tokens == total)
        if due:
            current = checkpoint(milestone); FLUSH = False
            if os.statvfs(run).f_bavail * os.statvfs(run).f_frsize < record.get("disk_floor_bytes", 5_000_000_000):
                raise OSError("disk free-space safety floor")
            if pause_step is not None and steps >= pause_step:
                atomic_json(run / "test_checkpoint_ready.json", current)
                while not STOP: time.sleep(.05)
    if last_saved != steps: current = checkpoint()
    receipt = {"schema": "followup-train-receipt-v1", "experiment_id": args.experiment_id,
               "status": "TRAINED" if tokens == total else "INTERRUPTED_INFRA", "step": steps,
               "logical_tokens": tokens, "processed_tokens": processed, "checkpoint_sha256": current["sha256"],
               "checkpoint_file": current["file"], "wall_seconds": prior_wall + time.perf_counter()-started,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, "at": utc()}
    atomic_json(run / "train_receipt.json", receipt); heartbeat(steps, tokens, receipt["status"])
    return receipt


def main():
    p = argparse.ArgumentParser(); p.add_argument("--experiment-id", required=True); p.add_argument("--stop-epoch", type=float, required=True)
    args = p.parse_args(); run = RESEARCH / "runs" / args.experiment_id; run.mkdir(parents=True, exist_ok=True)
    def stop(*_):
        global STOP; STOP = True
    def flush(*_):
        global FLUSH; FLUSH = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGUSR1, flush)
    with (run / ".trainer.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("duplicate trainer refused")
        (run / "failure.json").unlink(missing_ok=True)
        try: print(json.dumps(train(args)), flush=True)
        except Exception as exc:
            classification = ("FAILED_DATA" if isinstance(exc, ValueError) else
                              "FAILED_SCIENCE" if isinstance(exc, FloatingPointError) else "FAILED_CODE")
            atomic_json(run / "failure.json", {"at": utc(), "classification": classification,
                       "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__": main()
