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
from latticelm.posthack_variants import build_posthack_model
from posthack_campaign import RESEARCH, ROOT, STATE, append, atomic_json, event, set_active, sha256, utc
from train_final_production import WeightedMixture, load_arrays, validation
from latticelm.research_sampling import ResearchSourceStream, load_document_maps

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


def sampler_diagnostics(streams: dict) -> dict:
    return {name: {key: getattr(stream, key) for key in (
        "accepted", "rejected", "accepted_difficulty_sum", "selected_component_count")}
        for name, stream in streams.items() if isinstance(stream, ResearchSourceStream)}


def next_batch_hash(mixture: WeightedMixture) -> str:
    state = mixture.state_dict()
    diagnostics = sampler_diagnostics(mixture.streams)
    x, y, _ = mixture.batch()
    mixture.load_state_dict(state)
    for name, values in diagnostics.items():
        for key, value in values.items():
            setattr(mixture.streams[name], key, value)
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()


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
    manifest = ROOT / record.get("manifest_path", str(MANIFEST.relative_to(ROOT)))
    tokenizer = ROOT / record.get("tokenizer_path", str(TOKENIZER.relative_to(ROOT)))
    if sha256(manifest) != record["sampling_manifest_sha256"] or sha256(tokenizer) != record["tokenizer_sha256"]:
        raise RuntimeError("registered data or tokenizer identity differs")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    top, train, valid = load_arrays(manifest)
    sampling_mode = record.get("sampling_mode", "baseline")
    if sampling_mode == "baseline":
        streams = {source: SourceStream(train[source], cfg.context_length, cfg.seed + 17 * index)
                   for index, source in enumerate(train)}
    else:
        index_path = ROOT / record["research_d_index_path"]
        if sha256(index_path) != record["research_d_index_sha256"]:
            raise RuntimeError("Research-D index identity differs from registration")
        maps = load_document_maps(index_path)
        base = manifest.parent
        shard_paths = {source: [] for source in train}
        for reference in top["shards"]:
            metadata = json.loads((base / reference["manifest_path"]).read_text())
            shard_paths[metadata["source"]].append(metadata["path"])
        rule = record.get("sampling_rule", {})
        streams = {source: ResearchSourceStream(train[source], cfg.context_length,
                   cfg.seed + 17 * position, shard_paths[source], maps, source, sampling_mode,
                   curriculum_fraction=rule.get("curriculum_fraction", .25),
                   curriculum_strength=rule.get("curriculum_strength", .8),
                   quality_floor=rule.get("quality_floor", .7),
                   capability_floor=rule.get("capability_floor", .9),
                   capability_fraction=rule.get("capability_fraction", .125))
                   for position, source in enumerate(train)}
    mixture_weights = record.get("mixture_weights", top["mixture_definition"])
    if set(mixture_weights) != set(top["mixture_definition"]) or abs(sum(mixture_weights.values()) - 1) > 1e-9:
        raise RuntimeError("registered source mixture is invalid")
    mixture = WeightedMixture(streams, mixture_weights, cfg.batch_size)
    variant = record.get("variant", "baseline")
    model = build_posthack_model(cfg, variant)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != record["parameter_count"]:
        raise RuntimeError("parameter count differs from registration")
    if (model.active_parameter_count() if hasattr(model, "active_parameter_count") else parameter_count) != record["active_parameter_count"]:
        raise RuntimeError("active parameter count differs from registration")
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
        if checkpoint_path.name == "previous.pt" and (run_dir / "latest.pt").exists():
            (run_dir / "latest.pt").unlink()
            (run_dir / "latest.sha256").unlink(missing_ok=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint["experiment_id"] != args.experiment_id or
                checkpoint["target_tokens"] != args.target_tokens or
                checkpoint.get("variant", "baseline") != variant):
            raise RuntimeError("resume identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizers"])
        mixture.load_state_dict(checkpoint["mixture"])
        mixture.cycle = tuple(checkpoint.get("sampling_cycle", mixture.cycle))
        mixture.annealed = bool(checkpoint.get("sampling_annealed", False))
        moe_transitioned = bool(checkpoint.get("moe_transitioned", False))
        step, tokens = checkpoint["step"], checkpoint["logical_tokens"]
        for stream in streams.values():
            if isinstance(stream, ResearchSourceStream):
                stream.set_progress(checkpoint.get("sampler_progress", tokens / args.target_tokens))
        for name, values in checkpoint.get("sampler_diagnostics", {}).items():
            for key, value in values.items():
                setattr(streams[name], key, value)
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if next_batch_hash(mixture) != checkpoint["next_batch_sha256"]:
            raise RuntimeError("resume data-order mismatch")
        processed_tokens = checkpoint["processed_tokens"]
        prior_seconds = checkpoint["wall_seconds"]
        event(args.experiment_id, "RESUMED", step=step, logical_tokens=tokens, checkpoint_sha256=sha256(checkpoint_path))
    elif any((run_dir / p).exists() for p in ("latest.pt", "previous.pt", "result.json")):
        raise RuntimeError("fresh run refuses to overwrite existing state")
    else:
        moe_transitioned = False
    step_tokens = cfg.batch_size * cfg.context_length
    total_steps = math.ceil(args.target_tokens / step_tokens)
    started = time.perf_counter()
    set_active(args.experiment_id)
    event(args.experiment_id, "TRAIN_START", target_tokens=args.target_tokens,
          variant=variant, trainer_sha256=sha256(Path(__file__)),
          manifest_sha256=record["sampling_manifest_sha256"], mixture_weights=mixture_weights,
          threads=args.threads, batch_size=cfg.batch_size, context_length=cfg.context_length)
    try:
        while tokens < args.target_tokens:
            progress = tokens / args.target_tokens
            for stream in streams.values():
                if isinstance(stream, ResearchSourceStream):
                    stream.set_progress(progress)
            anneal = record.get("anneal_rule")
            if anneal and progress >= anneal["at_fraction"] and not getattr(mixture, "annealed", False):
                mixture.cycle = WeightedMixture(streams, anneal["mixture_weights"], cfg.batch_size).cycle
                mixture.annealed = True
                event(args.experiment_id, "DATA_ANNEAL_START", logical_tokens=tokens,
                      mixture_weights=anneal["mixture_weights"], quality_floor=anneal["quality_floor"])
            moe_rule = record.get("moe_rule")
            if moe_rule and progress >= moe_rule["transition_at_fraction"] and not moe_transitioned:
                probe = (torch.arange(16, dtype=torch.long) % cfg.vocab_size)[None, :]
                with torch.inference_mode():
                    before = model(probe)[0]
                birth_record = None
                if moe_rule["birth_strategy"] != "none":
                    birth_record = model.birth(moe_rule["birth_strategy"])
                with torch.inference_mode():
                    after = model(probe)[0]
                parity = float((before - after).abs().max())
                if moe_rule["birth_strategy"] == "clone" and parity > 1e-6:
                    raise RuntimeError(f"expert birth parity failure: {parity}")
                mixture.cycle = WeightedMixture(streams, moe_rule["later_mixture"], cfg.batch_size).cycle
                moe_transitioned = True
                event(args.experiment_id, "EXPERT_CURRICULUM_TRANSITION",
                      logical_tokens=tokens, birth=birth_record,
                      immediate_max_logit_change=parity,
                      later_mixture=moe_rule["later_mixture"])
            if moe_rule:
                model.set_catchup(moe_transitioned and moe_rule["birth_strategy"] == "clone"
                                  and moe_rule.get("specialty_catchup_tokens", 0) > 0
                                  and tokens < int(moe_rule["transition_at_fraction"] * args.target_tokens)
                                  + moe_rule.get("specialty_catchup_tokens", 0))
            if variant == "progressive_depth":
                for growth in record.get("growth_schedule", []):
                    if tokens >= growth["at_tokens"] and int(model.active_depth) < growth["depth"]:
                        probe = (torch.arange(16, dtype=torch.long) % cfg.vocab_size)[None, :]
                        with torch.inference_mode():
                            before = model(probe)[0]
                        previous_depth = int(model.active_depth)
                        model.grow(int(growth["depth"]))
                        with torch.inference_mode():
                            after = model(probe)[0]
                        maximum_change = float((before - after).abs().max())
                        if maximum_change > 1e-6:
                            raise RuntimeError(f"growth parity failure: {maximum_change}")
                        event(args.experiment_id, "DEPTH_GROWTH", logical_tokens=tokens,
                              old_depth=previous_depth, new_depth=int(model.active_depth),
                              max_immediate_logit_change=maximum_change,
                              optimizer_policy="old states retained; dormant new blocks begin with empty state")
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
            if moe_rule:
                model.set_domains(labels)
            optimizer.zero_grad()
            logits, _ = model(x_tensor)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size)[:increment], y_tensor.reshape(-1)[:increment])
            if moe_rule:
                lm_loss = loss
                loss = loss + model.auxiliary_loss()
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
                                  "active_depth": int(model.active_depth) if variant == "progressive_depth" else cfg.n_layers,
                                  "sampler_diagnostics": sampler_diagnostics(streams) if sampling_mode != "baseline" else None,
                                  "source_labels": list(labels)})
            if moe_rule:
                # Compact per-step routing counts preserve the attribution record.
                append(run_dir / "routing.jsonl", {"at": utc(), "step": step,
                       "logical_tokens": tokens, "lm_loss": float(lm_loss.detach()),
                       "routing": model.routing_diagnostics(),
                       "catchup_active": any(block.ffn.specialty_catchup for block in model.blocks)})
            extra_milestones = record.get("extra_checkpoint_tokens", {})
            crossed_milestone = any(old_tokens < int(target) <= tokens for target in extra_milestones.values())
            checkpoint_due = (STOP or tokens >= args.target_tokens or
                              tokens // args.checkpoint_tokens != old_tokens // args.checkpoint_tokens or
                              crossed_milestone)
            if checkpoint_due:
                eval_start = time.perf_counter()
                if moe_rule:
                    model.set_domains(None)
                source_losses, paired = evaluate(model, valid, cfg.context_length)
                eval_seconds = time.perf_counter() - eval_start
                append(run_dir / "evaluations.jsonl", {"at": utc(), "logical_tokens": tokens,
                       "source_losses": source_losses, "paired_losses": paired,
                       "aggregate_loss": sum(source_losses.values()) / len(source_losses),
                       "evaluation_seconds": eval_seconds})
                payload = {"schema": "posthack-checkpoint-v1", "experiment_id": args.experiment_id,
                           "variant": variant,
                           "trainer_sha256": sha256(Path(__file__)),
                           "model": model.state_dict(), "optimizers": optimizer.state_dict(),
                           "config": cfg.to_dict(), "step": step, "logical_tokens": tokens,
                           "processed_tokens": processed_tokens, "target_tokens": args.target_tokens,
                           "wall_seconds": elapsed, "mixture": mixture.state_dict(),
                           "sampling_cycle": list(mixture.cycle),
                           "sampling_annealed": bool(getattr(mixture, "annealed", False)),
                           "moe_transitioned": moe_transitioned,
                           "sampler_progress": progress,
                           "sampler_diagnostics": sampler_diagnostics(streams),
                           "next_batch_sha256": next_batch_hash(mixture),
                           "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                           "torch_rng": torch.get_rng_state(), "optimizer_partition": partition,
                           "manifest_sha256": record["sampling_manifest_sha256"],
                           "tokenizer_sha256": record["tokenizer_sha256"]}
                digest = save_checkpoint(run_dir, payload)
                for milestone_name, target in extra_milestones.items():
                    if old_tokens < int(target) <= tokens:
                        milestone_dir = run_dir / "milestones"
                        milestone_dir.mkdir(exist_ok=True)
                        destination = milestone_dir / f"{milestone_name}.pt"
                        if destination.exists():
                            raise RuntimeError(f"milestone already exists: {destination}")
                        os.link(run_dir / "latest.pt", destination)
                        destination.with_suffix(".sha256").write_text(digest + "\n")
                        event(args.experiment_id, "MILESTONE_PRESERVED",
                              name=milestone_name, logical_tokens=tokens,
                              checkpoint_sha256=digest)
                event(args.experiment_id, "CHECKPOINT", logical_tokens=tokens, processed_tokens=processed_tokens,
                      train_loss=float(loss.detach()), source_losses=source_losses,
                      checkpoint_sha256=digest, rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      evaluation_seconds=eval_seconds)
                if os.statvfs(run_dir).f_bavail * os.statvfs(run_dir).f_frsize < 8_000_000_000:
                    raise RuntimeError("disk safety floor crossed")
            if STOP:
                break
        result = {"experiment_id": args.experiment_id, "variant": variant,
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
