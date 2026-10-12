"""Post-warmup C4A training throughput with the planned scientific batch."""
from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.data_d import SourceStream
from latticelm.final_recipe import make_optimizer
from latticelm.model import build_model
from posthack_campaign import RESEARCH, ROOT, event, init, record_calibration, register, set_active, sha256, utc
from train_final_production import WeightedMixture, load_arrays

MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"


def benchmark(path: Path, threads: int, top: dict, train: dict[str, list[np.ndarray]],
              warmup: int, measured: int) -> dict:
    cfg = LatticeConfig.from_json(path)
    torch.set_num_threads(threads)
    torch.manual_seed(cfg.seed)
    streams = {source: SourceStream(train[source], cfg.context_length, cfg.seed + 17 * index)
               for index, source in enumerate(train)}
    sampler = WeightedMixture(streams, top["mixture_definition"], cfg.batch_size)
    model = build_model(cfg)
    optimizer, partition = make_optimizer(model, cfg.optimizer, cfg.learning_rate,
                                          cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    regime = path.stem.split("_")[0]
    architecture = path.stem.split("_")[1]
    run_id = f"CAL-{regime}-{architecture}-{threads}t"
    step_tokens = cfg.batch_size * cfg.context_length
    register({"experiment_id": run_id, "hypothesis": "Measure actual post-warmup training throughput at the fixed batch semantics",
              "parent_control": None, "config": cfg.to_dict(), "config_sha256": sha256(path),
              "tokenizer_sha256": sha256(TOKENIZER), "sampling_manifest_sha256": sha256(MANIFEST),
              "code_commit": init()["source_commit"], "seed": cfg.seed,
              "parameter_count": params, "active_parameter_count": params,
              "context_length": cfg.context_length, "optimizer_partition": partition,
              "token_budget": step_tokens * (warmup + measured),
              "planned_evaluation_points": [step_tokens * (warmup + measured)],
              "promotion_criterion": "Calibration only; no model promotion",
              "abort_criterion": "Nonfinite loss or less than ten measured steps",
              "expected_wall_seconds": step_tokens * (warmup + measured) / 5000,
              "estimate_provenance": "Historical C4A small-model throughput anchor; measurement supersedes it",
              "threads": threads, "batch_semantics": {"batch_size": cfg.batch_size,
                                                   "context_length": cfg.context_length,
                                                   "effective_tokens_per_step": step_tokens}})
    set_active(run_id)
    event(run_id, "CALIBRATION_START", at_utc=utc())
    durations: list[float] = []
    losses: list[float] = []
    start = time.perf_counter()
    try:
        for index in range(warmup + measured):
            tick = time.perf_counter()
            x, y, _ = sampler.batch()
            x_tensor = torch.from_numpy(x.astype(np.int64, copy=False))
            y_tensor = torch.from_numpy(y.astype(np.int64, copy=False))
            optimizer.zero_grad()
            logits, _ = model(x_tensor)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y_tensor.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at calibration step {index + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if index >= warmup:
                durations.append(time.perf_counter() - tick)
                losses.append(float(loss.detach()))
        ordered = sorted(durations)
        tps = measured * step_tokens / sum(durations)
        result = {"experiment_id": run_id, "regime": regime, "architecture": architecture,
                  "threads": threads, "warmup_steps": warmup, "measured_steps": measured,
                  "step_tokens": step_tokens, "batch_size": cfg.batch_size,
                  "context_length": cfg.context_length, "parameter_count": params,
                  "median_step_seconds": statistics.median(durations),
                  "p10_step_seconds": ordered[max(0, math.floor(.1 * (len(ordered) - 1)))],
                  "p90_step_seconds": ordered[min(len(ordered) - 1, math.ceil(.9 * (len(ordered) - 1)))],
                  "tokens_per_second": tps, "hours_per_100m": 100_000_000 / tps / 3600,
                  "last_loss": losses[-1], "measured_wall_seconds": sum(durations),
                  "total_wall_seconds": time.perf_counter() - start,
                  "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "completed_utc": utc()}
        record_calibration(result)
        return result
    except Exception as exc:
        event(run_id, "CALIBRATION_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        set_active(None)
        del optimizer, model
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", nargs="+", default=["8m", "16m", "32m"])
    parser.add_argument("--architectures", nargs="+", default=["co4", "dense"])
    parser.add_argument("--threads", nargs="+", type=int, default=[12, 16])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=12)
    args = parser.parse_args()
    if args.measured_steps < 10 or args.warmup_steps < 1:
        parser.error("at least one warmup and ten measured steps required")
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    top, train, _ = load_arrays(MANIFEST)
    preprocessing = time.perf_counter() - started
    state = init()
    state["measured_preprocessing_seconds"]["load_certified_shards"] = preprocessing
    from posthack_campaign import STATE, atomic_json
    atomic_json(STATE, state)
    for regime in args.regimes:
        for arch in args.architectures:
            path = RESEARCH / "configs" / f"{regime}_{arch}.json"
            for threads in args.threads:
                key = f"{regime}/{arch}/{threads}"
                if key in json.loads(STATE.read_text())["throughput_calibration"]:
                    continue
                result = benchmark(path, threads, top, train, args.warmup_steps, args.measured_steps)
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
