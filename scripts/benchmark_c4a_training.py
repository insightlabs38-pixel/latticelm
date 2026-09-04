"""Benchmark the complete frozen Co4-S DATA-C optimization step on CPU."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import statistics
import time

import numpy as np
import psutil
import torch

from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.final_data import PermutedBlocks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/data/phase7a"
FIELDS = ["thread_count", "interop_threads", "microbatch", "grad_accumulation",
          "effective_batch", "median_step_time", "mean_step_time", "std_step_time",
          "iqr_step_time", "tokens_per_second", "cpu_utilization", "peak_rss_bytes", "notes"]


def load(name: str) -> torch.Tensor:
    return torch.tensor(np.memmap(DATA / f"{name}.int32", mode="r", dtype=np.int32), dtype=torch.long)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run(threads: int, microbatch: int, warmup: int, measured: int) -> dict:
    config = LatticeConfig.from_json(ROOT / "configs/phase7c_final_10m.json")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(90210)
    baby = PermutedBlocks(load("babylm_train"), 128, 90221)
    web = PermutedBlocks(load("finewebedu_train"), 128, 90239)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay,
                                  betas=(config.adam_beta1, config.adam_beta2), eps=1e-8)

    def step() -> None:
        # Keep the selected source mixture exact for all tested multiples of four.
        baby_count = microbatch * 3 // 4
        pairs = [baby.one() for _ in range(baby_count)] + [web.one() for _ in range(microbatch - baby_count)]
        x = torch.stack([p[0] for p in pairs]); y = torch.stack([p[1] for p in pairs])
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    for _ in range(warmup):
        step()
    process = psutil.Process()
    cpu_before = sum(process.cpu_times()[:2])
    timings = []
    for _ in range(measured):
        started = time.perf_counter(); step(); timings.append(time.perf_counter() - started)
    cpu_after = sum(process.cpu_times()[:2])
    elapsed = sum(timings)
    median = statistics.median(timings)
    return {"thread_count": threads, "interop_threads": 1, "microbatch": microbatch,
            "grad_accumulation": 1, "effective_batch": microbatch,
            "median_step_time": median, "mean_step_time": statistics.mean(timings),
            "std_step_time": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "iqr_step_time": percentile(timings, .75) - percentile(timings, .25),
            "tokens_per_second": microbatch * 128 / median,
            "cpu_utilization": 100 * (cpu_after - cpu_before) / max(elapsed * threads, 1e-9),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "notes": ("frozen effective batch" if microbatch == 8 else
                      "throughput-only; effective batch differs and is not authorized for canonical training")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--microbatch", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/c4a_training_benchmark.csv")
    args = parser.parse_args()
    row = run(args.threads, args.microbatch, args.warmup, args.steps)
    exists = args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if not exists: writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
