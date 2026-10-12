from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from .kernels.reference import conditional_memory_gather, residual_rmsnorm

ROOT = Path(__file__).resolve().parents[2]


def measure(fn, iterations: int = 50, warmup: int = 10) -> dict:
    for _ in range(warmup): fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": round(statistics.median(samples), 5), "p25_ms": round(statistics.quantiles(samples, n=4)[0], 5), "p75_ms": round(statistics.quantiles(samples, n=4)[2], 5)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--threads", type=int, default=1); args = parser.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(1337)
    x, residual, weight = torch.randn(8, 128, 256), torch.randn(8, 128, 256), torch.ones(256)
    tables = [torch.randn(1024, 64) for _ in range(3)]
    ids = [torch.randint(0, 1024, (8, 128)) for _ in range(3)]
    result = {"threads": args.threads, "residual_rmsnorm_reference": measure(lambda: residual_rmsnorm(x, residual, weight)), "memory_gather_reference": measure(lambda: conditional_memory_gather(*tables, *ids))}
    try:
        import triton  # noqa: F401
        result["triton_cpu"] = "imported; no training-compatible kernel enabled"
    except ImportError:
        result["triton_cpu"] = "unavailable: Python package triton is not installed"
    target = ROOT / "artifacts" / "kernel_benchmarks.jsonl"; target.parent.mkdir(exist_ok=True)
    with target.open("a", encoding="utf-8") as handle: handle.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
