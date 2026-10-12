"""Reproducible, non-destructive CPU microbenchmarks for the cloud host."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def timed(fn, warmup: int = 5, repeats: int = 15) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p25_ms": statistics.quantiles(samples, n=4)[0],
        "p75_ms": statistics.quantiles(samples, n=4)[2],
    }


def command(command: str) -> str:
    return subprocess.run(command, shell=True, text=True, capture_output=True).stdout.strip()


def benchmark(quick: bool) -> list[dict]:
    rows: list[dict] = []
    repeats = 7 if quick else 15
    exposed = os.cpu_count() or 1
    thread_counts = sorted(set((1, 2, 4, exposed)))
    for threads in thread_counts:
        torch.set_num_threads(threads)
        for m, k, n in ((256, 256, 256), (256, 768, 256), (512, 512, 512), (1024, 256, 768), (1, 256, 64), (128, 256, 64)):
            a, b = torch.randn(m, k), torch.randn(k, n)
            rows.append({"category": "gemm", "case": f"{m}x{k}@{k}x{n}", "threads": threads, **timed(lambda: a @ b, repeats=repeats)})
        x, weight = torch.randn(8, 128, 256), torch.ones(256)
        rows.append({"category": "rmsnorm", "case": "8x128x256", "threads": threads,
                     **timed(lambda: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight, repeats=repeats)})
        for seq in (64, 128, 256):
            q = torch.randn(2, 4, seq, 64)
            rows.append({"category": "attention", "case": f"b2_h4_t{seq}_d64", "threads": threads,
                         **timed(lambda: F.scaled_dot_product_attention(q, q, q, is_causal=True), repeats=repeats)})
    torch.set_num_threads(min(4, exposed))
    for params in (500_000, 2_000_000, 8_000_000, 16_000_000):
        dim, count = 64, params // 64
        table = torch.randn(count, dim)
        random_ids = torch.randint(count, (8192,))
        local_ids = torch.arange(8192) % count
        for locality, ids in (("random", random_ids), ("local", local_ids)):
            rows.append({"category": "gather", "case": f"{params}_{locality}", "threads": torch.get_num_threads(),
                         **timed(lambda ids=ids, table=table: F.embedding(ids, table), repeats=repeats)})
    # Large tensor triad: report effective bytes moved per second.
    elements = 8_000_000 if quick else 16_000_000
    a, b, c = torch.empty(elements), torch.ones(elements), torch.full((elements,), 2.0)
    result = timed(lambda: torch.add(b, c, out=a), repeats=repeats)
    result["bandwidth_gbps"] = elements * 4 * 3 / (result["median_ms"] / 1000) / 1e9
    rows.append({"category": "memory", "case": "fp32_add_triad", "threads": torch.get_num_threads(), **result})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    rows = benchmark(args.quick)
    (artifacts / "cpu_microbenchmarks.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (artifacts / "cpu_microbenchmarks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"cases": len(rows), "output": str(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
