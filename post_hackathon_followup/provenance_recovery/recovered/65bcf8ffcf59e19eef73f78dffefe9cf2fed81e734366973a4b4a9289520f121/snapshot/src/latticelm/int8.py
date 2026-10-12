from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn

from .kernels.reference import conditional_memory_gather

ROOT = Path(__file__).resolve().parents[2]


class RowwiseInt8Memory:
    """Symmetric per-row INT8 storage for conditional-memory inference."""
    def __init__(self, tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        self.quantized: list[torch.Tensor] = []
        self.scales: list[torch.Tensor] = []
        for table in tables:
            scale = table.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            self.quantized.append(torch.round(table / scale[:, None]).clamp(-127, 127).to(torch.int8))
            self.scales.append(scale)

    def gather(self, id2: torch.Tensor, id3: torch.Tensor, id4: torch.Tensor) -> torch.Tensor:
        output = 0
        for table, scale, ids in zip(self.quantized, self.scales, (id2, id3, id4)):
            output = output + table[ids].float() * scale[ids][..., None]
        return output


def measure(fn, warmup: int = 20, iterations: int = 100) -> float:
    for _ in range(warmup): fn()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns() - start) / 1e6)
    return round(statistics.median(samples), 6)


def run(checkpoint: str, threads: int = 4) -> dict:
    torch.set_num_threads(threads); torch.manual_seed(1337)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    tables = tuple(state[f"memory.{name}.weight"].float().contiguous() for name in ("bigram", "trigram", "fourgram"))
    projection_weight = state["memory.projection.weight"].float().contiguous()
    slots, dim = tables[0].shape
    ids = tuple(torch.randint(0, slots, (8, 128)) for _ in range(3))
    fp32 = conditional_memory_gather(*tables, *ids)
    packed = RowwiseInt8Memory(tables)
    int8 = packed.gather(*ids)
    projection = nn.Linear(dim, projection_weight.size(0), bias=False); projection.weight.data.copy_(projection_weight)
    dynamic = torch.ao.quantization.quantize_dynamic(nn.Sequential(projection), {nn.Linear}, dtype=torch.qint8).eval()
    fp_projected, int8_projected = projection(fp32), dynamic(int8)
    result = {
        "experiment": "int8_memory_inference",
        "threads": threads,
        "shape": [8, 128, dim], "slots_per_table": slots,
        "fp32_gather_median_ms": measure(lambda: conditional_memory_gather(*tables, *ids)),
        "int8_rowwise_gather_median_ms": measure(lambda: packed.gather(*ids)),
        "fp32_projection_median_ms": measure(lambda: projection(fp32)),
        "int8_dynamic_projection_median_ms": measure(lambda: dynamic(int8)),
        "gather_max_abs_error": float((fp32 - int8).abs().max().detach()),
        "gather_mean_abs_error": float((fp32 - int8).abs().mean().detach()),
        "projected_max_abs_error": float((fp_projected - int8_projected).abs().max().detach()),
        "projected_mean_abs_error": float((fp_projected - int8_projected).abs().mean().detach()),
        "quantized_engine": torch.backends.quantized.engine,
        "machine_code_vnni_verified": False,
        "vnni_note": "Attempted PyTorch x86 dynamic-INT8 projection; no compiler/disassembler is installed, so emitted VNNI instructions cannot be verified.",
    }
    target = ROOT / "artifacts" / "int8_benchmark.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/lattice_64k_128_vectorized_12_step12.pt")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.threads), indent=2))


if __name__ == "__main__": main()
