"""Minimal bounded correctness smoke test for the experimental CPU backend."""
from __future__ import annotations

import json

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, output, count: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < count
    tl.store(output + offsets, tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask), mask=mask)


def main() -> None:
    triton.runtime.driver.set_active_to_cpu()
    x = torch.arange(257, dtype=torch.float32)
    y = torch.arange(257, dtype=torch.float32).flip(0)
    output = torch.empty_like(x)
    add_kernel[(1,)](x, y, output, count=x.numel(), block=512)
    print(json.dumps({"triton_version": triton.__version__, "correct": torch.equal(output, x + y),
                      "elements": x.numel(), "sum": float(output.sum())}))


if __name__ == "__main__":
    main()
