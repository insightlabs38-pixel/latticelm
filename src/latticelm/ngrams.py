from __future__ import annotations

import torch

MASK64 = (1 << 64) - 1


def stable_hash_ngram(values: list[int], seed: int) -> int:
    """Deterministic 64-bit FNV-1a + avalanche hash; independent of hash seed."""
    h = (1469598103934665603 ^ seed) & MASK64
    for value in values:
        h ^= int(value) & MASK64
        h = (h * 1099511628211) & MASK64
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & MASK64
    h ^= h >> 33
    return h & MASK64


def causal_ngram_ids(tokens: torch.Tensor, slots: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return suffix n-gram IDs for [B,T] inputs, never reading t+1 or later."""
    if tokens.ndim != 2:
        raise ValueError("tokens must be [batch, sequence]")
    rows = tokens.detach().to("cpu", torch.int64).tolist()
    result = [[[0] * len(row) for row in rows] for _ in range(3)]
    seeds = (0xB1, 0xC3, 0xD4)
    for b, row in enumerate(rows):
        for t in range(len(row)):
            for idx, n in enumerate((2, 3, 4)):
                start = max(0, t - n + 1)
                # Include explicit leading zeroes so short prefixes have stable meaning.
                suffix = [0] * (n - (t - start + 1)) + row[start : t + 1]
                result[idx][b][t] = stable_hash_ngram(suffix, seeds[idx]) % slots
    return tuple(torch.tensor(x, device=tokens.device, dtype=torch.long) for x in result)
