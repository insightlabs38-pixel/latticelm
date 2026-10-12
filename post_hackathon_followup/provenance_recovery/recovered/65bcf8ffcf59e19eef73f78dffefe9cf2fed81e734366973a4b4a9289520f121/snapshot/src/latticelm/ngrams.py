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
    # Vectorized polynomial hashes avoid a Python/CPU round trip on every
    # training batch. Each position uses only x[t], x[t-1], ... and explicit
    # zeroes for missing prefix items, so addressing remains causal.
    values = tokens.to(torch.long)
    length = values.size(1)

    def suffix(order: int, seed: int) -> torch.Tensor:
        h = torch.full_like(values, seed)
        for offset in range(order - 1, -1, -1):
            shifted = torch.zeros_like(values)
            if offset == 0:
                shifted = values
            elif offset < length:
                shifted[:, offset:] = values[:, : length - offset]
            h = torch.remainder(h * 1_000_003 + shifted + 0x9E37, 2_147_483_647)
        return torch.remainder(h, slots)

    return suffix(2, 0xB1), suffix(3, 0xC3), suffix(4, 0xD4)
