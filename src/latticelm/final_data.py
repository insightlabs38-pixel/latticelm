"""Deterministic resumable block streams for the frozen final data regime."""
from __future__ import annotations

import math
import random

import torch


class PermutedBlocks:
    """Without-replacement affine block permutation serialized by draw count."""

    def __init__(self, tokens: torch.Tensor, context: int, seed: int):
        self.tokens, self.context, self.seed, self.draws = tokens, context, seed, 0
        self.blocks = (len(tokens) - 1) // context

    def one(self) -> tuple[torch.Tensor, torch.Tensor]:
        epoch, position = divmod(self.draws, self.blocks)
        rng = random.Random(self.seed + epoch)
        multiplier = rng.randrange(1, self.blocks)
        while math.gcd(multiplier, self.blocks) != 1:
            multiplier = (multiplier + 1) % self.blocks or 1
        offset = rng.randrange(self.blocks)
        block = (multiplier * position + offset) % self.blocks
        self.draws += 1
        start = block * self.context
        return (self.tokens[start:start + self.context],
                self.tokens[start + 1:start + self.context + 1])
