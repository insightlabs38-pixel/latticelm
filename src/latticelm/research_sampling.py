"""Deterministic Research-D sampling overlays for certified token shards."""
from __future__ import annotations

import bisect
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data_d import SourceStream


def unit_hash(seed: int, count: int, salt: int = 0) -> float:
    """Stable [0,1) pseudorandom value independent of Python hash seeding."""
    value = (seed ^ (count * 0x9E3779B97F4A7C15) ^ salt) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return ((value ^ (value >> 31)) & ((1 << 53) - 1)) / float(1 << 53)


def load_document_maps(index: str | Path) -> dict[str, dict]:
    maps: dict[str, dict[str, list]] = defaultdict(lambda: {
        "starts": [], "ends": [], "difficulty": [], "quality": [], "capability": []})
    with Path(index).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "train":
                continue
            bucket = maps[row["shard_path"]]
            bucket["starts"].append(int(row["token_start"]))
            bucket["ends"].append(int(row["token_end"]))
            bucket["difficulty"].append(float(row["difficulty_score"]))
            bucket["quality"].append(float(row["quality_percentile"] or 0.0))
            bucket["capability"].append(float(row["capability_selection_score"] or 0.0))
    for path, bucket in maps.items():
        if bucket["starts"] != sorted(bucket["starts"]):
            raise RuntimeError(f"document starts are unsorted in {path}")
        if len(bucket["starts"]) != len(bucket["ends"]):
            raise RuntimeError(f"document metadata is inconsistent in {path}")
    return dict(maps)


class ResearchSourceStream(SourceStream):
    """SourceStream-compatible affine stream with registered rejection rules."""

    def __init__(self, arrays: Sequence[np.ndarray], context: int, seed: int,
                 shard_paths: Sequence[str], document_maps: Mapping[str, dict],
                 source: str, mode: str, *, curriculum_fraction: float = .25,
                 curriculum_strength: float = .8, quality_floor: float = .70,
                 capability_floor: float = .90, capability_fraction: float = .125) -> None:
        super().__init__(arrays, context, seed)
        if len(shard_paths) != len(arrays):
            raise ValueError("shard path and array order must match")
        if mode not in {"curriculum", "quality_anneal", "curriculum_anneal", "capability"}:
            raise ValueError("unsupported Research-D sampling mode")
        if not 0 < curriculum_fraction <= 1 or not 0 <= curriculum_strength < 1:
            raise ValueError("invalid curriculum rule")
        if not 0 <= quality_floor <= 1 or not 0 <= capability_floor <= 1 or not 0 <= capability_fraction <= 1:
            raise ValueError("invalid quality or capability rule")
        self.shard_paths = tuple(shard_paths)
        self.maps = [document_maps[path] for path in shard_paths]
        self.source = source
        self.mode = mode
        self.curriculum_fraction = curriculum_fraction
        self.curriculum_strength = curriculum_strength
        self.quality_floor = quality_floor
        self.capability_floor = capability_floor
        self.capability_fraction = capability_fraction
        self.progress = 0.0
        self.accepted = 0
        self.rejected = 0
        self.accepted_difficulty_sum = 0.0
        self.selected_component_count = 0

    def set_progress(self, progress: float) -> None:
        if not 0 <= progress <= 1:
            raise ValueError("training progress must be in [0,1]")
        self.progress = progress

    def _candidate(self) -> tuple[np.ndarray, np.ndarray, dict, int]:
        epoch, position = divmod(self.draws, self.total_blocks)
        rng = random.Random(self.seed + epoch)
        multiplier = rng.randrange(1, self.total_blocks)
        while math.gcd(multiplier, self.total_blocks) != 1:
            multiplier = (multiplier + 1) % self.total_blocks or 1
        block = (multiplier * position + rng.randrange(self.total_blocks)) % self.total_blocks
        candidate_number = self.draws
        self.draws += 1
        shard_index = 0
        while block >= self.blocks[shard_index]:
            block -= self.blocks[shard_index]
            shard_index += 1
        start = block * self.context
        array = self.arrays[shard_index]
        metadata = self.maps[shard_index]
        document_index = max(0, bisect.bisect_right(metadata["starts"], start) - 1)
        return (np.asarray(array[start:start+self.context]),
                np.asarray(array[start+1:start+self.context+1]),
                {key: metadata[key][document_index] for key in ("difficulty", "quality", "capability")},
                candidate_number)

    def one(self) -> tuple[np.ndarray, np.ndarray]:
        select_capability = (self.mode == "capability" and
                             unit_hash(self.seed, self.draws, 0xA15) < self.capability_fraction)
        for _ in range(1000):
            x, y, metadata, candidate = self._candidate()
            accept = True
            if self.mode in {"curriculum", "curriculum_anneal"} and self.progress < self.curriculum_fraction:
                strength = self.curriculum_strength * max(0.0, 1 - self.progress / self.curriculum_fraction)
                accept = unit_hash(self.seed, candidate, 0xC33) < 1 - strength * metadata["difficulty"]
            elif self.mode in {"quality_anneal", "curriculum_anneal"} and self.progress >= .85 and self.source == "fineweb_edu":
                accept = metadata["quality"] >= self.quality_floor
            elif self.mode == "capability" and select_capability:
                accept = metadata["capability"] >= self.capability_floor
            if accept:
                self.accepted += 1
                self.accepted_difficulty_sum += metadata["difficulty"]
                self.selected_component_count += int(select_capability)
                return x, y
            self.rejected += 1
        raise RuntimeError(f"Research-D {self.mode} sampler found no eligible block")
