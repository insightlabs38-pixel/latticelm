"""Deterministic offline data primitives for DATA-D-BROAD-v1.

The module intentionally contains no network acquisition.  Builders may fetch
upstream rows, but the training path consumes only verified little-endian int32
shards and immutable JSON manifests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "data-d-shard-v1"
CORPUS_ID = "DATA-D-BROAD-v1"
SOURCES = ("fineweb_edu", "wikipedia", "fineweb", "babylm")
MIXTURE_COUNTS = {"fineweb_edu": 20, "wikipedia": 8, "fineweb": 6, "babylm": 6}
DECONTAMINATION_VERSION = "normalized-exact+13gram+simhash-v1"
DEDUP_VERSION = "sha256-paragraph+simhash64-v1"


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text).lower())


def document_hash(text: str) -> str:
    return sha256_bytes(normalize_text(text).encode())


def paragraph_hashes(text: str) -> tuple[str, ...]:
    pieces = re.split(r"\n\s*\n|(?<=[.!?])\s{2,}", normalize_text(text))
    return tuple(document_hash(piece) for piece in pieces if len(normalized_words(piece)) >= 8)


def simhash64(text: str, n: int = 5) -> int:
    words = normalized_words(text)
    grams = (" ".join(words[i:i+n]) for i in range(max(1, len(words)-n+1)))
    weights = [0] * 64
    for gram in grams:
        value = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(frozen=True)
class Document:
    source: str
    document_id: str
    text: str
    upstream: Mapping[str, object]
    license_label: str

    @property
    def exact_hash(self) -> str:
        return document_hash(self.text)


class GlobalDeduplicator:
    """Order-independent selection using a frozen priority then stable ID.

    ``select`` accepts the bounded candidate collection for a materialization
    window, sorts it canonically, and therefore cannot change winners merely
    because workers finish in a different order.
    """

    PRIORITY = {"data_c": 0, "fineweb_edu": 1, "wikipedia": 2, "fineweb": 3, "babylm": 4}

    def select(self, documents: Iterable[Document]) -> tuple[list[Document], list[dict[str, object]]]:
        ordered = sorted(documents, key=lambda d: (self.PRIORITY[d.source], d.document_id, d.exact_hash))
        kept: list[Document] = []
        decisions: list[dict[str, object]] = []
        exact: dict[str, str] = {}
        paragraphs: dict[str, str] = {}
        # Four 16-bit bands give a practical deterministic candidate index.
        bands: list[dict[int, list[tuple[int, str]]]] = [dict() for _ in range(4)]
        for doc in ordered:
            reason = winner = None
            if doc.exact_hash in exact:
                reason, winner = "exact_document", exact[doc.exact_hash]
            ph = paragraph_hashes(doc.text)
            if reason is None:
                matches = sorted({paragraphs[x] for x in ph if x in paragraphs})
                if matches:
                    reason, winner = "exact_paragraph", matches[0]
            fingerprint = simhash64(doc.text)
            if reason is None:
                candidates: dict[str, int] = {}
                for index in range(4):
                    key = (fingerprint >> (index * 16)) & 0xFFFF
                    for prior, prior_id in bands[index].get(key, []): candidates[prior_id] = prior
                close = sorted((hamming(fingerprint, prior), prior_id) for prior_id, prior in candidates.items())
                if close and close[0][0] <= 3:
                    reason, winner = "near_duplicate", close[0][1]
            if reason is not None:
                decisions.append({"document_id": doc.document_id, "source": doc.source,
                                  "decision": "removed", "rule": reason, "retained_document_id": winner})
                continue
            kept.append(doc); exact[doc.exact_hash] = doc.document_id
            for value in ph: paragraphs[value] = doc.document_id
            for index in range(4):
                key = (fingerprint >> (index * 16)) & 0xFFFF
                bands[index].setdefault(key, []).append((fingerprint, doc.document_id))
            decisions.append({"document_id": doc.document_id, "source": doc.source, "decision": "retained"})
        return kept, decisions


class ContaminationRegistry:
    def __init__(self, examples: Mapping[str, Sequence[str]], ngram: int = 13):
        self.ngram = ngram; self.exact: dict[str, set[str]] = {}; self.grams: dict[str, set[int]] = {}
        self.simhashes: dict[str, list[int]] = {}
        for benchmark, rows in examples.items():
            self.exact[benchmark] = {" ".join(normalized_words(row)) for row in rows if len(normalized_words(row)) >= 8}
            self.grams[benchmark] = set()
            self.simhashes[benchmark] = []
            for row in rows:
                words = normalized_words(row)
                self.grams[benchmark].update(self._grams(words))
                if len(words) >= 20: self.simhashes[benchmark].append(simhash64(row))

    def _grams(self, words: Sequence[str]) -> set[int]:
        return {int.from_bytes(hashlib.blake2b(" ".join(words[i:i+self.ngram]).encode(), digest_size=8).digest(), "big")
                for i in range(max(0, len(words)-self.ngram+1))}

    def matches(self, document: Document) -> list[dict[str, object]]:
        words = normalized_words(document.text); phrase = " ".join(words); grams = self._grams(words)
        output = []
        for benchmark in sorted(self.exact):
            exact = next((item for item in self.exact[benchmark] if item and item in phrase), None)
            overlap = sorted(grams & self.grams[benchmark])
            near = min((hamming(simhash64(document.text), value) for value in self.simhashes[benchmark]), default=65)
            rule = "normalized_exact" if exact else ("long_ngram" if len(overlap) >= 2 else ("near_duplicate" if near <= 3 else None))
            if rule:
                fingerprint = sha256_bytes((exact or str(overlap[:2]) or str(near)).encode())
                output.append({"source": document.source, "document_id": document.document_id,
                               "benchmark": benchmark, "rule": rule, "match_fingerprint": fingerprint,
                               "decontamination_version": DECONTAMINATION_VERSION})
        return output


class Int32Shard:
    def __init__(self, path: str | Path, manifest: Mapping[str, object], verify: bool = True):
        self.path = Path(path); self.manifest = dict(manifest)
        if verify:
            if self.manifest.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported shard schema")
            if self.manifest.get("sha256") != sha256_file(self.path): raise ValueError("shard hash mismatch")
            if self.path.stat().st_size != int(self.manifest["token_count"]) * 4: raise ValueError("shard size mismatch")
        self.tokens = np.memmap(self.path, mode="r", dtype="<i4")


class SourceStream:
    """Affine without-replacement sequence stream over one or more mmaps."""
    def __init__(self, arrays: Sequence[np.ndarray], context: int, seed: int):
        self.arrays = arrays; self.context = context; self.seed = seed; self.draws = 0
        self.blocks = [max(0, (len(value)-1)//context) for value in arrays]
        self.total_blocks = sum(self.blocks)
        if not self.total_blocks: raise ValueError("source contains no complete sequence")

    def one(self) -> tuple[np.ndarray, np.ndarray]:
        epoch, position = divmod(self.draws, self.total_blocks)
        rng = random.Random(self.seed + epoch); multiplier = rng.randrange(1, self.total_blocks)
        while math.gcd(multiplier, self.total_blocks) != 1: multiplier = (multiplier + 1) % self.total_blocks or 1
        block = (multiplier * position + rng.randrange(self.total_blocks)) % self.total_blocks
        self.draws += 1
        shard_index = 0
        while block >= self.blocks[shard_index]: block -= self.blocks[shard_index]; shard_index += 1
        start = block * self.context; array = self.arrays[shard_index]
        return np.asarray(array[start:start+self.context]), np.asarray(array[start+1:start+self.context+1])


class ExactMixture:
    """Five-batch cycle of eight sequences with exact 50/20/15/15 ratio."""
    CYCLE = (
        ("fineweb_edu", "fineweb_edu", "fineweb_edu", "fineweb_edu", "wikipedia", "wikipedia", "fineweb", "babylm"),
        ("fineweb_edu", "fineweb_edu", "fineweb_edu", "fineweb_edu", "wikipedia", "wikipedia", "fineweb", "babylm"),
        ("fineweb_edu", "fineweb_edu", "fineweb_edu", "fineweb_edu", "wikipedia", "fineweb", "fineweb", "babylm"),
        ("fineweb_edu", "fineweb_edu", "fineweb_edu", "fineweb_edu", "wikipedia", "wikipedia", "fineweb", "babylm"),
        ("fineweb_edu", "fineweb_edu", "fineweb_edu", "fineweb_edu", "wikipedia", "fineweb", "babylm", "babylm"),
    )

    def __init__(self, streams: Mapping[str, SourceStream]):
        self.streams = dict(streams); self.counter = 0

    def batch(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        labels = self.CYCLE[self.counter % len(self.CYCLE)]; self.counter += 1
        pairs = [self.streams[source].one() for source in labels]
        return np.stack([x for x, _ in pairs]), np.stack([y for _, y in pairs]), labels

    def state_dict(self) -> dict[str, object]:
        return {"source_selector_counter": self.counter,
                "source_draw_counts": {key: stream.draws for key, stream in self.streams.items()}}

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        self.counter = int(value["source_selector_counter"])
        draws = value["source_draw_counts"]
        for key, stream in self.streams.items(): stream.draws = int(draws[key])


def verify_top_manifest(path: str | Path, tokenizer_path: str | Path) -> dict[str, object]:
    path = Path(path); payload = json.loads(path.read_text())
    if payload.get("schema_version") != "data-d-corpus-v1" or payload.get("corpus_identity") != CORPUS_ID:
        raise ValueError("invalid DATA-D top-level manifest")
    if payload.get("tokenizer_sha256") != sha256_file(tokenizer_path): raise ValueError("wrong tokenizer")
    for child in payload.get("shards", []):
        manifest_path = path.parent / child["manifest_path"]
        if sha256_file(manifest_path) != child["manifest_sha256"]: raise ValueError("child manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text())
        Int32Shard(path.parent / manifest["path"], manifest)
    return payload
