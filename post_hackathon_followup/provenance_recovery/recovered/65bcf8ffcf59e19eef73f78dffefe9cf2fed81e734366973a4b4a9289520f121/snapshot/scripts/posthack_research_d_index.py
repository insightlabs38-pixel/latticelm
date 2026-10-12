"""Derive a document control plane from certified DATA-D-v4 shards.

The corpus is never rebuilt or retokenized here. Exact raw bytes are available
per shard, not per document, so document bytes are explicitly labeled estimates.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from posthack_campaign import RESEARCH, ROOT, atomic_json, sha256, utc

CORPUS = ROOT / "artifacts/data/data_d_v4/canonical-2250m"
MANIFEST = CORPUS / "manifest.json"


def exact_hash_lookup() -> dict[str, str]:
    connection = sqlite3.connect(f"file:{CORPUS / 'dedup.sqlite'}?mode=ro", uri=True)
    try:
        return {identifier: digest for digest, identifier in connection.execute("SELECT h,id FROM exact")}
    finally:
        connection.close()


def score(metadata: dict) -> float | None:
    for name in ("score", "int_score", "language_score", "quality_score"):
        value = metadata.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def documents(top: dict):
    for collection in ("shards", "validation_shards"):
        for reference in top[collection]:
            path = CORPUS / reference["manifest_path"]
            meta = json.loads(path.read_text())
            token_path = CORPUS / meta["path"]
            values = np.memmap(token_path, mode="r", dtype="<i4")
            offsets = meta["document_boundary_offsets"]
            ids = meta["stable_document_ids"]
            upstream = meta["document_upstream_metadata"]
            if len(offsets) != len(ids) + 1 or len(upstream) != len(ids):
                raise RuntimeError(f"document boundary or upstream metadata mismatch: {path}")
            for index, identifier in enumerate(ids):
                start, end = int(offsets[index]), int(offsets[index + 1])
                if not 0 <= start < end <= len(values):
                    raise RuntimeError(f"invalid document offsets: {identifier}")
                sampled = values[start:min(end, start + 256)]
                lexical_diversity = len(np.unique(sampled)) / len(sampled)
                yield {"document_id": identifier, "source": meta["source"],
                       "split": meta["split"], "shard_path": meta["path"],
                       "shard_manifest_path": str(path.relative_to(CORPUS)),
                       "token_start": start, "token_end": end, "token_count": end - start,
                       "shard_raw_byte_count": int(meta["raw_byte_count"]),
                       "raw_bytes_estimate": (end - start) * int(meta["raw_byte_count"]) / int(meta["token_count"]),
                       "raw_bytes_exact": False,
                       "dedup_status": "retained", "upstream_quality_score": score(upstream[index]),
                       "lexical_diversity_first256": lexical_diversity}


def percentile(sorted_values: list[float], value: float) -> float:
    return bisect.bisect_right(sorted_values, value) / max(1, len(sorted_values))


def build() -> dict:
    start = time.perf_counter()
    top = json.loads(MANIFEST.read_text())
    out = RESEARCH / "research_d"
    out.mkdir(parents=True, exist_ok=True)
    intermediate = out / ".metadata_raw.jsonl.tmp"
    if (out / "documents.jsonl").exists():
        raise RuntimeError("Research-D index already exists; refusing overwrite")
    lengths: dict[str, list[float]] = defaultdict(list)
    diversity: dict[str, list[float]] = defaultdict(list)
    quality: dict[str, list[float]] = defaultdict(list)
    count = 0
    with intermediate.open("w", encoding="utf-8") as handle:
        for row in documents(top):
            if row["split"] == "train":
                source = row["source"]
                lengths[source].append(math.log1p(row["token_count"]))
                diversity[source].append(row["lexical_diversity_first256"])
                if row["upstream_quality_score"] is not None:
                    quality[source].append(row["upstream_quality_score"])
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    for collection in (lengths, diversity, quality):
        for values in collection.values():
            values.sort()
    hashes = exact_hash_lookup()
    destination = out / "documents.jsonl"
    missing_hashes = 0
    train_count = 0
    with intermediate.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8") as handle:
        for line in source:
            row = json.loads(line)
            name = row["source"]
            row["normalized_text_sha256"] = hashes.get(row["document_id"])
            if row["normalized_text_sha256"] is None:
                missing_hashes += 1
            row["difficulty_length_percentile"] = percentile(lengths[name], math.log1p(row["token_count"]))
            row["difficulty_lexical_percentile"] = percentile(diversity[name], row["lexical_diversity_first256"])
            row["difficulty_score"] = (row["difficulty_length_percentile"] + row["difficulty_lexical_percentile"]) / 2
            if row["upstream_quality_score"] is not None and quality[name]:
                row["quality_percentile"] = percentile(quality[name], row["upstream_quality_score"])
            else:
                row["quality_percentile"] = None
            row["capability_selection_score"] = None
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            train_count += row["split"] == "train"
        handle.flush()
        os.fsync(handle.fileno())
    intermediate.unlink()
    report = {"schema": "research-d-index-v1", "created_utc": utc(),
              "data_manifest_sha256": sha256(MANIFEST),
              "index_sha256": sha256(destination), "documents": count,
              "train_documents": train_count, "missing_normalized_hashes": missing_hashes,
              "raw_bytes_policy": "exact per shard; document bytes estimated by shard bytes/token ratio",
              "difficulty_policy": "equal mean of within-source training percentiles of log document token length and first-256-token lexical diversity",
              "quality_policy": "within-source percentile of existing upstream score where available",
              "capability_policy": "unscored until the disjoint selector is frozen",
              "build_seconds": time.perf_counter() - start}
    atomic_json(out / "index_report.json", report)
    return report


def main() -> None:
    print(json.dumps(build(), indent=2), flush=True)


if __name__ == "__main__":
    main()
