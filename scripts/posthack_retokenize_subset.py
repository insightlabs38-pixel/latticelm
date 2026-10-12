"""Retokenize one frozen raw-document subset with 4K, 8K, and 16K tokenizers.

Document identity, source, ordering and train/validation assignment are shared
between tokenizers. This is an ablation subset, not a DATA-D-v4 rebuild.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from final_recipe_prepare import CACHE, MATH_SAMPLE
from latticelm.data_d import SCHEMA_VERSION, validation_assignment
from posthack_campaign import RESEARCH, ROOT, atomic_json, sha256, utc

OUT = RESEARCH / "retokenized_subset"
TOKENIZERS = {4096: ROOT / "artifacts/tokenizers/final_corpus_4k.json",
              8192: ROOT / "artifacts/tokenizers/final_corpus_8k.json",
              16384: RESEARCH / "tokenizers/final_corpus_16k.json"}
SOURCES = ("fineweb_edu", "wikipedia", "fineweb", "finemath")
MIXTURE = {"fineweb_edu": .4483, "wikipedia": .25, "fineweb": .2011, "finemath": .1006}


def source_rows():
    for path in (CACHE, MATH_SAMPLE):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if path == MATH_SAMPLE:
                    # The pinned math sample lacks upstream IDs; line hashes
                    # identify exact normalized bytes in both tokenizations.
                    row["document_id"] = "finemath:sample:" + sha256_text(row["normalized_text"])
                yield row


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build() -> dict:
    if (OUT / "report.json").exists():
        raise RuntimeError("completed retokenized subset exists; refusing overwrite")
    # A run cannot be registered until the common-subset report exists. Clear
    # unfinished derived shards before replaying the pinned source sample.
    for size in TOKENIZERS:
        folder = OUT / f"{size//1024}k"
        if folder.exists():
            shutil.rmtree(folder)
    if any(not path.exists() for path in (*TOKENIZERS.values(), CACHE, MATH_SAMPLE)):
        raise RuntimeError("frozen raw sample and 4K/8K/16K tokenizers are required")
    start = time.perf_counter()
    tokenizers = {size: Tokenizer.from_file(str(path)) for size, path in TOKENIZERS.items()}
    state = {}
    for size in TOKENIZERS:
        base = OUT / f"{size//1024}k"
        base.mkdir(parents=True, exist_ok=True)
        for source in SOURCES:
            for split in ("train", "validation"):
                key = (size, source, split)
                path = base / f"{source}-{split}.int32"
                state[key] = {"path": path, "handle": path.open("wb"),
                              "ids": [], "offsets": [0], "bytes": 0,
                              "characters": 0, "words": 0,
                              "upstream": []}
    source_counts = defaultdict(int)
    try:
        for row in source_rows():
            source = row["source"]
            if source not in SOURCES:
                continue
            identifier = row["document_id"]
            text = row["normalized_text"]
            if not text.strip():
                continue
            split = validation_assignment(identifier)
            source_counts[(source, split)] += 1
            raw_bytes = len(text.encode("utf-8"))
            for size, tokenizer in tokenizers.items():
                ids = tokenizer.encode(text).ids + [2]
                if not ids:
                    continue
                item = state[(size, source, split)]
                item["handle"].write(np.asarray(ids, dtype="<i4").tobytes())
                item["ids"].append(identifier)
                item["offsets"].append(item["offsets"][-1] + len(ids))
                item["bytes"] += raw_bytes
                item["characters"] += len(text)
                item["words"] += len(text.split())
                item["upstream"].append(row.get("upstream", {}))
    finally:
        for item in state.values():
            item["handle"].flush()
            os.fsync(item["handle"].fileno())
            item["handle"].close()
    reports = {}
    for size, tokenizer_path in TOKENIZERS.items():
        base = OUT / f"{size//1024}k"
        manifest = {"schema_version": "posthack-retokenized-subset-v1",
                    "created_utc": utc(), "source_sample_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                                                   for path in (CACHE, MATH_SAMPLE)},
                    "tokenizer_sha256": sha256(tokenizer_path),
                    "mixture_definition": MIXTURE, "shards": [], "validation_shards": []}
        source_metrics = {}
        for source in SOURCES:
            source_metrics[source] = {}
            for split in ("train", "validation"):
                item = state[(size, source, split)]
                token_count = item["offsets"][-1]
                metadata = {"schema_version": SCHEMA_VERSION,
                            "path": item["path"].name, "source": source,
                            "split": split, "token_count": token_count,
                            "sha256": sha256(item["path"]),
                            "tokenizer_sha256": sha256(tokenizer_path),
                            "document_count": len(item["ids"]),
                            "stable_document_ids": item["ids"],
                            "document_boundary_offsets": item["offsets"],
                            "document_upstream_metadata": item["upstream"],
                            "raw_byte_count": item["bytes"],
                            "validation_assignment_method": "sha256-document-id-mod50-v1"}
                metadata_path = base / f"{source}-{split}.manifest.json"
                atomic_json(metadata_path, metadata)
                reference = {"manifest_path": metadata_path.name,
                             "manifest_sha256": sha256(metadata_path)}
                manifest["shards" if split == "train" else "validation_shards"].append(reference)
                source_metrics[source][split] = {"documents": len(item["ids"]),
                                                 "raw_bytes": item["bytes"],
                                                 "characters": item["characters"],
                                                 "words": item["words"],
                                                 "tokens": token_count,
                                                 "bytes_per_token": item["bytes"] / max(1, token_count),
                                                 "characters_per_token": item["characters"] / max(1, token_count),
                                                 "words_per_token": item["words"] / max(1, token_count)}
        atomic_json(base / "manifest.json", manifest)
        reports[str(size)] = {"manifest_path": str((base / "manifest.json").relative_to(ROOT)),
                              "manifest_sha256": sha256(base / "manifest.json"),
                              "source_metrics": source_metrics,
                              "train_tokens": sum(source_metrics[s]["train"]["tokens"] for s in SOURCES)}
    report = {"schema": "retokenized-common-subset-v1", "created_utc": utc(),
              "source_count_by_split": {f"{s}/{split}": count
                                        for (s, split), count in source_counts.items()},
              "tokenizations": reports,
              "training_bytes_policy": "same document identities and order within each source; source mixture by token count; source byte exposure reported separately",
              "validation_policy": "frozen hash-of-document-ID assignment; validation documents excluded from training",
              "wall_seconds": time.perf_counter() - start}
    atomic_json(OUT / "report.json", report)
    return report


if __name__ == "__main__":
    result = build()
    print(json.dumps({"wall_seconds": result["wall_seconds"],
                      "train_tokens": {k: v["train_tokens"] for k, v in result["tokenizations"].items()}}), flush=True)
