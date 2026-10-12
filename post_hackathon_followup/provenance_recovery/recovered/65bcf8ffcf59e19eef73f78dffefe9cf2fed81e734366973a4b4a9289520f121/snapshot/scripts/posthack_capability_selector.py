"""Train a disjoint, inexpensive capability selector and score Research-D.

Only TRAIN splits of three pinned non-evaluation datasets are features. The
result is a within-source rank, not a calibrated probability of correctness.
No benchmark prompt or answer from the held-out suite is used for fitting.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from tokenizers import Tokenizer

from posthack_campaign import RESEARCH, ROOT, atomic_json, sha256, utc

SOURCES = {
    "allenai/sciq": "2c94ad3e1aafab77146f384e23536f97a4849815",
    "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "allenai/qasc": "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070",
}
BASE = RESEARCH / "research_d/documents.jsonl"
OUTPUT = RESEARCH / "research_d/capability_documents.jsonl"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
SEED = 738291


def positive_texts() -> tuple[list[str], dict]:
    texts = []
    provenance = {}
    for repo, revision in SOURCES.items():
        path = Path(hf_hub_download(repo, "data/train-00000-of-00001.parquet",
                                    repo_type="dataset", revision=revision))
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            if repo.endswith("/sciq"):
                content = f"{row['support']}\n{row['question']} {row['correct_answer']}"
            elif repo.endswith("/commonsense_qa"):
                choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
                content = f"{row['question']} {choices[row['answerKey']]}"
            else:
                choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
                content = f"{row['fact1']} {row['fact2']} {row['question']} {choices[row['answerKey']]}"
            texts.append(content)
        provenance[repo] = {"revision": revision, "train_rows": len(rows),
                            "train_parquet_sha256": sha256(path)}
    return texts, provenance


def sparse_rows(sequences: list[list[int]], vocab_size: int) -> csr_matrix:
    data, indices, indptr = [], [], [0]
    for ids in sequences:
        counts = Counter(int(token) for token in ids[:256] if 4 <= int(token) < vocab_size)
        norm = max(1, sum(counts.values()))
        for token, count in sorted(counts.items()):
            indices.append(token)
            data.append(count / norm)
        indptr.append(len(indices))
    return csr_matrix((np.asarray(data, dtype=np.float32),
                       np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int64)),
                      shape=(len(sequences), vocab_size), dtype=np.float32)


def shard_array(row: dict, cache: dict[str, np.memmap]) -> np.memmap:
    path = row["shard_path"]
    if path not in cache:
        cache[path] = np.memmap(ROOT / "artifacts/data/data_d_v4/canonical-2250m" / path,
                                mode="r", dtype="<i4")
    return cache[path]


def ids_for(row: dict, cache: dict[str, np.memmap]) -> list[int]:
    array = shard_array(row, cache)
    return array[int(row["token_start"]):min(int(row["token_end"]),
                                               int(row["token_start"]) + 256)].tolist()


def negative_examples(limit_per_source: int = 5000) -> tuple[list[list[int]], dict]:
    rng = random.Random(SEED)
    reservoirs: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    with BASE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "train" or row["token_count"] < 32:
                continue
            source = row["source"]
            seen[source] += 1
            reservoir = reservoirs[source]
            if len(reservoir) < limit_per_source:
                reservoir.append(row)
            else:
                position = rng.randrange(seen[source])
                if position < limit_per_source:
                    reservoir[position] = row
    cache: dict[str, np.memmap] = {}
    samples = [ids_for(row, cache) for source in sorted(reservoirs)
               for row in reservoirs[source]]
    return samples, {source: {"eligible_documents": seen[source],
                              "negative_samples": len(rows)}
                     for source, rows in sorted(reservoirs.items())}


def fit_selector() -> tuple[np.ndarray, float, dict]:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    positives, provenance = positive_texts()
    positive_ids = [tokenizer.encode(text).ids for text in positives]
    negative_ids, negative_counts = negative_examples()
    matrix = sparse_rows(positive_ids + negative_ids, tokenizer.get_vocab_size())
    labels = np.r_[np.ones(len(positive_ids), dtype=np.int8),
                   np.zeros(len(negative_ids), dtype=np.int8)]
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(labels))
    split = int(.9 * len(labels))
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=150,
                        tol=1e-4, class_weight="balanced", random_state=SEED)
    clf.fit(matrix[indices[:split]], labels[indices[:split]])
    heldout = clf.score(matrix[indices[split:]], labels[indices[split:]])
    return clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), {
        "positive_resources": provenance, "positive_examples": len(positive_ids),
        "negative_corpus_samples": negative_counts,
        "heldout_example_accuracy": float(heldout),
        "vocabulary_size": tokenizer.get_vocab_size(),
        "feature_policy": "4K token-count frequency over first 256 tokens; no evaluation examples",
        "classifier": "scikit-learn SGD logistic regression, class balanced",
        "seed": SEED,
    }


def raw_score(ids: list[int], weights: np.ndarray, intercept: float) -> float:
    counts = Counter(token for token in ids if 4 <= token < len(weights))
    total = max(1, sum(counts.values()))
    return intercept + sum(float(weights[token]) * count for token, count in counts.items()) / total


def score_index(weights: np.ndarray, intercept: float, tokenizer: Tokenizer) -> dict:
    if OUTPUT.exists():
        raise RuntimeError("refusing to overwrite existing capability-scored index")
    start = time.perf_counter()
    cache: dict[str, np.memmap] = {}
    scores_by_source: dict[str, list[float]] = defaultdict(list)
    ordered_scores = RESEARCH / "research_d/.capability_raw_scores.f32"
    count = 0
    with BASE.open(encoding="utf-8") as source, ordered_scores.open("wb") as raw:
        for line in source:
            row = json.loads(line)
            value = raw_score(ids_for(row, cache), weights, intercept)
            raw.write(np.float32(value).tobytes())
            if row["split"] == "train":
                scores_by_source[row["source"]].append(value)
            count += 1
    for values in scores_by_source.values():
        values.sort()
    samples: dict[str, list[dict]] = defaultdict(list)
    selected_seen: dict[str, int] = defaultdict(int)
    rng = random.Random(SEED + 1)
    raw_array = np.memmap(ordered_scores, mode="r", dtype="<f4", shape=(count,))
    temp = OUTPUT.with_suffix(".jsonl.tmp")
    with BASE.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as out:
        for index, line in enumerate(source):
            row = json.loads(line)
            if row["split"] == "train":
                distribution = scores_by_source[row["source"]]
                percentile = bisect.bisect_right(distribution, float(raw_array[index])) / len(distribution)
                row["capability_selection_score"] = percentile
                if percentile >= .90:
                    name = row["source"]
                    selected_seen[name] += 1
                    reservoir = samples[name]
                    if len(reservoir) < 10:
                        reservoir.append(row)
                    else:
                        place = rng.randrange(selected_seen[name])
                        if place < 10:
                            reservoir[place] = row
            out.write(json.dumps(row, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, OUTPUT)
    del raw_array
    ordered_scores.unlink()
    inspection = {}
    for source, rows in samples.items():
        inspection[source] = [{"document_id": row["document_id"],
                               "rank": row["capability_selection_score"],
                               "text_prefix": tokenizer.decode(ids_for(row, cache)[:128])[:650]}
                              for row in rows]
    return {"index_sha256": sha256(OUTPUT), "documents_scored": count,
            "selected_at_90th_percentile": dict(selected_seen),
            "raw_score_quantiles_by_source": {
                name: {str(q): values[min(len(values)-1, int(q * (len(values)-1)))]
                       for q in (0, .1, .5, .9, .99, 1)}
                for name, values in scores_by_source.items()},
            "random_selected_document_inspection": inspection,
            "score_seconds": time.perf_counter() - start}


def main() -> None:
    if not BASE.exists():
        raise RuntimeError("Research-D metadata index must be built first")
    started = time.perf_counter()
    weights, intercept, fit = fit_selector()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    scoring = score_index(weights, intercept, tokenizer)
    corpus = json.loads((ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json").read_text())
    report = {"schema": "research-d-disjoint-capability-selector-v1",
              "created_utc": utc(), "base_index_sha256": sha256(BASE),
              "tokenizer_sha256": sha256(TOKENIZER), "fit": fit, "scoring": scoring,
              "selection_rule": "Within-source top decile, sampled as 12.5% of each source stream",
              "evaluation_exclusion": "SciQ, CommonsenseQA and QASC TRAIN only; none are in the frozen competition or transfer evaluation suite",
              "contamination_audit": {"certified_source_corpus": corpus.get("certified"),
                                      "decontamination_version": corpus.get("decontamination_version"),
                                      "benchmark_registry_sizes": corpus.get("benchmark_registry_sizes"),
                                      "overlap_registry": corpus.get("overlap_registry"),
                                      "positive_training_resources_disjoint_from_evaluation_tasks": True,
                                      "selected_document_hashes_preserved": True},
              "contamination_caveat": "Inherited DATA-D-v4 exact/13-gram/SimHash exclusion covers its registered benchmarks; semantic overlap with transfer tasks remains an audit limit. Random selected snippets are preserved for manual inspection.",
              "total_seconds": time.perf_counter() - started}
    atomic_json(RESEARCH / "research_d/capability_selector_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "scoring"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
