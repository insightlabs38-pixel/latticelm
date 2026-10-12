"""Acquire, screen, tokenize, and manifest the Phase 7A corpora.

Raw text and token arrays are deliberately written below ``artifacts/data``
(git-ignored).  The compact manifests and diagnostics are committed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata
import urllib.request
import zipfile

import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download

from latticelm.tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/data/phase7a"
ARTIFACTS = ROOT / "artifacts"
BABY_REPO = "BabyLM-community/BabyLM-2026-Strict"
BABY_REVISION = "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf"
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SEED = 1337


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[a-z0-9]+", text)


def reference_documents() -> dict[str, list[str]]:
    specs = {
        "hellaswag": ("Rowan/hellaswag", None, ["validation", "test"]),
        "arc_easy": ("allenai/ai2_arc", "ARC-Easy", ["validation", "test"]),
        "piqa": ("ybisk/piqa", None, ["validation", "test"]),
        "winogrande": ("allenai/winogrande", "winogrande_xl", ["validation", "test"]),
        "wikitext_103": ("Salesforce/wikitext", "wikitext-103-raw-v1", ["validation", "test"]),
    }
    output: dict[str, list[str]] = {}
    for task, (repo, config, splits) in specs.items():
        rows = []
        for split in splits:
            try:
                ds = load_dataset(repo, config, split=split)
            except RuntimeError as error:
                # PIQA still uses a retired dataset script.  Its authoritative
                # repository publishes the underlying JSONL directly.
                if repo != "ybisk/piqa" or "Dataset scripts" not in str(error):
                    raise
                cache = DATA / "piqa_reference"; cache.mkdir(parents=True, exist_ok=True)
                if split == "test":
                    path = cache / "tests.jsonl"
                    if not path.exists(): urllib.request.urlretrieve("https://yonatanbisk.com/piqa/data/tests.jsonl", path)
                else:
                    archive = cache / "physicaliqa-train-dev.zip"
                    if not archive.exists(): urllib.request.urlretrieve("https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip", archive)
                    with zipfile.ZipFile(archive) as z: z.extract("physicaliqa-train-dev/dev.jsonl", cache)
                    path = cache / "physicaliqa-train-dev/dev.jsonl"
                ds = [json.loads(line) for line in Path(path).read_text().splitlines()]
            for row in ds:
                parts = []
                for value in row.values():
                    if isinstance(value, str):
                        parts.append(value)
                    elif isinstance(value, list) and value and isinstance(value[0], str):
                        parts.extend(value)
                    elif isinstance(value, dict):
                        parts.extend(str(x) for x in value.values() if isinstance(x, (str, int, float)))
                if parts:
                    rows.append(" ".join(parts))
        output[task] = rows
    return output


TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande", "wikitext_103"]


def fingerprints(refs: dict[str, list[str]], n: int = 13) -> tuple[dict[int, int], set[str]]:
    grams: dict[int, int] = {}
    exact: set[str] = set()
    for task_index, (task, documents) in enumerate(refs.items()):
        for document in documents:
            words = normalized_words(document)
            phrase = " ".join(words)
            if len(phrase) >= 80:
                exact.add(phrase)
            for i in range(max(0, len(words) - n + 1)):
                key = int.from_bytes(hashlib.blake2b(" ".join(words[i:i+n]).encode(), digest_size=8).digest(), "big")
                grams[key] = grams.get(key, 0) | (1 << task_index)
    return grams, exact


def screen(text: str, grams: dict[int, int], exact: set[str], n: int = 13) -> tuple[bool, list[str], str]:
    words = normalized_words(text)
    phrase = " ".join(words)
    task_mask = 0
    hits = 0
    for i in range(max(0, len(words) - n + 1)):
        key = int.from_bytes(hashlib.blake2b(" ".join(words[i:i+n]).encode(), digest_size=8).digest(), "big")
        if key in grams:
            hits += 1
            task_mask |= grams[key]
    exact_hit = any(len(item) >= 80 and item in phrase for item in exact) if hits else False
    remove = exact_hit or hits >= 2
    tasks = [task for index, task in enumerate(TASKS) if task_mask & (1 << index)]
    return remove, tasks, "normalized_exact" if exact_hit else f"13gram_hits={hits}"


def save_tokens(path: Path, tokens: list[int]) -> dict:
    array = np.asarray(tokens, dtype=np.int32)
    array.tofile(path)
    return {"path": str(path.relative_to(ROOT)), "tokens": int(array.size), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def diagnostics(name: str, documents: list[str], tokens: list[int], tokenizer) -> dict:
    lengths = [len(normalized_words(x)) for x in documents]
    chars = sum(len(x) for x in documents)
    words = sum(lengths)
    counts = Counter(tokens)
    repeated = sum(v for v in counts.values() if v > 1)
    ascii_chars = sum(sum(ord(c) < 128 for c in x) for x in documents)
    exact_dupes = len(documents) - len({hashlib.sha256(x.strip().encode()).digest() for x in documents})
    return {
        "dataset": name, "documents": len(documents), "words": words, "characters": chars,
        "tokens": len(tokens), "chars_per_token": chars / len(tokens), "words_per_token": words / len(tokens),
        "tokens_per_word": len(tokens) / words, "unknown_tokens": tokens.count(3),
        "unknown_rate": tokens.count(3) / len(tokens), "token_length_p50_document": statistics.median(lengths),
        "token_length_p95_document": sorted(lengths)[min(len(lengths)-1, math.floor(.95*len(lengths)))],
        "extremely_short_document_rate": sum(x < 50 for x in lengths) / len(lengths),
        "non_ascii_character_rate": 1 - ascii_chars / chars, "exact_duplicate_rate": exact_dupes / len(documents),
        "token_vocabulary_observed": len(counts), "token_repetition_rate": repeated / len(tokens),
        "byte_compression_ratio": chars / (4 * len(tokens)), "fallback_behavior": "byte-level; no UNK expected"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-tokens", type=int, default=20_000_000)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    baby_dir = DATA / "babylm_strict_100m"
    snapshot_download(BABY_REPO, repo_type="dataset", revision=BABY_REVISION,
                      allow_patterns=["*.train.txt", "README.md"], local_dir=baby_dir)
    baby_files = sorted(baby_dir.glob("*.train.txt"))
    sources = []
    for path in baby_files:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        sources.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "lines": len(lines)})
        del lines

    refs = reference_documents()
    grams, exact = fingerprints(refs)
    tok_path = ARTIFACTS / "tokenizers/babylm_2026_4k.json"
    tokenizer = load_tokenizer(tok_path)

    stream = load_dataset(FINEWEB_REPO, FINEWEB_CONFIG, split="train", streaming=True,
                          revision=FINEWEB_REVISION).shuffle(seed=SEED, buffer_size=1_000)
    fw_train: list[str] = []
    fw_val: list[str] = []
    removals = []
    accepted_tokens = 0
    validation_tokens = 0
    scanned = 0
    for row in stream:
        scanned += 1
        text = row["text"].strip()
        if len(normalized_words(text)) < 50 or float(row.get("language_score", 1)) < .9:
            continue
        remove, tasks, reason = screen(text, grams, exact)
        if remove:
            removals.append({"document_id": row["id"], "tasks": ";".join(tasks), "reason": reason})
            continue
        ids = tokenizer.encode(text + "\n")
        # Stable document-ID partition; validation is never admitted to training.
        val = int.from_bytes(hashlib.sha256((str(SEED)+row["id"]).encode()).digest()[:8], "big") % 20 == 0
        (fw_val if val else fw_train).append(text)
        if val:
            validation_tokens += len(ids)
        else:
            accepted_tokens += len(ids)
        if accepted_tokens >= args.fineweb_tokens and validation_tokens >= 250_000:
            break
    del stream, refs, grams, exact

    # BabyLM uses a deterministic per-source tail holdout, preserving source balance.
    baby_val = []
    baby_train_path, baby_val_path = DATA/"babylm_train.int32", DATA/"babylm_validation.int32"
    baby_train_path.write_bytes(b""); baby_val_path.write_bytes(b"")
    baby_documents=0; baby_words=0; baby_chars=0; baby_tokens=0; baby_unknown=0
    baby_lengths=[]; baby_non_ascii=0; baby_hashes=set(); baby_duplicates=0; baby_counts=Counter()
    for path in baby_files:
        nonempty=sum(bool(x.strip()) for x in path.open(encoding="utf-8",errors="replace")); train_count=nonempty-max(1,nonempty//100)
        chunk=[]; seen=0
        with path.open(encoding="utf-8",errors="replace") as handle:
          for raw in handle:
            document=raw.strip()
            if not document: continue
            if seen >= train_count: baby_val.append(document); seen+=1; continue
            seen+=1; chunk.append(document); words=normalized_words(document); baby_words+=len(words); baby_chars+=len(document)
            if len(baby_lengths)<100_000: baby_lengths.append(len(words))
            baby_non_ascii += sum(ord(c)>=128 for c in document); key=hashlib.sha256(document.strip().encode()).digest()
            if len(baby_hashes)<100_000:
                if key in baby_hashes: baby_duplicates += 1
                baby_hashes.add(key)
            baby_documents += 1
            if len(chunk)>=10_000:
                ids=tokenizer.encode("\n".join(chunk))
                with baby_train_path.open("ab") as output: np.asarray(ids,dtype=np.int32).tofile(output)
                baby_tokens+=len(ids); baby_unknown+=ids.count(3); baby_counts.update(ids); chunk=[]
        if chunk:
            ids=tokenizer.encode("\n".join(chunk))
            with baby_train_path.open("ab") as output: np.asarray(ids,dtype=np.int32).tofile(output)
            baby_tokens+=len(ids); baby_unknown+=ids.count(3); baby_counts.update(ids)
    baby_val_ids=tokenizer.encode("\n".join(baby_val)); np.asarray(baby_val_ids,dtype=np.int32).tofile(baby_val_path)

    text_sets = {"finewebedu_train": fw_train, "finewebedu_validation": fw_val}
    token_sets = {name: tokenizer.encode("\n".join(docs)) for name, docs in text_sets.items()}
    binary = {name: save_tokens(DATA / f"{name}.int32", ids) for name, ids in token_sets.items()}
    binary["babylm_train"]={"path":str(baby_train_path.relative_to(ROOT)),"tokens":baby_tokens,"bytes":baby_train_path.stat().st_size,"sha256":digest(baby_train_path)}
    binary["babylm_validation"]={"path":str(baby_val_path.relative_to(ROOT)),"tokens":len(baby_val_ids),"bytes":baby_val_path.stat().st_size,"sha256":digest(baby_val_path)}
    # Exactly balanced source-family token suite, fixed once here.
    common_each = min(250_000, len(baby_val_ids), len(token_sets["finewebedu_validation"]))
    common = baby_val_ids[:common_each] + token_sets["finewebedu_validation"][:common_each]
    binary["common_validation"] = save_tokens(DATA / "common_validation.int32", common)

    now = datetime.now(timezone.utc).date().isoformat()
    baby_manifest = {"dataset": BABY_REPO, "revision": BABY_REVISION, "url": f"https://huggingface.co/datasets/{BABY_REPO}",
        "download_date_utc": now, "license": "MIT (repository dataset card)", "word_budget": "Strict 100M-word track",
        "compatibility_note": "Same BabyLM 2026 release family as the previously pinned Strict-Small corpus; no replacement occurred.",
        "source_composition": [x["file"].replace(".train.txt", "") for x in sources], "files": sources,
        "split_policy": "last 1% nonempty lines of every source held out", "token_files": {k:v for k,v in binary.items() if k.startswith("babylm")}}
    fw_manifest = {"dataset": FINEWEB_REPO, "configuration": FINEWEB_CONFIG, "revision": FINEWEB_REVISION,
        "url": f"https://huggingface.co/datasets/{FINEWEB_REPO}", "download_date_utc": now,
        "license": "ODC-By 1.0 (dataset card); source pages retain their own terms", "seed": SEED,
        "procedure": "datasets streaming; shuffle(seed=1337, buffer_size=1000); stop after >=20M accepted training tokens and >=250K validation tokens",
        "filters": ["at least 50 normalized words", "language_score >= 0.9", "benchmark decontamination", "stable SHA-256 5% validation partition"],
        "documents_scanned": scanned, "documents_training": len(fw_train), "documents_validation": len(fw_val),
        "approximate_words": sum(len(normalized_words(x)) for x in fw_train), "token_files": {k:v for k,v in binary.items() if k.startswith("fineweb")}}
    (ARTIFACTS/"babylm_strict_100m_manifest.json").write_text(json.dumps(baby_manifest, indent=2)+"\n")
    (ARTIFACTS/"fineweb_edu_sample_manifest.json").write_text(json.dumps(fw_manifest, indent=2)+"\n")
    common_manifest = {"frozen": True, "construction": "equal token prefixes from held-out source-family partitions",
        "tokens_per_source": common_each, "total_tokens": len(common), "tokenizer_sha256": digest(tok_path),
        "sources": {"BabyLM Strict": binary["babylm_validation"], "FineWeb-Edu": binary["finewebedu_validation"]},
        "common_token_file": binary["common_validation"]}
    (ARTIFACTS/"common_validation_manifest.json").write_text(json.dumps(common_manifest, indent=2)+"\n")

    task_counts = Counter(task for x in removals for task in x["tasks"].split(";") if task)
    import csv
    with (ARTIFACTS/"data_decontamination.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=["document_id","tasks","reason"],lineterminator="\n"); w.writeheader(); w.writerows(removals)
    decon_summary = {"documents_before_filtering": scanned, "documents_removed": len(removals),
        "documents_accepted_train": len(fw_train), "documents_accepted_validation": len(fw_val),
        "threshold": "normalized exact benchmark string >=80 characters OR at least two matching normalized 13-word n-grams",
        "task_counts_nonexclusive": dict(task_counts), "reference_splits": "validation and test; screening only"}
    (ARTIFACTS/"data_decontamination_summary.json").write_text(json.dumps(decon_summary,indent=2)+"\n")
    quality_baby={"dataset":"BabyLM Strict","documents":baby_documents,"words":baby_words,"characters":baby_chars,
        "tokens":baby_tokens,"chars_per_token":baby_chars/baby_tokens,"words_per_token":baby_words/baby_tokens,
        "tokens_per_word":baby_tokens/baby_words,"unknown_tokens":baby_unknown,"unknown_rate":baby_unknown/baby_tokens,
        "token_length_p50_document":statistics.median(baby_lengths),"token_length_p95_document":sorted(baby_lengths)[math.floor(.95*len(baby_lengths))],
        "extremely_short_document_rate":sum(x<50 for x in baby_lengths)/baby_documents,"non_ascii_character_rate":baby_non_ascii/baby_chars,
        "exact_duplicate_rate":baby_duplicates/max(1,len(baby_hashes)),"token_vocabulary_observed":len(baby_counts),
        "token_repetition_rate":sum(v for v in baby_counts.values() if v>1)/baby_tokens,"byte_compression_ratio":baby_chars/(4*baby_tokens),
        "fallback_behavior":"byte-level; no UNK expected"}
    quality = [quality_baby, diagnostics("FineWeb-Edu", fw_train, token_sets["finewebedu_train"], tokenizer)]
    with (ARTIFACTS/"data_quality_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(quality[0]),lineterminator="\n"); w.writeheader(); w.writerows(quality)
    print(json.dumps({"baby_manifest": baby_manifest, "fineweb_manifest": fw_manifest,
                      "decontamination": decon_summary, "quality": quality}, indent=2))


if __name__ == "__main__":
    main()
