"""Pin disjoint capability SFT and task-tuned diagnostic training examples."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from posthack_campaign import RESEARCH, atomic_json, sha256, utc

OUT = RESEARCH / "posttraining/data"
DATASETS = {
    "sciq": ("allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815", "data/"),
    "commonsense_qa": ("tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", "data/"),
    "qasc": ("allenai/qasc", "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", "data/"),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", "data/"),
    "arc_easy": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Easy/"),
    "piqa": ("baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1", "piqa_"),
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", "winogrande_xl/"),
}


def fingerprint(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt.lower()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def parquet_rows(name: str, split: str) -> tuple[list[dict], dict]:
    repo, revision, prefix = DATASETS[name]
    suffix = "validation" if name == "piqa" and split == "validation" else split
    filename = (f"{prefix}{suffix}.parquet" if name == "piqa"
                else f"{prefix}{suffix}-00000-of-00001.parquet")
    path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    return pq.read_table(path).to_pylist(), {"repo": repo, "revision": revision,
                                              "path": filename, "sha256": sha256(path)}


def prompt_answer(name: str, row: dict, labeled: bool = True) -> tuple[str, str | None]:
    if name == "sciq":
        return f"Science context: {row['support']}\nQuestion: {row['question']}\nAnswer:", row["correct_answer"]
    if name == "commonsense_qa":
        choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
        return f"Question: {row['question']}\nAnswer:", choices.get(row.get("answerKey"))
    if name == "qasc":
        choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
        return f"Facts: {row['fact1']} {row['fact2']}\nQuestion: {row['question']}\nAnswer:", choices.get(row.get("answerKey"))
    if name == "hellaswag":
        index = row.get("label")
        return f"Continue naturally: {row['ctx']}", row["endings"][int(index)] if labeled and index not in (None, "") else None
    if name == "arc_easy":
        choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
        return f"Question: {row['question']}\nAnswer:", choices.get(row.get("answerKey"))
    if name == "piqa":
        index = row.get("label")
        return f"Physical task: {row['goal']}\nBest way:", row[f"sol{int(index)+1}"] if labeled and index is not None else None
    if name == "winogrande":
        answer = row.get("answer")
        return f"Resolve the blank: {row['sentence']}\nBlank:", row[f"option{answer}"] if labeled and answer in ("1", "2") else None
    raise ValueError(name)


def build() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    generic_path, direct_path = OUT / "generic_train.jsonl", OUT / "task_train.jsonl"
    if generic_path.exists() or direct_path.exists():
        raise RuntimeError("posttraining data already exists; refusing overwrite")
    all_provenance = {}
    counters = Counter()
    heldout_hashes = set()
    for name in ("hellaswag", "arc_easy", "piqa", "winogrande"):
        for split in (("validation", "test") if name == "arc_easy" else ("validation",)):
            rows, provenance = parquet_rows(name, split)
            all_provenance[f"{name}/{split}"] = provenance
            heldout_hashes.update(fingerprint(prompt_answer(name, row, labeled=False)[0]) for row in rows)
    for kind, names, destination in (
        ("generic", ("sciq", "commonsense_qa", "qasc"), generic_path),
        ("task_tuned", ("hellaswag", "arc_easy", "piqa", "winogrande"), direct_path),
    ):
        with destination.open("w", encoding="utf-8") as out:
            for name in names:
                rows, provenance = parquet_rows(name, "train")
                all_provenance[f"{name}/train"] = provenance
                for index, row in enumerate(rows):
                    prompt, answer = prompt_answer(name, row)
                    if not answer:
                        counters[f"{name}/unlabeled"] += 1
                        continue
                    digest = fingerprint(prompt)
                    if digest in heldout_hashes:
                        counters[f"{name}/heldout_exact_prompt_excluded"] += 1
                        continue
                    out.write(json.dumps({"kind": kind, "resource": name,
                                          "source_index": index, "prompt": prompt,
                                          "answer": answer, "prompt_sha256": digest},
                                         ensure_ascii=False, sort_keys=True) + "\n")
                    counters[f"{name}/included"] += 1
            out.flush()
            os.fsync(out.fileno())
    report = {"schema": "posthack-posttraining-data-v1", "created_utc": utc(),
              "files": {str(path.relative_to(RESEARCH)): sha256(path)
                        for path in (generic_path, direct_path)},
              "counts": dict(counters), "resources": all_provenance,
              "contamination_policy": "Only pinned TRAIN splits enter training; exact normalized prompt hashes from official held-out validation and ARC-Easy test are excluded",
              "comparison_label": "task_train is diagnostic benchmark-task tuning; never compare it as pretraining-only"}
    atomic_json(OUT / "manifest.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2), flush=True)
