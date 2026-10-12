"""Full eight-task 4K prompt and truncation audit using evaluator tokenization."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lm_eval.tasks import TaskManager, get_task_dict

from evaluate_gibc import encode_pair
from latticelm.tokenizer import load_tokenizer
from posthack_campaign import RESEARCH, ROOT, atomic_json, sha256, utc

TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande",
         "arc_challenge", "openbookqa", "boolq", "lambada_openai")
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
THRESHOLDS = (128, 256, 512, 1024)


def request_lengths(tokenizer, context: str, continuation: str,
                    window: int = 256) -> dict:
    context_ids, continuation_ids = encode_pair(tokenizer, context, continuation)
    total = len(context_ids) + len(continuation_ids)
    removed = max(0, total - (window + 1))
    dropped_context = min(removed, len(context_ids))
    dropped_continuation = max(0, removed - len(context_ids))
    return {"prompt_tokens": len(context_ids), "continuation_tokens": len(continuation_ids),
            "total_tokens": total, "dropped_context_tokens": dropped_context,
            "dropped_continuation_tokens": dropped_continuation,
            "meaningfully_truncated": dropped_context > 0 or dropped_continuation > 0}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"requests": 0}
    totals = sorted(row["total_tokens"] for row in rows)
    count = len(rows)
    return {"requests": count,
            "prompt_tokens_max": max(row["prompt_tokens"] for row in rows),
            "continuation_tokens_max": max(row["continuation_tokens"] for row in rows),
            "total_tokens_max": totals[-1],
            "mean_total_tokens": sum(totals) / count,
            "total_tokens_p95": totals[int(.95 * (count - 1))],
            "fraction_crossing": {str(limit): sum(row["total_tokens"] > limit for row in rows) / count
                                  for limit in THRESHOLDS},
            "meaningfully_truncated_fraction_at_256": sum(row["meaningfully_truncated"] for row in rows) / count,
            "dropped_context_tokens": sum(row["dropped_context_tokens"] for row in rows),
            "dropped_continuation_tokens": sum(row["dropped_continuation_tokens"] for row in rows)}


def audit(limit: int | None = None, *, tokenizer_path: Path = TOKENIZER,
          detail_path: Path | None = None) -> dict:
    tokenizer = load_tokenizer(str(tokenizer_path))
    tasks = get_task_dict(list(TASKS), task_manager=TaskManager())
    by_task = {}
    detail_path = detail_path or RESEARCH / "prompt_request_lengths.jsonl"
    detail_temp = detail_path.with_suffix(".jsonl.tmp")
    detail = detail_temp.open("w", encoding="utf-8") if limit is None else None
    for name, task in tasks.items():
        task.build_all_requests(limit=limit, rank=0, world_size=1,
                                cache_requests=False, rewrite_requests_cache=False)
        rows = []
        rolling = 0
        for request_index, request in enumerate(task.instances):
            if request.request_type == "loglikelihood":
                lengths = request_lengths(tokenizer, *request.args)
                rows.append(lengths)
                if detail:
                    detail.write(json.dumps({"task": name, "request_index": request_index,
                                             **lengths}, sort_keys=True) + "\n")
            elif request.request_type == "loglikelihood_rolling":
                rolling += 1
            else:
                raise RuntimeError(f"unsupported audit request type: {request.request_type}")
        by_task[name] = {**summarize(rows), "rolling_requests": rolling}
    if detail:
        detail.flush()
        os.fsync(detail.fileno())
        detail.close()
        os.replace(detail_temp, detail_path)
    max_fraction = max((row.get("meaningfully_truncated_fraction_at_256", 0)
                        for row in by_task.values()), default=0)
    result = {"schema": "posthack-full-prompt-audit-v1", "created_utc": utc(),
              "tokenizer_sha256": sha256(tokenizer_path),
              "tasks": by_task, "limited_examples_per_task": limit,
              "per_request_lengths_sha256": sha256(detail_path) if limit is None else None,
              "context512_trigger_fraction": .015,
              "max_task_meaningful_truncation_fraction_at_256": max_fraction,
              "train_context512": bool(limit is None and max_fraction >= .015),
              "windowing_policy": "joined context/continuation tokenization; evaluator keeps final context_length+1 tokens and scores surviving continuation; rolling requests use separate windows",
              "decision_rule": "train 512 only if any full-suite task has >=1.5% requests with lost context or continuation at 256"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detail-output", type=Path)
    args = parser.parse_args()
    result = audit(args.limit, tokenizer_path=args.tokenizer,
                   detail_path=args.detail_output)
    destination = args.output or RESEARCH / ("prompt_truncation_audit.json" if args.limit is None else
                                            "prompt_truncation_audit_smoke.json")
    atomic_json(destination, result)
    print(json.dumps({"tasks": len(result["tasks"]),
                      "max_truncation_fraction": result["max_task_meaningful_truncation_fraction_at_256"],
                      "train_context512": result["train_context512"]}), flush=True)


if __name__ == "__main__":
    main()
