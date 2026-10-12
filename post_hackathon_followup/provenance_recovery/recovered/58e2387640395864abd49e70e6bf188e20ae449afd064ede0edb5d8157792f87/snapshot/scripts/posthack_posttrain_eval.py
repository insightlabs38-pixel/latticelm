"""Shared completion-loss check on pinned, disjoint skill validation examples."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from latticelm.config import LatticeConfig
from latticelm.posthack_variants import build_posthack_model
from posthack_campaign import RESEARCH, ROOT, atomic_json, sha256, utc
from posthack_posttrain import encode_example

TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
DATA = RESEARCH / "posttraining/data/heldout_skill_validation.jsonl"


def evaluate(checkpoint_path: Path, output: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = LatticeConfig(**checkpoint["config"])
    model = build_posthack_model(cfg, checkpoint.get("variant", "baseline"))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    totals = defaultdict(lambda: {"examples": 0, "completion_tokens": 0, "nll_nats": 0.0})
    with DATA.open(encoding="utf-8") as handle, torch.inference_mode():
        for line in handle:
            row = json.loads(line)
            x, y, mask, _ = encode_example(tokenizer, row, cfg.context_length)
            logits, _ = model(torch.from_numpy(x[None, :]))
            target = torch.from_numpy(y)
            selected = F.cross_entropy(logits[0], target, reduction="none")
            count = int(mask.sum())
            entry = totals[row["resource"]]
            entry["examples"] += 1
            entry["completion_tokens"] += count
            entry["nll_nats"] += float(selected[torch.from_numpy(mask)].sum())
    result = {"schema": "posthack-heldout-skill-completion-v1",
              "created_utc": utc(), "checkpoint_sha256": sha256(checkpoint_path),
              "tokenizer_sha256": sha256(TOKENIZER), "data_sha256": sha256(DATA),
              "context_length": cfg.context_length,
              "prompt_policy": "same tokenization and answer mask as post-training; at most 128 pinned validation examples per resource",
              "by_resource": {name: {**values,
                                     "nll_nats_per_completion_token": values["nll_nats"] / values["completion_tokens"]}
                              for name, values in sorted(totals.items())},
              "aggregate_nll_nats_per_completion_token":
                  sum(value["nll_nats"] for value in totals.values()) /
                  sum(value["completion_tokens"] for value in totals.values())}
    if set(result["by_resource"]) != {"sciq", "commonsense_qa", "qasc"}:
        raise RuntimeError("pinned skill validation resources are incomplete")
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(json.dumps(evaluate(args.checkpoint, args.output), sort_keys=True), flush=True)
