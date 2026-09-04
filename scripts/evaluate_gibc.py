"""One-shot zero-shot GIBC evaluation for native LatticeLM checkpoints."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from lm_eval import evaluator
from lm_eval.api.model import LM

from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.tokenizer import load_tokenizer


class LatticeHarnessLM(LM):
    def __init__(self, checkpoint: str, tokenizer: str, batch_size: int = 8):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.config = LatticeConfig(**payload["config"])
        self.model = build_model(self.config)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.tok = load_tokenizer(tokenizer)
        self.batch_size = batch_size
        self._device = torch.device("cpu")

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def loglikelihood(self, requests):
        prepared = []
        for index, request in enumerate(requests):
            context, continuation = request.args
            context_ids = self._encode(context) or [getattr(self.tok, "bos_id", 1)]
            continuation_ids = self._encode(continuation)
            joined = (context_ids + continuation_ids)[-(self.config.context_length + 1):]
            input_ids, targets = joined[:-1], joined[1:]
            scored = min(len(continuation_ids), len(targets))
            prepared.append((index, input_ids, targets, scored))
        groups = defaultdict(list)
        for item in prepared:
            groups[len(item[1])].append(item)
        output = [None] * len(prepared)
        with torch.inference_mode():
            for group in groups.values():
                for offset in range(0, len(group), self.batch_size):
                    chunk = group[offset:offset + self.batch_size]
                    x = torch.tensor([item[1] for item in chunk], dtype=torch.long)
                    logits, _ = self.model(x)
                    log_probs = F.log_softmax(logits, dim=-1)
                    for row, (index, _, targets, scored) in enumerate(chunk):
                        if not scored:
                            output[index] = (0.0, True); continue
                        target = torch.tensor(targets[-scored:], dtype=torch.long)
                        selected = log_probs[row, -scored:]
                        value = selected.gather(-1, target[:, None]).sum().item()
                        greedy = bool(torch.equal(selected.argmax(-1), target))
                        output[index] = (value, greedy)
        return output

    def loglikelihood_rolling(self, requests):
        results = []
        for request in requests:
            ids = [getattr(self.tok, "bos_id", 1)] + self._encode(request.args[0])
            total = 0.0
            with torch.inference_mode():
                for start in range(0, len(ids) - 1, self.config.context_length):
                    end = min(len(ids) - 1, start + self.config.context_length)
                    left = max(0, end - self.config.context_length)
                    x = torch.tensor(ids[left:end], dtype=torch.long)[None, :]
                    targets = torch.tensor(ids[left + 1:end + 1], dtype=torch.long)
                    logits, _ = self.model(x)
                    score_from = start - left
                    lp = F.log_softmax(logits[0, score_from:], -1)
                    total += lp.gather(-1, targets[score_from:, None]).sum().item()
            results.append(total)
        return results

    def generate_until(self, requests):
        raise NotImplementedError("The Phase 6 tasks use likelihood scoring only")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="hellaswag,arc_easy,piqa,winogrande,wikitext")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    lm = LatticeHarnessLM(args.checkpoint, args.tokenizer, args.batch_size)
    result = evaluator.simple_evaluate(model=lm, tasks=args.tasks.split(","), num_fewshot=0,
                                       batch_size=args.batch_size, limit=args.limit,
                                       confirm_run_unsafe_code=True)
    result["phase6_metadata"] = {"checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint), "tokenizer_sha256": sha256(args.tokenizer),
        "wall_seconds": time.perf_counter() - started, "lm_eval_version": "0.4.13",
        "pytorch_threads": args.threads}
    Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__": main()
