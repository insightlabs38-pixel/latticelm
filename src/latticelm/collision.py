"""Collision-aware exact-heavy-hitter / hashed-tail preprocessing experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .data import ensure_corpus
from .ngrams import stable_hash_ngram
from .tokenizer import BytePairTokenizer

ROOT = Path(__file__).resolve().parents[2]


def analyze(tokens: list[int], exact_budgets: tuple[int, int, int] = (1024, 1024, 512), tail_slots: int = 65536) -> dict:
    orders = {}
    for order, budget, seed in zip((2, 3, 4), exact_budgets, (0xB1, 0xC3, 0xD4)):
        counts = Counter(tuple(tokens[max(0, pos - order + 1) : pos + 1]) for pos in range(len(tokens)))
        exact = {ngram for ngram, _ in counts.most_common(budget)}
        tail = [ngram for ngram in counts if ngram not in exact]
        ids = {stable_hash_ngram(list(ngram), seed) % tail_slots for ngram in tail}
        exact_occurrences = sum(counts[ngram] for ngram in exact)
        orders[str(order)] = {
            "unique_ngrams": len(counts), "exact_entries": len(exact), "tail_unique_ngrams": len(tail),
            "tail_slots": tail_slots, "tail_occupied_slots": len(ids), "tail_collisions": len(tail) - len(ids),
            "exact_hit_rate": exact_occurrences / len(tokens),
        }
    return {"experiment": "collision_aware_memory_analysis", "tokens": len(tokens), "orders": orders}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--download", action="store_true")
    parser.add_argument("--tokenizer", default="artifacts/tokenizers/tinyshakespeare_100k_4k.json")
    args = parser.parse_args()
    text, source = ensure_corpus(None, args.download)
    tokens = BytePairTokenizer.load(ROOT / args.tokenizer).encode(text)
    result = analyze(tokens); result["data_source"] = source
    (ROOT / "artifacts" / "collision_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
