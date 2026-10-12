"""Train the missing 16K tokenizer on the exact frozen 4K/8K raw sample."""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from final_recipe_prepare import CACHE, MATH_SAMPLE, final_corpus_rows
from posthack_campaign import RESEARCH, ROOT, atomic_json, sha256, utc

EXPECTED_CACHE_SHA256 = "56abab5e5ec495ab768ece1052252ea30569cd1be8060b08f8119ac78ea56c49"
EXPECTED_MATH_SHA256 = "74f124fc2c609cbeaddcdd797e1f02e47f3151cb2cf6e17e710479d2462ef59b"


def train_16k() -> dict:
    if sha256(CACHE) != EXPECTED_CACHE_SHA256 or sha256(MATH_SAMPLE) != EXPECTED_MATH_SHA256:
        raise RuntimeError("the frozen common tokenizer-training sample has changed")
    out = RESEARCH / "tokenizers"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "final_corpus_16k.json"
    if destination.exists():
        if (out / "16k_report.json").exists():
            raise RuntimeError("completed 16K tokenizer already exists; refusing overwrite")
        # An interrupted unregistered tokenizer build may leave a partial or
        # unreported artifact; retrain it from the verified frozen sample.
        destination.unlink()
    samples: dict[str, list[str]] = defaultdict(list)
    sample_length: dict[str, int] = defaultdict(int)
    for source, content in final_corpus_rows():
        if sample_length[source] < 2_000_000:
            samples[source].append(content)
            sample_length[source] += len(content)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=16384, min_frequency=2,
                         special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    started = time.perf_counter()
    tokenizer.train_from_iterator((text for _, text in final_corpus_rows()), trainer=trainer)
    if tokenizer.get_vocab_size() != 16384:
        raise RuntimeError("16K tokenizer did not reach requested vocabulary")
    tokenizer.save(str(destination))
    metrics = {}
    for source, pieces in samples.items():
        content = "\n".join(pieces)
        count = len(tokenizer.encode(content).ids)
        metrics[source] = {"utf8_bytes": len(content.encode("utf-8")),
                           "characters": len(content), "words": len(content.split()),
                           "tokens": count,
                           "bytes_per_token": len(content.encode("utf-8")) / count,
                           "characters_per_token": len(content) / count,
                           "words_per_token": len(content.split()) / count}
    result = {"schema": "posthack-16k-tokenizer-audit-v1", "created_utc": utc(),
              "training_sample": {"raw_cache_sha256": EXPECTED_CACHE_SHA256,
                                  "finemath_sample_sha256": EXPECTED_MATH_SHA256,
                                  "same_sample_as_4k_8k": True},
              "tokenizer_path": str(destination.relative_to(ROOT)),
              "tokenizer_sha256": sha256(destination), "artifact_bytes": destination.stat().st_size,
              "training_seconds": time.perf_counter() - started,
              "vocab_size": tokenizer.get_vocab_size(), "source_metrics": metrics,
              "embedding_economics_at_16m_width328": {
                  str(vocab): {"untied_parameters": 2 * vocab * 328,
                               "tied_parameters": vocab * 328}
                  for vocab in (4096, 8192, 16384)}}
    atomic_json(out / "16k_report.json", result)
    return result


def main() -> None:
    print(json.dumps(train_16k(), indent=2), flush=True)


if __name__ == "__main__":
    main()
