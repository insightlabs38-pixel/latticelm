"""Train a deterministic, real 4K byte-level BPE on the research corpus."""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def normalized_identity(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return " ".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()
    started = time.perf_counter()
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(vocab_size=args.vocab_size, min_frequency=2,
                         special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train([args.corpus], trainer)
    if tokenizer.get_vocab_size() != args.vocab_size:
        raise RuntimeError(f"requested {args.vocab_size} tokens but learned {tokenizer.get_vocab_size()}")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    text = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    sample = text[:1_000_000]
    encoded = tokenizer.encode(sample).ids
    key_to_id: dict[str, int] = {}
    memory_map = []
    for token_id in range(tokenizer.get_vocab_size()):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        key = normalized_identity(decoded) or decoded
        if "�" in decoded:
            key = tokenizer.id_to_token(token_id)
        memory_map.append(key_to_id.setdefault(key, len(key_to_id)))
    report = {
        "corpus": str(Path(args.corpus).resolve()), "corpus_bytes": len(text.encode()),
        "corpus_words_whitespace": len(text.split()), "vocab_size": tokenizer.get_vocab_size(),
        "learned_merges": tokenizer.get_vocab_size() - 256 - 4,
        "sample_characters": len(sample), "sample_tokens": len(encoded),
        "average_chars_per_token": len(sample) / len(encoded),
        "average_tokens_per_word": len(encoded) / len(sample.split()),
        "training_seconds": time.perf_counter() - started,
        "memory_identity_size": len(key_to_id), "memory_token_map": memory_map,
        "normalization": "NFKC -> NFD/strip accents -> lowercase -> collapse whitespace",
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "memory_token_map"}, indent=2))


if __name__ == "__main__":
    main()
