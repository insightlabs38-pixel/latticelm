from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


class BytePairTokenizer:
    """Small deterministic byte-level BPE trained only on supplied corpus text."""
    bos_id, eos_id, byte_offset = 1, 2, 4

    def __init__(self, merges: list[tuple[int, int]], vocab_size: int | None = None) -> None:
        self.merges = [tuple(pair) for pair in merges]
        self.merge_to_id = {pair: 260 + i for i, pair in enumerate(self.merges)}
        self._vocab_size = max(260 + len(self.merges), vocab_size or 0)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BytePairTokenizer":
        # A fixed sample bounds tokenizer construction on a laptop while keeping
        # the sample deterministic and entirely inside the allowed corpus.
        sample = text.encode("utf-8")[:20_000]
        words = [[byte + cls.byte_offset for byte in piece] for piece in sample.split()]
        merges: list[tuple[int, int]] = []
        # Full naive BPE is quadratic in vocabulary size. 128 learned merges
        # keeps CPU preparation bounded; remaining vocabulary IDs are reserved
        # so the model still has the configured 4K tied vocabulary capacity.
        merge_limit = min(vocab_size, 260 + 128)
        while 260 + len(merges) < merge_limit:
            counts: Counter[tuple[int, int]] = Counter()
            for word in words:
                counts.update(zip(word, word[1:]))
            if not counts:
                break
            pair, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break
            new_id = 260 + len(merges)
            merges.append(pair)
            for i, word in enumerate(words):
                joined: list[int] = []
                j = 0
                while j < len(word):
                    if j + 1 < len(word) and (word[j], word[j + 1]) == pair:
                        joined.append(new_id); j += 2
                    else:
                        joined.append(word[j]); j += 1
                words[i] = joined
        return cls(merges, vocab_size)

    def encode(self, text: str) -> list[int]:
        values = [byte + self.byte_offset for byte in text.encode("utf-8")]
        for pair, new_id in self.merge_to_id.items():
            output: list[int] = []
            i = 0
            while i < len(values):
                if i + 1 < len(values) and (values[i], values[i + 1]) == pair:
                    output.append(new_id); i += 2
                else:
                    output.append(values[i]); i += 1
            values = output
        return values

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges, "vocab_size": self.vocab_size}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["merges"], payload.get("vocab_size"))


class HuggingFaceBPE:
    """Thin adapter around a fully learned `tokenizers` BPE JSON file."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def save(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "HuggingFaceBPE":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(path)))


def load_tokenizer(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "model" in payload and payload.get("version"):
        return HuggingFaceBPE.load(path)
    return BytePairTokenizer.load(path)
