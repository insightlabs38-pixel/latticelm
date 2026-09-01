from __future__ import annotations

from pathlib import Path
import urllib.request

import torch

from .tokenizer import BytePairTokenizer, load_tokenizer

SMOKE_CORPUS = """The little model reads a small public-domain-style corpus. Language models predict the next token. A careful experiment measures what worked and what failed. Conditional memory stores causal local patterns. The dense baseline uses the same data, seed, context, and optimizer. Reproducible research keeps configurations, checkpoints, metrics, and negative results. """ * 120


def ensure_corpus(path: str | None, download: bool = False) -> tuple[str, str]:
    if path:
        source = Path(path)
        return source.read_text(encoding="utf-8", errors="replace"), str(source)
    if download:
        target = Path("artifacts/data/tinyshakespeare.txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", target)
        # A 100K-character deterministic prefix is ample for a short CPU
        # architecture-validation run and avoids an overnight preprocessing job.
        return target.read_text(encoding="utf-8")[:100_000], "Tiny Shakespeare 100K-character public subset (GitHub source)"
    return SMOKE_CORPUS, "built-in synthetic smoke corpus (correctness only; not a quality result)"


def make_data(text: str, vocab_size: int, tokenizer_path: Path, validation_text: str | None = None) -> tuple[BytePairTokenizer, torch.Tensor, torch.Tensor]:
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else BytePairTokenizer.train(text, vocab_size)
    if not tokenizer_path.exists():
        tokenizer.save(tokenizer_path)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if validation_text is not None:
        validation_ids = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
        return tokenizer, ids, validation_ids
    split = max(2, int(len(ids) * 0.9))
    return tokenizer, ids[:split], ids[split:]


def batch_from_tokens(tokens: torch.Tensor, batch_size: int, context: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= context + 1:
        raise ValueError("corpus is shorter than context length")
    starts = torch.randint(0, tokens.numel() - context - 1, (batch_size,), generator=generator)
    x = torch.stack([tokens[i : i + context] for i in starts.tolist()])
    y = torch.stack([tokens[i + 1 : i + context + 1] for i in starts.tolist()])
    return x, y
