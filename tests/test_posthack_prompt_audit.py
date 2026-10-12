from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from posthack_prompt_audit import request_lengths, summarize


class CharacterTokenizer:
    bos_id = 1

    def encode(self, value):
        return [ord(character) for character in value]


def test_prompt_audit_counts_evaluator_left_truncation():
    tokenization = CharacterTokenizer()
    row = request_lengths(tokenization, "a" * 260, "b" * 5)
    assert row["prompt_tokens"] == 260
    assert row["continuation_tokens"] == 5
    assert row["total_tokens"] == 265
    assert row["dropped_context_tokens"] == 8
    assert row["dropped_continuation_tokens"] == 0
    report = summarize([row, request_lengths(tokenization, "short", "answer")])
    assert report["fraction_crossing"]["256"] == .5
    assert report["meaningfully_truncated_fraction_at_256"] == .5
