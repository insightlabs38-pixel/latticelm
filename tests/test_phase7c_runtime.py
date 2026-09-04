import hashlib

import torch

from latticelm.final_data import PermutedBlocks


def _hash(pair: tuple[torch.Tensor, torch.Tensor]) -> str:
    return hashlib.sha256(pair[0].numpy().tobytes() + pair[1].numpy().tobytes()).hexdigest()


def test_phase7c_stream_fresh_process_equivalence_and_resume():
    tokens = torch.arange(128 * 101 + 1)
    first = PermutedBlocks(tokens, 128, 314170)
    expected = [_hash(first.one()) for _ in range(24)]
    second = PermutedBlocks(tokens, 128, 314170)
    assert [_hash(second.one()) for _ in range(24)] == expected
    resumed = PermutedBlocks(tokens, 128, 314170)
    resumed.draws = 13
    assert [_hash(resumed.one()) for _ in range(11)] == expected[13:]
