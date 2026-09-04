import torch

from scripts.run_phase7b_final import PermutedBlocks


def test_stream_resume_and_without_replacement():
    tokens = torch.arange(129 * 17)
    uninterrupted = PermutedBlocks(tokens, 128, 314170)
    starts = [int(uninterrupted.one()[0][0]) for _ in range(17)]
    assert len(set(starts)) == 17
    resumed = PermutedBlocks(tokens, 128, 314170)
    resumed.draws = 9
    assert [int(resumed.one()[0][0]) for _ in range(8)] == starts[9:]
