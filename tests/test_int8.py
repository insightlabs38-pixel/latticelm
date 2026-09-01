import torch

from latticelm.int8 import RowwiseInt8Memory


def test_rowwise_int8_memory_has_small_error() -> None:
    torch.manual_seed(1337)
    tables = tuple(torch.randn(32, 8) for _ in range(3))
    ids = tuple(torch.randint(0, 32, (2, 4)) for _ in range(3))
    reference = sum(table[index] for table, index in zip(tables, ids))
    approximation = RowwiseInt8Memory(tables).gather(*ids)
    assert (reference - approximation).abs().max() < 0.05
