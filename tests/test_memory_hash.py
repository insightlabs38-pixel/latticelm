import torch

from latticelm.ngrams import causal_ngram_ids, stable_hash_ngram


def test_hash_is_stable_and_not_python_hash() -> None:
    assert stable_hash_ngram([1, 2, 3], 0xC3) == stable_hash_ngram([1, 2, 3], 0xC3)
    assert stable_hash_ngram([1, 2, 3], 0xC3) != stable_hash_ngram([3, 2, 1], 0xC3)


def test_ids_have_expected_shape() -> None:
    ids = causal_ngram_ids(torch.tensor([[1, 2, 3, 4]]), 97)
    assert all(value.shape == (1, 4) and int(value.max()) < 97 for value in ids)
