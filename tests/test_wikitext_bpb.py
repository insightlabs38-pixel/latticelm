import math

import pytest

from scripts.evaluate_wikitext103 import bits_per_byte


def test_bits_per_byte_converts_nats_and_counts_utf8_bytes():
    text = "Aé"
    nll = 6.0 * math.log(2)
    assert len(text.encode("utf-8")) == 3
    assert bits_per_byte(nll, len(text.encode("utf-8"))) == pytest.approx(2.0)


def test_bits_per_byte_rejects_empty_denominator():
    with pytest.raises(ValueError):
        bits_per_byte(1.0, 0)
