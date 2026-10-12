from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from latticelm.data_d import SourceStream
from latticelm.research_sampling import ResearchSourceStream
from posthack_train import next_batch_hash
from scripts.train_final_production import WeightedMixture


def fixture(mode: str) -> ResearchSourceStream:
    values = np.arange(257, dtype="<i4")
    maps = {"fixture.int32": {"starts": [0, 128], "ends": [128, 257],
                              "difficulty": [0.0, 1.0], "quality": [1.0, 0.0],
                              "capability": [1.0, 0.0]}}
    return ResearchSourceStream([values], 8, 41, ["fixture.int32"], maps,
                                "fineweb_edu", mode)


def test_research_sampler_at_flat_target_matches_baseline_exactly() -> None:
    values = np.arange(257, dtype="<i4")
    baseline = SourceStream([values], 8, 41)
    research = fixture("curriculum")
    research.set_progress(1.0)
    for _ in range(100):
        expected = baseline.one()
        actual = research.one()
        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1], expected[1])
    assert research.draws == baseline.draws
    assert research.rejected == 0


def test_curriculum_and_quality_rules_bias_then_flatten() -> None:
    curriculum = fixture("curriculum")
    curriculum.set_progress(0)
    for _ in range(200):
        curriculum.one()
    early = curriculum.accepted_difficulty_sum / curriculum.accepted
    assert early < .4
    assert curriculum.rejected > 0
    before = curriculum.rejected
    curriculum.set_progress(.25)
    for _ in range(100):
        curriculum.one()
    assert curriculum.rejected == before
    anneal = fixture("quality_anneal")
    anneal.set_progress(.9)
    for _ in range(100):
        x, _ = anneal.one()
        assert int(x[0]) < 128
    combined = fixture("curriculum_anneal")
    combined.set_progress(.9)
    for _ in range(20):
        x, _ = combined.one()
        assert int(x[0]) < 128


def test_capability_component_has_registered_frequency() -> None:
    stream = fixture("capability")
    for _ in range(1000):
        stream.one()
    fraction = stream.selected_component_count / stream.accepted
    assert .09 < fraction < .16


def test_sampler_hash_and_resume_preserve_draws_and_diagnostics() -> None:
    stream = fixture("curriculum")
    stream.set_progress(.1)
    mixture = WeightedMixture({"fineweb_edu": stream}, {"fineweb_edu": 1.0}, 2)
    before = (stream.draws, stream.accepted, stream.rejected, stream.accepted_difficulty_sum)
    expected_hash = next_batch_hash(mixture)
    assert before == (stream.draws, stream.accepted, stream.rejected, stream.accepted_difficulty_sum)
    state = mixture.state_dict()
    first = mixture.batch()
    clone = fixture("curriculum")
    clone.set_progress(.1)
    other = WeightedMixture({"fineweb_edu": clone}, {"fineweb_edu": 1.0}, 2)
    other.load_state_dict(state)
    assert next_batch_hash(other) == expected_hash
    resumed = other.batch()
    assert np.array_equal(first[0], resumed[0])
    assert np.array_equal(first[1], resumed[1])
