from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from posthack_uncertainty import paired_capability


def test_paired_accuracy_uses_identical_item_identities_and_reports_mcnemar(tmp_path):
    candidate = tmp_path / "candidate.json"
    control = tmp_path / "control.json"
    tasks = ("task_a", "task_b")
    before = [{"doc_id": str(index), "target_hash": str(index), "acc": 0.0}
              for index in range(12)]
    after = [{**row, "acc": 1.0} for row in before]
    candidate.write_text(json.dumps({task: list(reversed(after)) for task in tasks}))
    control.write_text(json.dumps({task: before for task in tasks}))
    result = paired_capability(candidate, control, tasks, bootstrap_draws=100)
    assert result["four_task_equal_weight_delta"] == 1.0
    assert result["four_task_equal_weight_ci95"] == [1.0, 1.0]
    assert result["per_task"]["task_a"]["mcnemar_exact_p_two_sided"] < .001
    corrupt = {task: before for task in tasks}
    corrupt["task_b"] = before[:-1]
    control.write_text(json.dumps(corrupt))
    with pytest.raises(RuntimeError, match="unpaired"):
        paired_capability(candidate, control, tasks, bootstrap_draws=100)
