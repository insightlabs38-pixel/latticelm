"""Paired capability uncertainty from the common evaluator's per-example samples."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

COMPETITION = ("hellaswag", "arc_easy", "piqa", "winogrande")


def paired_capability(candidate_path: Path, control_path: Path,
                      tasks: tuple[str, ...] = COMPETITION,
                      *, bootstrap_draws: int = 2000, seed: int = 2026) -> dict:
    candidate = json.loads(candidate_path.read_text())
    control = json.loads(control_path.read_text())
    rng = np.random.default_rng(seed)
    task_rows = {}
    task_bootstrap = []
    for task in tasks:
        def indexed(payload):
            rows = {}
            for row in payload[task]:
                key = (str(row["doc_id"]), row["target_hash"])
                if key in rows:
                    raise RuntimeError(f"duplicate capability sample identity in {task}")
                rows[key] = float(row["acc"])
            return rows
        new, old = indexed(candidate), indexed(control)
        if not new or set(new) != set(old):
            raise RuntimeError(f"unpaired or empty capability samples in {task}")
        differences = np.asarray([new[key] - old[key] for key in sorted(new)], dtype=np.float64)
        if not np.isin(differences, (-1, 0, 1)).all():
            raise RuntimeError(f"capability accuracy is not binary in {task}")
        indices = rng.integers(0, len(differences), size=(bootstrap_draws, len(differences)))
        draws = differences[indices].mean(axis=1)
        task_bootstrap.append(draws)
        gains, losses = int((differences > 0).sum()), int((differences < 0).sum())
        task_rows[task] = {"n": len(differences), "candidate_minus_control": float(differences.mean()),
                           "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                           "discordant_candidate_wins": gains,
                           "discordant_control_wins": losses,
                           "mcnemar_exact_p_two_sided": float(binomtest(min(gains, losses),
                                                                         gains + losses, .5).pvalue)
                           if gains + losses else 1.0}
    aggregate = np.mean(task_bootstrap, axis=0)
    return {"schema": "posthack-paired-capability-v1", "candidate_samples": str(candidate_path),
            "control_samples": str(control_path), "bootstrap_draws": bootstrap_draws,
            "seed": seed, "per_task": task_rows,
            "four_task_equal_weight_delta": float(np.mean([row["candidate_minus_control"]
                                                       for row in task_rows.values()])),
            "four_task_equal_weight_ci95": [float(np.quantile(aggregate, .025)),
                                            float(np.quantile(aggregate, .975))],
            "interpretation_limit": "paired evaluation-item uncertainty only; does not include training-seed or task-selection uncertainty"}
