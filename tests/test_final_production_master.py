import json
from pathlib import Path

import pytest

import scripts.run_final_production_master as master
from scripts.train_final_production import choose_downshift, semantic_milestones


def test_state_machine_contract_is_closed_and_ordered():
    assert master.PHASES[0] == "INSPECT_AND_FREEZE"
    assert master.PHASES[-1] == "PRODUCTION_COMPLETE"
    assert master.PHASES.index("FINAL_RECIPE_FREEZE") < master.PHASES.index("FINAL_LONG_RUN_FRESH")
    assert master.PHASES.index("FINAL_LONG_RUN_FRESH") < master.PHASES.index("FINAL_CHECKPOINT_EVALUATION")
    assert set(master.HANDLERS) == set(master.PHASES)


@pytest.mark.parametrize("hours,expected", [(250, 2_000_000_000), (200, 1_500_000_000)])
def test_budget_selection_reserves_fifty_execution_hours(hours, expected):
    selected = master.select_budget(hours, 3157.573615954999)
    assert selected == expected
    assert selected / (3157.573615954999 * .90) / 3600 <= hours - 50


def test_budget_rounds_down_and_never_violates_reserve():
    selected = master.select_budget(100, 3000)
    assert selected % 50_000_000 == 0
    assert selected / (3000 * .90) / 3600 <= 50


def test_semantic_milestones_preserve_decay_evidence():
    points = semantic_milestones(2_000_000_000)
    assert points == {"early-health": 10_000_000, "quarter": 500_000_000, "half": 1_000_000_000,
                      "end-stable": 1_700_000_000, "mid-decay": 1_850_000_000, "final": 2_000_000_000}


def test_downshift_never_increases_target_and_preserves_reserve():
    now = 1_000_000.0; deadline = now + 100 * 3600
    selected = choose_downshift(2_250_000_000, 100_000_000, 3000, deadline, now=now)
    assert 100_000_000 <= selected <= 2_250_000_000
    capacity = 100_000_000 + (50 * 3600) * 3000 * .90
    assert selected <= capacity


def test_dry_run_declares_failure_fallthrough_and_fresh_guard():
    result = master.dry_run()
    assert result["status"] == "DRY_RUN_PASS"
    assert "TRITON_REJECTED" in result["optional_failures_fall_through"]
    assert "MUON_REJECTED" in result["optional_failures_fall_through"]
    assert result["pilot_initialization_forbidden"] is True
    assert "refuses existing" in result["fresh_only_guard"]


def test_paired_bootstrap_direction_and_determinism():
    first = master.paired_bootstrap([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], draws=100)
    second = master.paired_bootstrap([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], draws=100)
    assert first == second
    assert first["mean_difference"] == -1.0
    assert first["ci95"] == [-1.0, -1.0]


def test_muon_uses_reference_consistent_adamw_lr():
    import torch
    from latticelm.config import LatticeConfig
    from latticelm.final_recipe import make_optimizer
    from latticelm.model import build_model
    cfg = LatticeConfig(vocab_size=512, d_model=96, n_layers=1, n_heads=4, n_kv_heads=2,
                        ffn_hidden=192, context_length=16, architecture="co4_causal", real_gqa=True)
    bundle, _ = make_optimizer(build_model(cfg), "muon_hybrid", 8e-4, .1, (.9, .95))
    assert bundle.optimizers[0].param_groups[0]["lr"] == 8e-4
    assert bundle.optimizers[0].param_groups[0]["adjust_lr_fn"] == "match_rms_adamw"


def test_service_and_artifact_contracts_present():
    service = Path("ops/latticelm-final-production-20260917.service").read_text()
    assert "Restart=on-failure" in service and "KillMode=mixed" in service
    text = Path("scripts/run_final_production_master.py").read_text()
    for name in ("final_submission_results.json", "final_submission_results.csv", "final_training_curve.csv",
                 "final_training_events.jsonl", "final_model_facts.json", "final_experiment_decisions.json",
                 "final_checkpoint_comparison.md", "final_production_report.md", "demo_metrics.json", "demo_timeline.json"):
        assert name in text


def test_no_unauthorized_research_axes_in_state_machine():
    phases = " ".join(master.PHASES).lower()
    for forbidden in ("tokenizer_search", "architecture_search", "context_search", "data_mixture", "rlvr", "grpo", "dpo"):
        assert forbidden not in phases


def test_simulated_triton_timeout_falls_through(monkeypatch, tmp_path):
    written = {}
    interpreter = tmp_path / "python"; interpreter.write_text("")
    monkeypatch.setattr(master, "TRITON_PYTHON", interpreter)
    monkeypatch.setattr(master, "run_child", lambda *a, **k: (124, "timeout"))
    monkeypatch.setattr(master, "atomic", lambda path, payload: written.update({path.name: payload}))
    monkeypatch.setattr(master, "complete", lambda state, phase, **fields: state.update(fields, phase="MUON_IMPLEMENTATION_AUDIT"))
    state = {"research_seconds": 0, "measured_tps": 3157.57}
    master.phase_triton(state)
    assert state["triton"]["status"] == "TRITON_REJECTED"
    assert state["phase"] == "MUON_IMPLEMENTATION_AUDIT"


def test_triton_gate_uses_existing_isolated_interpreter():
    assert master.TRITON_PYTHON == master.ROOT / ".triton-cpu-venv/bin/python"
    assert master.TRITON_PYTHON.is_file()


def test_simulated_muon_rejection_falls_through(monkeypatch):
    monkeypatch.setattr(master, "atomic", lambda *a, **k: None)
    monkeypatch.setattr(master, "complete", lambda state, phase, **fields: state.update(fields, phase="OPTIMIZER_FREEZE"))
    state = {"muon_audit": {"status": "MUON_REJECTED_IMPLEMENTATION"}}
    master.phase_muon_screen(state)
    assert state["muon"]["status"] == "MUON_REJECTED_IMPLEMENTATION"
    assert state["phase"] == "OPTIMIZER_FREEZE"


def test_completed_training_result_prevents_duplicate_launch(monkeypatch, tmp_path):
    run = tmp_path / "run"; run.mkdir(); (run / "result.json").write_text(json.dumps({"status": "TRAINING_COMPLETE", "tokens": 10}))
    monkeypatch.setattr(master, "RUN", run)
    monkeypatch.setattr(master, "run_child", lambda *a, **k: pytest.fail("duplicate launch attempted"))
    monkeypatch.setattr(master, "complete", lambda state, phase, **fields: state.update(fields))
    state = {}
    master.phase_train(state)
    assert state["training"]["tokens"] == 10


def test_terminal_artifact_generation(monkeypatch, tmp_path):
    art = tmp_path / "artifacts"; data = art / "data"; data.mkdir(parents=True)
    manifest = data / "manifest.json"; manifest.write_text("{}\n")
    certificate = data / "certification.json"; certificate.write_text('{"acceptance":"PASS"}\n')
    tokenizer = art / "tokenizer.json"; tokenizer.write_text("{}\n")
    audit = art / "audit.json"
    audit.write_text(json.dumps({"dataset": {"unique_training_tokens": 2_259_629_459,
                                               "source_shares": {"fineweb_edu": .45}},
                                 "experiments": [{"id": "integrated-100m"}]}) + "\n")
    recipe = art / "final_production_recipe.json"; recipe.write_text("{}\n")
    (art / "final_training_curve.csv").write_text("tokens,loss\n1,2\n")
    events = art / "final_training_events.jsonl"; events.write_text('{"event":"x"}\n')
    monkeypatch.setattr(master, "ART", art); monkeypatch.setattr(master, "DATA", data)
    monkeypatch.setattr(master, "MANIFEST", manifest); monkeypatch.setattr(master, "TOKENIZER", tokenizer)
    monkeypatch.setattr(master, "AUDIT", audit); monkeypatch.setattr(master, "EVENTS", events)
    monkeypatch.setattr(master, "STATE", art / "state.json"); monkeypatch.setattr(master, "PREVIOUS", art / "previous.json")
    metric = {"checkpoint": "final.pt", "checkpoint_sha256": "abc", "tokens": 100,
              "validation_loss": 2.0, "source_validation": {}, "wikitext_ppl": 10.0, "wikitext_bpb": 1.0,
              "hellaswag": .3, "arc_easy": .3, "piqa": .55, "winogrande": .51}
    state = {"schema": "final-production-master-state-v1", "phase": "SUBMISSION_EVIDENCE_PACKAGE", "completed": [],
             "triton": {"status": "TRITON_REJECTED"}, "muon_audit": {"status": "PASS"},
             "muon": {"status": "MUON_REJECTED"}, "lr": {"selected_lr": 5e-4},
             "recipe": {"architecture": {}, "wsd": {"warmup": .02, "stable": .83, "cosine_decay": .15}},
             "parameter_count": 48_636_168, "optimizer": "adamw", "selected_lr": 5e-4,
             "final_token_budget": 1_800_000_000,
             "training": {"tokens": 1_800_000_000, "training_seconds": 1, "effective_tokens_per_second": 3000,
                          "restart_count": 1},
             "checkpoint_comparison": {"checkpoints": {"final": metric}, "pareto_challengers": []},
             "persistence": {"uploads": {"final": {"verified": True}}}}
    master.phase_package(state)
    for name in ("final_submission_results.json", "final_submission_results.csv", "final_model_facts.json",
                 "final_experiment_decisions.json", "final_production_report.md", "demo_metrics.json", "demo_timeline.json"):
        assert (art / name).stat().st_size > 0
