"""Pre-register the matched replications without changing historical state."""
from __future__ import annotations
import argparse
import copy
import json
import math
import subprocess
from pathlib import Path
import torch
from latticelm.config import LatticeConfig
from latticelm.final_recipe import make_optimizer, wsd_lr_scale
from followup_common import ROOT, RESEARCH, atomic_json, init_state, object_hash, register, sha256, utc
from followup_train import build_model

SOURCES = ["scripts/followup_train.py", "src/latticelm/model.py", "src/latticelm/config.py",
           "src/latticelm/final_recipe.py", "src/latticelm/data_d.py", "src/latticelm/posthack_variants.py",
           "src/latticelm/co4_reference.py", "src/latticelm/research_sampling.py", "scripts/train_final_production.py"]


def make_record(name, config, budget, rate, control, variant="baseline", **extra):
    cfg = LatticeConfig(**config)
    model = build_model({"config": config, "variant": variant, **extra})
    _, partition = make_optimizer(model, cfg.optimizer, cfg.learning_rate, cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
    manifest = "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
    tokenizer = "artifacts/tokenizers/final_corpus_4k.json"
    top = json.loads((ROOT / manifest).read_text())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    step_tokens = cfg.batch_size * cfg.context_length; total_steps = math.ceil(budget / step_tokens)
    record = {"schema": "followup-run-registration-v1", "experiment_id": name, "registered_at": utc(),
              "config": json.loads(json.dumps(cfg.to_dict())), "config_sha256": object_hash(config),
              "parameter_count": params, "active_parameter_count": params, "optimizer_partition": partition,
              "token_budget": budget, "planning_tps": rate, "parent_control": control, "variant": variant,
              "manifest_path": manifest, "sampling_manifest_sha256": sha256(ROOT / manifest),
              "tokenizer_path": tokenizer, "tokenizer_sha256": sha256(ROOT / tokenizer),
              "source_order": list(top["mixture_definition"]), "threads": 16,
              "source_hashes": {rel: sha256(ROOT / rel) for rel in SOURCES},
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "checkpoint_tokens": 4_000_000 if params > 24_000_000 else 7_500_000 if params > 12_000_000 else 10_000_000,
              "max_checkpoint_seconds": 1100, "milestone_tokens": [t for t in [10_000_000, 50_000_000, 100_000_000, 150_000_000, 200_000_000, 300_000_000] if t <= budget],
              "schedule_total_steps": total_steps, "schedule": {"family": "wsd", "peak_lr": cfg.learning_rate,
              "warmup_fraction": cfg.warmup_fraction, "stable_fraction": cfg.stable_fraction, "decay_fraction": cfg.decay_fraction,
              "warmup_steps": max(1, round(total_steps * cfg.warmup_fraction)), "decay_steps": max(1, round(total_steps * cfg.decay_fraction)),
              "sample_lrs": {str(step): cfg.learning_rate*wsd_lr_scale(step,total_steps,cfg.warmup_fraction,cfg.stable_fraction,cfg.decay_fraction) for step in [1, total_steps//2, total_steps]}},
              "batch_semantics": "8192 processed tokens per optimizer step; final CE is masked to the remaining logical tokens; no accumulation",
              "precision": "float32", "evaluation_tasks": "hellaswag,arc_easy,piqa,winogrande", "evaluation_seconds": 1800,
              "ready": True, "launch_certificate": f"certificates/{name}.json", "disk_floor_bytes": 5_000_000_000,
              "promotion_criterion": "replicated, coherent LM/BPB gain with acceptable wall cost and no severe full-task regression",
              "abort_criterion": "nonfinite loss/gradient, source/data identity mismatch, code failure, or hard scientific stop",
              **extra}
    (RESEARCH / "configs").mkdir(exist_ok=True)
    atomic_json(RESEARCH / "configs" / (name + ".json"), record["config"])
    return record


def prepare():
    state = init_state("2026-10-17T20:10:00+00:00")
    old = ROOT / "post_hackathon_research"
    rows = {r["experiment_id"]: r for r in map(json.loads,(old / "RUN_REGISTRY.jsonl").read_text().splitlines())}
    audit = json.loads((RESEARCH / "R32_CO4_PROVENANCE_AUDIT.json").read_text())
    if not audit["passed"]: raise RuntimeError("historical control provenance is unverified")
    dense = copy.deepcopy(rows["R32-CO4-CONTROL"]["config"]); dense["architecture"] = "lattice"
    record = make_record("F32-DENSE-S1337-300M", dense, 300_000_000, 4108, "R32-CO4-CONTROL",
                         initial_batch_sha256=audit["initial_batch_sha256"],
                         hypothesis="Fresh 300M dense trajectory closes MOD-Co4 vs dense at matched seed/data/optimizer/schedule",
                         initialization_policy="fresh random initialization; empty optimizers; sampler cursor zero; no warm-start")
    assert record["parameter_count"] == rows["A32-DENSE-S1337"]["parameter_count"] == 31_986_396
    register(record)
    d4 = copy.deepcopy(rows["D4-16M-ANNEAL"])
    cfg = d4["config"]; cfg["seed"] = 2026
    extra = {k:d4[k] for k in ("sampling_mode","sampling_rule","research_d_index_path","research_d_index_sha256","anneal_rule")}
    register(make_record("F16-D4-S2026-150M", cfg, 150_000_000, 6942, "R16-S2026-CONTROL",
                         hypothesis="D4 BPB and source-wise direction replicates at paired seed 2026", **extra))
    cfg = copy.deepcopy(rows["A8-CO4-S1337"]["config"]); cfg["seed"] = 7331
    register(make_record("F8-MOD-S7331-100M", cfg, 100_000_000, 11330, "F8-DIFF-S7331-100M",
                         hypothesis="Fresh paired MOD control for Differential replication"))
    register(make_record("F8-DIFF-S7331-100M", cfg, 100_000_000, 10035, "F8-MOD-S7331-100M", "differential_co4",
                         hypothesis="Differential operator replicates under one matched fresh seed"))
    atomic_json(RESEARCH / "DEADLINE_CONFIRMATION.json", {"at":utc(),"hard_deadline_utc":state["hard_deadline_utc"],
                "scientific_stop_utc":utc(state["scientific_stop_epoch"]),"confirmed_by_user":"October 17, 2026 at 16:10 EDT",
                "earlier_blocker":"resolved by explicit corrected date"})
    return record


if __name__ == "__main__":
    prepare(); print("Registered fresh dense 300M, D4 150M, and one matched Differential 100M pair.")
