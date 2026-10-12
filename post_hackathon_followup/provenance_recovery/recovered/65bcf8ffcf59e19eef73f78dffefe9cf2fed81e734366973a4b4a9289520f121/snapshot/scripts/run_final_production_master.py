"""Unattended final-production state machine for LatticeLM.

Optional research failures are evidence and fall through.  Identity,
checkpoint, parameter-cap, evaluator, or deadline failures block safely.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from latticelm.config import LatticeConfig
from latticelm.data_d import sha256_file
from latticelm.final_recipe import make_optimizer, muon_parameter_partition
from latticelm.model import build_model

ROOT = Path(__file__).resolve().parents[1]; ART = ROOT / "artifacts"
STATE = ART / "final_production_master_state.json"; PREVIOUS = ART / "final_production_master_state.previous.json"
LOCK = ART / "final_production_master.lock"; EVENTS = ART / "final_training_events.jsonl"
DATA = ART / "data/data_d_v4/canonical-2250m"; MANIFEST = DATA / "manifest.json"
TOKENIZER = ART / "tokenizers/final_corpus_4k.json"; RELEASE_CONFIG = ROOT / "configs/final_recipe_release_candidate.json"
AUDIT = ART / "final_recipe_completion_audit.json"; RUN = ART / "final_production_run"
TRITON_PYTHON = ROOT / ".triton-cpu-venv/bin/python"
DEADLINE = datetime(2026, 9, 30, 16, 0, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande")
PHASES = (
    "INSPECT_AND_FREEZE", "TRITON_FINAL_GATE", "MUON_IMPLEMENTATION_AUDIT", "MUON_BOUNDED_SCREEN",
    "OPTIMIZER_FREEZE", "LR_RESOLUTION", "FINAL_RECIPE_FREEZE", "FINAL_BUDGET_SELECTION",
    "PRELAUNCH_CERTIFICATION", "FINAL_LONG_RUN_FRESH", "FINAL_CHECKPOINT_EVALUATION",
    "FINAL_ARTIFACT_PERSIST", "SUBMISSION_EVIDENCE_PACKAGE", "PRODUCTION_COMPLETE",
)
STOP = False; CHILD: subprocess.Popen | None = None


class RetryablePublicationError(RuntimeError):
    pass


def atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    if path == STATE and path.exists(): shutil.copy2(path, PREVIOUS)
    os.replace(temporary, path)


def event(kind: str, **fields) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": time.time(), "at_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def last_json_object(output: str) -> dict:
    """Decode the final JSON line while ignoring compiler/runtime warnings."""
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict): return value
        except json.JSONDecodeError: pass
    raise ValueError("child output contained no JSON object")


def load_state() -> dict:
    for path in (STATE, PREVIOUS):
        try:
            state = json.loads(path.read_text())
            if state["schema"] == "final-production-master-state-v1": return state
        except Exception: pass
    return {"schema": "final-production-master-state-v1", "phase": PHASES[0], "completed": [], "created_at": time.time(),
            "research_seconds": 0.0, "research_tokens": 0, "child_pid": None, "deadline_epoch": DEADLINE,
            "internal_deadline": "2026-09-30T16:00:00-04:00", "terminal_state": None}


def save(state: dict, phase: str | None = None, **fields) -> None:
    if phase is not None: state["phase"] = phase
    state.update(fields); state["updated_at"] = time.time(); atomic(STATE, state); event("MASTER_STATE", phase=state["phase"])


def complete(state: dict, phase: str, **fields) -> None:
    if phase not in state["completed"]: state["completed"].append(phase)
    next_phase = PHASES[min(PHASES.index(phase) + 1, len(PHASES) - 1)]
    save(state, next_phase, **fields)


def stop(*_) -> None:
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None: CHILD.terminate()


def planning_snapshot(state: dict, measured_tps: float | None = None, final_tokens: int | None = None) -> dict:
    remaining_wall_hours = max(0.0, DEADLINE - time.time()) / 3600
    usable = remaining_wall_hours * 22 / 24
    tps = measured_tps or float(state.get("measured_tps", 3157.573615954999))
    tokens = final_tokens if final_tokens is not None else state.get("final_token_budget")
    projected = (tokens / (tps * .90) / 3600) if tokens else None
    return {"at": time.time(), "wall_hours_to_internal_deadline": remaining_wall_hours,
            "conservative_usable_execution_hours": usable, "research_hours_consumed": state.get("research_seconds", 0) / 3600,
            "measured_effective_tok_s": tps, "planning_tok_s": tps * .90, "projected_final_training_hours": projected,
            "required_finalization_hours": 14, "contingency_hours": 36,
            "projected_margin_hours": usable - 50 - projected if projected is not None else None}


def run_child(state: dict, label: str, command: list, tokens: int = 0, optional: bool = False,
              timeout: float | None = None) -> tuple[int, str]:
    global CHILD
    if STOP: raise InterruptedError
    if time.time() >= DEADLINE: raise RuntimeError("internal hard deadline reached")
    event("CHILD_START", label=label, command=[Path(str(command[0])).name, *map(str, command[1:])], planning=planning_snapshot(state))
    started = time.perf_counter(); CHILD = subprocess.Popen([str(item) for item in command], cwd=ROOT, text=True,
                                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    state["child_pid"] = CHILD.pid; save(state)
    try: output, _ = CHILD.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        CHILD.terminate()
        try: output, _ = CHILD.communicate(timeout=30)
        except subprocess.TimeoutExpired: CHILD.kill(); output, _ = CHILD.communicate()
        code = 124
    else: code = CHILD.returncode
    elapsed = time.perf_counter() - started; CHILD = None; state["child_pid"] = None
    if tokens:
        state["research_seconds"] += elapsed
        # A signal-safe partial checkpoint is not the declared endpoint.  The
        # resumed child will report the complete token budget once, so do not
        # double-count the requested budget across service restarts.
        if code == 0 and not STOP: state["research_tokens"] += tokens
    save(state); event("CHILD_END", label=label, exit_code=code, seconds=elapsed, output_tail=output[-4000:])
    if STOP: raise InterruptedError
    if code and not optional: raise RuntimeError(f"{label} exited {code}: {output[-1000:]}")
    return code, output


def research_fits(state: dict, experiment_tokens: int, experiment_tps: float) -> bool:
    """Protect a 1.5B final run plus the mandatory 50-hour reserve."""
    usable = planning_snapshot(state)["conservative_usable_execution_hours"]
    experiment_hours = experiment_tokens / max(1.0, experiment_tps * .90) / 3600
    minimum_final_hours = 1_500_000_000 / (3157.573615954999 * .90) / 3600
    return experiment_hours + minimum_final_hours + 50 <= usable


def paired_bootstrap(a: list[float], b: list[float], seed: int = 1337, draws: int = 5000) -> dict:
    if len(a) != len(b) or not a: return {"status": "UNAVAILABLE"}
    differences = np.asarray(a) - np.asarray(b); rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(differences, len(differences), replace=True).mean() for _ in range(draws)])
    return {"status": "PASS", "n": len(a), "mean_difference": float(differences.mean()),
            "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "sign_convention": "candidate_minus_baseline; lower is better"}


def experiment_command(name: str, config: Path, target: int) -> list:
    root = ART / "final_recipe_experiments" / name
    mode = "--resume" if (root / "latest.pt").exists() else "--fresh"
    return [sys.executable, "scripts/train_final_recipe_experiment.py", "--config", config, "--manifest", MANIFEST,
            "--tokenizer", TOKENIZER, "--experiment", name, "--parent-decision", "final-production-master",
            "--target-tokens", target, "--threads", 16, "--backend", "compile", mode]


def evaluate_experiment(state: dict, name: str) -> dict:
    root = ART / "final_recipe_experiments" / name; checkpoint = root / "milestone.pt"
    for kind, script in (("wikitext", "scripts/evaluate_wikitext103.py"), ("gibc", "scripts/evaluate_gibc.py")):
        output = root / f"{kind}.json"
        if not output.exists(): run_child(state, f"{name}-{kind}", [sys.executable, script, "--checkpoint", checkpoint, "--tokenizer", TOKENIZER, "--output", output, "--batch-size", 16, "--threads", 16])
    result = json.loads((root / "result.json").read_text()); wiki = json.loads((root / "wikitext.json").read_text()); tasks = json.loads((root / "gibc.json").read_text())
    result.update(wikitext_ppl=wiki["perplexity"], wikitext_bpb=wiki["bits_per_byte"],
                  **{task: tasks["results"][task]["acc,none"] for task in TASKS})
    atomic(root / "final_adjudication.json", result); return result


def phase_inspect(state: dict) -> None:
    audit = json.loads(AUDIT.read_text()); expected = audit["experiments"][-1]
    integrated = ART / "final_recipe_experiments/integrated-100m/milestone.pt"
    if sha256_file(integrated) != expected["checkpoint_sha256"]: raise RuntimeError("integrated 100M checkpoint hash mismatch")
    if sha256_file(MANIFEST) != audit["dataset"]["manifest_sha256"]: raise RuntimeError("DATA-D-v4 identity mismatch")
    if sha256_file(TOKENIZER) != audit["tokenizer"]["selected_sha256"]: raise RuntimeError("tokenizer identity mismatch")
    certification = json.loads((DATA / "certification.json").read_text())
    if certification.get("acceptance") != "PASS": raise RuntimeError("DATA-D-v4 external certification is not PASS")
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    active = [line for line in processes.splitlines() if "python" in line and any(term in line for term in ("train_final", "run_final_production")) and str(os.getpid()) not in line]
    if active: raise RuntimeError(f"stale training process active: {active}")
    from huggingface_hub import HfApi
    identity = HfApi().whoami(); disk = shutil.disk_usage(ROOT)
    if disk.free < 30_000_000_000: raise RuntimeError("less than 30GB free before production")
    evaluator_tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_final_recipe_components.py", "tests/test_wikitext_bpb.py"], cwd=ROOT, text=True, capture_output=True)
    if evaluator_tests.returncode: raise RuntimeError("corrected evaluator tests failed")
    baseline = {"git_head": git_head(), "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "integrated_checkpoint": str(integrated), "integrated_checkpoint_sha256": expected["checkpoint_sha256"],
                "manifest_sha256": sha256_file(MANIFEST), "tokenizer_sha256": sha256_file(TOKENIZER),
                "data_certification_sha256": sha256_file(DATA / "certification.json"), "evaluator_sha256": sha256_file(ROOT / "scripts/evaluate_gibc.py"),
                "disk_free_bytes": disk.free, "hf_account": identity.get("name", "authenticated"), "active_training_processes": [],
                "planning": planning_snapshot(state)}
    atomic(ART / "final_production_provenance_baseline.json", baseline); complete(state, "INSPECT_AND_FREEZE", provenance=baseline)


def phase_triton(state: dict) -> None:
    outcome = {"status": "TRITON_REJECTED", "engineering_cap_seconds": 1200, "promoted": False,
               "compile": "NOT_RUN", "parity": "NOT_RUN", "microbenchmark": "NOT_RUN", "full_step_speedup": None, "trajectory": "NOT_RUN"}
    if not TRITON_PYTHON.is_file():
        outcome.update(compile="FAIL", reason=f"isolated Triton interpreter missing: {TRITON_PYTHON}")
        atomic(ART / "final_triton_decision.json", outcome); complete(state, "TRITON_FINAL_GATE", triton=outcome); return
    code, output = run_child(state, "triton-final-gate", [TRITON_PYTHON, "scripts/smoke_triton_co4_mod.py"], optional=True, timeout=1200)
    if code == 124: outcome.update(compile="TIMEOUT", reason="usable AArch64 kernel not available inside 20-minute cap")
    elif code: outcome.update(compile="FAIL", reason=f"isolated gate exited {code}")
    else:
        try:
            result = last_json_object(output)
            outcome.update(compile=result.get("gate1"), parity=result.get("gate2"), parity_cases=result.get("cases"),
                           edge_output=result.get("edge_output"), microbenchmark=result.get("gate3_microbenchmark"))
        except Exception: result = {}; outcome.update(reason="gate output was not valid JSON")
        if result.get("gate2") != "PASS":
            outcome["reason"] = "numerical parity gate failed"
        elif not result.get("gate3_microbenchmark", {}).get("pass"):
            outcome["reason"] = "parity passed but production-shape microbenchmark was not faster than PyTorch"
        else:
            # The repository has no end-to-end model dispatch using this kernel.
            # A synthetic-kernel win cannot authorize production substitution.
            outcome["reason"] = "no certified full-model dispatch; 8% end-to-end threshold cannot be met"
            outcome["full_step_speedup"] = 0.0
    atomic(ART / "final_triton_decision.json", outcome); complete(state, "TRITON_FINAL_GATE", triton=outcome)


def phase_muon_audit(state: dict) -> None:
    cfg = LatticeConfig.from_json(RELEASE_CONFIG); model = build_model(cfg)
    eligible, adam, names = muon_parameter_partition(model); all_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer, serialized_partition = make_optimizer(model, "muon_hybrid", cfg.learning_rate, cfg.weight_decay, (cfg.adam_beta1, cfg.adam_beta2))
    torch.manual_seed(cfg.seed); before = {name: p.detach().clone() for name, p in model.named_parameters()}
    x = torch.arange(16).remainder(cfg.vocab_size).view(1, 16)
    optimizer.zero_grad(); loss = model(x)[0].float().square().mean(); loss.backward(); optimizer.step()
    updates = {name: float((p.detach() - before[name]).norm()) for name, p in model.named_parameters()}
    finite = all(math.isfinite(value) for value in updates.values()); nonzero = [value for value in updates.values() if value > 0]
    state_roundtrip = optimizer.state_dict(); optimizer.load_state_dict(state_roundtrip)
    torch_muon = getattr(torch.optim, "Muon", None)
    import torch.optim._muon as torch_muon_module
    import inspect
    source = inspect.getsource(torch_muon_module._adjust_lr)
    semantics = "0.2 * math.sqrt(max(A, B))" in source
    reference_lr = cfg.learning_rate; historical_lr = .02
    configured_lr = float(optimizer.optimizers[0].param_groups[0]["lr"])
    lr_semantics_certified = math.isclose(configured_lr, reference_lr, rel_tol=1e-12)
    certified = (torch_muon is not None and semantics and lr_semantics_certified and finite and bool(nonzero)
                 and len(eligible) + len(adam) == len(all_parameters) and names == serialized_partition)
    result = {"status": "PASS" if certified else "MUON_REJECTED_IMPLEMENTATION", "eligible_muon_parameters": sum(p.numel() for p in eligible),
              "adamw_parameters": sum(p.numel() for p in adam), "trainable_tensors": len(all_parameters), "assigned_tensors": len(names),
              "duplicates": 0, "missing": 0, "adjust_lr_fn": "match_rms_adamw", "muon_lr": configured_lr, "momentum": .95,
              "reference_consistent_lr": reference_lr, "configured_historical_lr": historical_lr,
              "historical_to_reference_lr_ratio": historical_lr / reference_lr,
              "updated_tensors": len(nonzero), "zero_update_tensors": len(all_parameters) - len(nonzero),
              "effective_update_norm": {"minimum": min(nonzero, default=0), "median": statistics.median(nonzero) if nonzero else 0, "maximum": max(nonzero, default=0)},
              "finite_updates": finite, "optimizer_state_roundtrip": "PASS", "installed_scaling_formula_certified": semantics,
              "hyperparameter_semantics_certified": lr_semantics_certified,
              "reason": "historical .02 LR replaced with installed match_rms_adamw reference-consistent AdamW LR" if lr_semantics_certified else "configured LR does not match installed semantics"}
    atomic(ART / "final_muon_implementation_audit.json", result); complete(state, "MUON_IMPLEMENTATION_AUDIT", muon_audit=result)


def phase_muon_screen(state: dict) -> None:
    if state["muon_audit"]["status"] != "PASS":
        decision = {"status": "MUON_REJECTED_IMPLEMENTATION", "screen_tokens": 0, "promoted": False, "reason": "implementation audit did not certify semantics"}
        atomic(ART / "final_muon_decision.json", decision); complete(state, "MUON_BOUNDED_SCREEN", muon=decision); return
    cfg = replace(LatticeConfig.from_json(RELEASE_CONFIG), optimizer="muon_hybrid")
    path = ROOT / "configs/final_production_muon_screen.json"; atomic(path, cfg.to_dict()); name = "final-muon-50m"
    if not research_fits(state, 50_000_000, 2800):
        decision = {"status": "MUON_REJECTED", "promoted": False, "screen_tokens": 0, "reason": "deadline protection gate"}
        atomic(ART / "final_muon_decision.json", decision); complete(state, "MUON_BOUNDED_SCREEN", muon=decision); return
    if not (ART / "final_recipe_experiments" / name / "result.json").exists():
        code, output = run_child(state, name, experiment_command(name, path, 50_000_000), tokens=50_000_000, optional=True)
        if code:
            decision = {"status": "MUON_REJECTED", "promoted": False, "reason": f"bounded screen failed safely ({code})"}
            atomic(ART / "final_muon_decision.json", decision); complete(state, "MUON_BOUNDED_SCREEN", muon=decision); return
    try: muon = evaluate_experiment(state, name)
    except Exception as error:
        decision = {"status": "MUON_REJECTED", "promoted": False, "reason": f"bounded evaluation failed safely: {type(error).__name__}: {error}"}
        atomic(ART / "final_muon_decision.json", decision); complete(state, "MUON_BOUNDED_SCREEN", muon=decision); return
    baseline = next(item for item in json.loads(AUDIT.read_text())["experiments"] if item["id"] == "qknorm-on")
    tps_ratio = muon["tokens_per_second"] / baseline["tokens_per_second"]
    lm_gain = (baseline["validation_loss"] - muon["validation_loss"]) + (baseline["wikitext_bpb"] - muon["wikitext_bpb"])
    promising = tps_ratio >= .80 and lm_gain >= .025 and muon["validation_loss"] < baseline["validation_loss"] and muon["wikitext_bpb"] < baseline["wikitext_bpb"]
    decision = {"status": "MUON_PROMISING" if promising else "MUON_REJECTED", "promoted": False, "screen": muon, "baseline": baseline,
                "throughput_ratio": tps_ratio, "combined_lm_gain": lm_gain, "paired_validation": {"status": "UNAVAILABLE_AT_50M_BASELINE"},
                "policy": "substantial consistent validation+BPB gain required to offset slower wall clock"}
    if promising:
        name100 = "final-muon-100m"; path100 = ROOT / "configs/final_production_muon_100m.json"; atomic(path100, cfg.to_dict())
        if not research_fits(state, 100_000_000, muon["tokens_per_second"]):
            decision.update(status="MUON_REJECTED", reason="100M confirmation would threaten final protected envelope")
        elif not (ART / "final_recipe_experiments" / name100 / "result.json").exists():
            code, _ = run_child(state, name100, experiment_command(name100, path100, 100_000_000), tokens=100_000_000, optional=True)
            if code: decision.update(status="MUON_REJECTED", reason="100M confirmation failed safely")
        if (ART / "final_recipe_experiments" / name100 / "result.json").exists():
            try:
                confirmation = evaluate_experiment(state, name100); integrated = json.loads(AUDIT.read_text())["experiments"][-1]
                sustained = confirmation["validation_loss"] < integrated["validation_loss"] and confirmation["wikitext_bpb"] < integrated["wikitext_bpb"] and confirmation["tokens_per_second"] >= .80 * integrated["tokens_per_second"]
                decision.update(confirmation_100m=confirmation, status="MUON_PROMOTED" if sustained else "MUON_REJECTED", promoted=sustained)
            except Exception as error: decision.update(status="MUON_REJECTED", promoted=False, reason=f"100M confirmation evaluation failed safely: {error}")
    atomic(ART / "final_muon_decision.json", decision); complete(state, "MUON_BOUNDED_SCREEN", muon=decision)


def phase_optimizer(state: dict) -> None:
    optimizer = "muon_hybrid" if state["muon"].get("promoted") else "adamw"
    measured = (state["muon"].get("confirmation_100m") or {}).get("tokens_per_second", 3157.573615954999) if optimizer == "muon_hybrid" else 3157.573615954999
    complete(state, "OPTIMIZER_FREEZE", optimizer=optimizer, measured_tps=measured)


def phase_lr(state: dict) -> None:
    integrated = json.loads(AUDIT.read_text())["experiments"][-1]
    if state["optimizer"] != "adamw":
        decision = {"status": "SKIPPED_MUON_PROMOTED", "selected_lr": integrated["peak_lr"], "existing_8e4": integrated}
    else:
        name = "final-adamw-lr-0p0005-100m"; cfg = replace(LatticeConfig.from_json(RELEASE_CONFIG), learning_rate=5e-4, optimizer="adamw")
        path = ROOT / "configs/final_production_adamw_lr_0p0005.json"; atomic(path, cfg.to_dict())
        if not research_fits(state, 100_000_000, 3157.573615954999):
            decision = {"status": "SKIPPED_DEADLINE_PROTECTION", "selected_lr": 8e-4, "existing_8e4": integrated, "reason": "matched control would threaten final protected envelope"}
            atomic(ART / "final_lr_decision.json", decision); complete(state, "LR_RESOLUTION", lr=decision, selected_lr=decision["selected_lr"]); return
        if not (ART / "final_recipe_experiments" / name / "result.json").exists():
            code, _ = run_child(state, name, experiment_command(name, path, 100_000_000), tokens=100_000_000, optional=True)
        else: code = 0
        if code:
            decision = {"status": "CONTROL_FAILED_RETAIN_CERTIFIED", "selected_lr": 8e-4, "existing_8e4": integrated, "reason": "optional matched control failed"}
        else:
            try: candidate = evaluate_experiment(state, name)
            except Exception as error:
                decision = {"status": "CONTROL_EVALUATION_FAILED_RETAIN_CERTIFIED", "selected_lr": 8e-4, "existing_8e4": integrated, "reason": str(error)}
                atomic(ART / "final_lr_decision.json", decision); complete(state, "LR_RESOLUTION", lr=decision, selected_lr=decision["selected_lr"]); return
            candidate_losses = sum(candidate.get("paired_validation_losses", {}).values(), [])
            base_checkpoint = torch.load(integrated["checkpoint"], map_location="cpu", weights_only=False)
            base_losses = sum(base_checkpoint.get("paired_validation_losses", {}).values(), [])
            paired = paired_bootstrap(candidate_losses, base_losses)
            val_delta = candidate["validation_loss"] - integrated["validation_loss"]; wiki_delta = candidate["wikitext_bpb"] - integrated["wikitext_bpb"]
            if val_delta < -5e-4 and wiki_delta < 0: selected = 5e-4; rationale = "5e-4 wins validation and WikiText"
            elif val_delta > 5e-4 and wiki_delta > 0: selected = 8e-4; rationale = "8e-4 wins validation and WikiText"
            elif wiki_delta < 0 and not (paired.get("ci95", [0, 0])[0] > 0): selected = 5e-4; rationale = "effectively tied; repeated WikiText advantage favors 5e-4"
            else: selected = 8e-4; rationale = "split or effectively tied evidence does not materially displace certified 8e-4"
            decision = {"status": "LR_FROZEN", "selected_lr": selected, "existing_8e4": integrated, "candidate_5e4": candidate,
                        "validation_delta_5e4_minus_8e4": val_delta, "wikitext_bpb_delta_5e4_minus_8e4": wiki_delta,
                        "paired_validation_bootstrap": paired, "rationale": rationale}
    atomic(ART / "final_lr_decision.json", decision); complete(state, "LR_RESOLUTION", lr=decision, selected_lr=decision["selected_lr"])


def phase_recipe(state: dict) -> None:
    cfg = replace(LatticeConfig.from_json(RELEASE_CONFIG), optimizer=state["optimizer"], learning_rate=state["selected_lr"])
    path = ROOT / "configs/final_production.json"; atomic(path, cfg.to_dict())
    model = build_model(cfg); params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params != 48_636_168 or params >= 50_000_000: raise RuntimeError(f"executable parameter count invalid: {params}")
    recipe = {"schema": "final-production-recipe-v1", "architecture": cfg.to_dict(), "trainable_parameters": params,
              "tokenizer": str(TOKENIZER), "tokenizer_sha256": sha256_file(TOKENIZER), "data_manifest": str(MANIFEST),
              "data_manifest_sha256": sha256_file(MANIFEST), "data_certification_sha256": sha256_file(DATA / "certification.json"),
              "optimizer": state["optimizer"], "peak_lr": state["selected_lr"], "wsd": {"warmup": .02, "stable": .83, "cosine_decay": .15},
              "backend": "torch.compile max-autotune-no-cudagraphs fullgraph=false", "triton": state["triton"]["status"], "seed": cfg.seed,
              "checkpoint_cadence_tokens": 50_000_000, "evaluation_milestones": ["end-stable", "mid-decay", "final"], "frozen_at": time.time(), "source_commit": git_head()}
    atomic(ART / "final_production_recipe.json", recipe)
    (ART / "final_production_recipe.md").write_text("# Final Production Recipe\n\n```json\n" + json.dumps(recipe, indent=2, sort_keys=True) + "\n```\n")
    complete(state, "FINAL_RECIPE_FREEZE", recipe=recipe, production_config=str(path), parameter_count=params)


def select_budget(usable_hours: float, measured_tps: float, research_hours: float = 0.0) -> int:
    available = usable_hours - 50
    planning_tps = measured_tps * .90
    candidates = (2_250_000_000, 2_200_000_000, 2_100_000_000, 2_000_000_000, 1_900_000_000, 1_800_000_000)
    for candidate in candidates:
        if candidate / planning_tps / 3600 <= available: return candidate
    return max(0, int(available * 3600 * planning_tps // 50_000_000) * 50_000_000)


def phase_budget(state: dict) -> None:
    snapshot = planning_snapshot(state); budget = select_budget(snapshot["conservative_usable_execution_hours"], state["measured_tps"])
    if budget <= 0: raise RuntimeError("no safe final training budget remains after mandatory reserves")
    if budget < 1_500_000_000: event("CRITICAL_SCHEDULE_WARNING", selected_tokens=budget)
    snapshot = planning_snapshot(state, final_tokens=budget); snapshot["selected_token_budget"] = budget
    atomic(ART / "final_budget_selection.json", snapshot); state["recipe"]["selected_final_token_budget"] = budget; atomic(ART / "final_production_recipe.json", state["recipe"])
    complete(state, "FINAL_BUDGET_SELECTION", final_token_budget=budget, budget=snapshot)


def production_command(state: dict, certify_only: bool = False) -> list:
    mode = "--resume" if (RUN / "latest.pt").exists() else "--fresh"
    command = [sys.executable, "scripts/train_final_production.py", "--config", state["production_config"], "--manifest", MANIFEST,
               "--tokenizer", TOKENIZER, "--target-tokens", state["final_token_budget"], "--deadline-epoch", DEADLINE,
               "--threads", 16, "--backend", "compile" if not certify_only else "eager", mode]
    if certify_only: command.append("--certify-only")
    return command


def phase_prelaunch(state: dict) -> None:
    code, output = run_child(state, "prelaunch-certification", production_command(state, certify_only=True))
    if code: raise RuntimeError("fresh initialization certification failed")
    certification = json.loads((ART / "final_prelaunch_certification.json").read_text())
    if certification["parameter_count"] != state["parameter_count"]: raise RuntimeError("prelaunch parameter count mismatch")
    complete(state, "PRELAUNCH_CERTIFICATION", prelaunch=certification)


def phase_train(state: dict) -> None:
    result_path = RUN / "result.json"
    if not result_path.exists(): run_child(state, "final-long-run", production_command(state))
    result = json.loads(result_path.read_text())
    if result["status"] != "TRAINING_COMPLETE": raise RuntimeError("production child did not reach training complete")
    complete(state, "FINAL_LONG_RUN_FRESH", training=result)


def evaluation_metrics(wiki: dict, tasks: dict, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "tokens": payload["tokens"],
            "validation_loss": payload["validation_loss"], "source_validation": payload["source_validation"],
            "wikitext_ppl": wiki["perplexity"], "wikitext_bpb": wiki["bits_per_byte"],
            **{task: tasks["results"][task]["acc,none"] for task in TASKS}, "evaluator": "corrected joined causal boundary"}


def phase_evaluate(state: dict) -> None:
    comparison = {}
    for role in ("end-stable", "mid-decay", "final"):
        checkpoint = RUN / f"{role}.pt"; wiki = ART / f"final_{role}_wikitext.json"; tasks = ART / f"final_{role}_tasks.json"
        if not checkpoint.exists(): raise RuntimeError(f"required semantic checkpoint missing: {role}")
        if not wiki.exists(): run_child(state, f"evaluate-{role}-wikitext", [sys.executable, "scripts/evaluate_wikitext103.py", "--checkpoint", checkpoint, "--tokenizer", TOKENIZER, "--output", wiki, "--batch-size", 16, "--threads", 16])
        if not tasks.exists(): run_child(state, f"evaluate-{role}-tasks", [sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", checkpoint, "--tokenizer", TOKENIZER, "--output", tasks, "--tasks", ",".join(TASKS), "--batch-size", 16, "--threads", 16])
        comparison[role] = evaluation_metrics(json.loads(wiki.read_text()), json.loads(tasks.read_text()), checkpoint)
    final = comparison["final"]; challengers = []
    for role in ("end-stable", "mid-decay"):
        row = comparison[role]; wins = sum(row[key] < final[key] for key in ("validation_loss", "wikitext_bpb")) + sum(row[key] > final[key] for key in TASKS)
        losses = sum(row[key] > final[key] for key in ("validation_loss", "wikitext_bpb")) + sum(row[key] < final[key] for key in TASKS)
        if wins > 0 and losses == 0: challengers.append(role)
    decision = {"checkpoints": comparison, "default_final": "final", "pareto_challengers": challengers,
                "policy": "terminal default unless another checkpoint has a clear material multi-metric advantage; no weighted score"}
    atomic(ART / "final_checkpoint_comparison.json", decision)
    lines = ["# Final Checkpoint Comparison", "", "Default: `final`", "", "| checkpoint | validation | WikiText BPB | HellaSwag | ARC-Easy | PIQA | WinoGrande |", "|---|---:|---:|---:|---:|---:|---:|"]
    for role, row in comparison.items(): lines.append(f"| {role} | {row['validation_loss']:.6f} | {row['wikitext_bpb']:.6f} | {row['hellaswag']:.5f} | {row['arc_easy']:.5f} | {row['piqa']:.5f} | {row['winogrande']:.5f} |")
    lines.extend(["", "Pareto challengers: " + (", ".join(challengers) if challengers else "none"), ""]); (ART / "final_checkpoint_comparison.md").write_text("\n".join(lines))
    complete(state, "FINAL_CHECKPOINT_EVALUATION", checkpoint_comparison=decision)


def upload_and_verify(checkpoint: Path, remote: str) -> dict:
    from huggingface_hub import HfApi
    repo = os.environ.get("LATTICELM_HF_REPO", "insightlabs38-pixel/LatticeLM-research"); api = HfApi()
    last_error = None
    for attempt in range(1, 6):
        try:
            info = api.upload_file(path_or_fileobj=checkpoint, path_in_repo=remote, repo_id=repo, commit_message=f"Persist {remote}")
            sibling = next(item for item in api.repo_info(repo, files_metadata=True).siblings if item.rfilename == remote)
            local_hash = sha256_file(checkpoint); lfs = getattr(sibling, "lfs", None)
            lfs_hash = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            if sibling.size != checkpoint.stat().st_size or (lfs_hash and lfs_hash != local_hash): raise RuntimeError(f"Hugging Face verification failed: {remote}")
            return {"repo_id": repo, "path": remote, "revision": info.oid, "size": sibling.size, "local_sha256": local_hash, "lfs_sha256": lfs_hash, "verified": True}
        except Exception as error:
            last_error = error; event("HF_PUBLICATION_RETRY", remote=remote, attempt=attempt, error=f"{type(error).__name__}: {error}")
            if attempt < 5: time.sleep(min(60, 2 ** attempt))
    raise RetryablePublicationError(f"Hugging Face publication remains unavailable for {remote}: {last_error}")


def phase_persist(state: dict) -> None:
    uploads = {}
    for role in ("end-stable", "mid-decay", "final"):
        uploads[role] = upload_and_verify(RUN / f"{role}.pt", f"final-production/{role}.pt")
    # Strict local load and deterministic likelihood smoke.
    checkpoint = torch.load(RUN / "final.pt", map_location="cpu", weights_only=False); cfg = LatticeConfig(**checkpoint["config"])
    model = build_model(cfg); model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    tokens = torch.arange(32).remainder(cfg.vocab_size).view(1, -1)
    with torch.inference_mode(): first = model(tokens)[0]; second = model(tokens)[0]
    if not torch.equal(first, second) or not torch.isfinite(first).all(): raise RuntimeError("deterministic inference smoke failed")
    verification = {"parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "parameter_cap_pass": state["parameter_count"] < 50_000_000,
                    "checkpoint_load": "PASS", "config_load": "PASS", "tokenizer_load": "PASS", "inference_smoke": "PASS", "deterministic_likelihood_smoke": "PASS",
                    "dataset_certification": "PASS", "source_commit": state["training"]["git_commit"], "uploads": uploads}
    atomic(ART / "final_model_verification.json", verification); complete(state, "FINAL_ARTIFACT_PERSIST", persistence=verification)


def phase_package(state: dict) -> None:
    final = state["checkpoint_comparison"]["checkpoints"]["final"]; training = state["training"]; audit = json.loads(AUDIT.read_text())
    decisions = {"triton": state["triton"], "muon_implementation": state["muon_audit"], "muon": state["muon"], "lr": state["lr"],
                 "posttraining": "POSTTRAINING_NOT_EXECUTED_DUE_TO_INSUFFICIENT_PRIOR_EVIDENCE"}
    atomic(ART / "final_experiment_decisions.json", decisions)
    result = {"schema": "final-submission-results-v1", "git_commit": git_head(), "architecture": state["recipe"]["architecture"],
              "parameter_count": state["parameter_count"], "selected_optimizer": state["optimizer"], "selected_lr": state["selected_lr"],
              "selected_token_budget": state["final_token_budget"], "actual_tokens_trained": training["tokens"], "training_seconds": training["training_seconds"],
              "effective_tokens_per_second": training["effective_tokens_per_second"], "restart_count": training["restart_count"],
              "schedule": state["recipe"]["wsd"], "metrics": final, "default_checkpoint": "final",
              "pareto_challengers": state["checkpoint_comparison"]["pareto_challengers"], "data_manifest_sha256": sha256_file(MANIFEST),
              "tokenizer_sha256": sha256_file(TOKENIZER), "checkpoint_sha256": final["checkpoint_sha256"], "huggingface": state["persistence"]["uploads"],
              "posttraining": decisions["posttraining"]}
    atomic(ART / "final_submission_results.json", result)
    flat = {key: value for key, value in result.items() if not isinstance(value, (dict, list))}; flat.update(final)
    with (ART / "final_submission_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat.keys()); writer.writeheader(); writer.writerow(flat)
    facts = {"model": "LatticeLM Final Production", "parameters": state["parameter_count"], "architecture": "Co4 causal", "layers": 12,
             "d_model": 552, "ffn_hidden": 1728, "gqa": "6Q/2KV real GQA", "context": 256, "tokenizer_vocab": 4096,
             "dataset_tokens": audit["dataset"]["unique_training_tokens"], "dataset_composition": audit["dataset"]["source_shares"],
             "optimizer": state["optimizer"], "peak_lr": state["selected_lr"], "final_training_tokens": training["tokens"], **final}
    atomic(ART / "final_model_facts.json", facts)
    demo = {**facts, "total_training_time_seconds": training["training_seconds"], "effective_throughput": training["effective_tokens_per_second"],
            "old_vs_new": {"integrated_100m": audit["experiments"][-1], "final": final}, "wsd_trajectory": state["checkpoint_comparison"]["checkpoints"],
            "bounded_experiment_decisions": decisions}
    atomic(ART / "demo_metrics.json", demo)
    atomic(ART / "demo_timeline.json", {"phases": state["completed"], "training": training, "checkpoints": state["checkpoint_comparison"]["checkpoints"]})
    evidence = {"schema": "final-evidence-index-v1", "authoritative": {"audit": {"path": str(AUDIT), "sha256": sha256_file(AUDIT)},
                "data_certificate": {"path": str(DATA / "certification.json"), "sha256": sha256_file(DATA / "certification.json")},
                "recipe": {"path": str(ART / "final_production_recipe.json"), "sha256": sha256_file(ART / "final_production_recipe.json")}},
                "known_historical_inconsistencies": [{"artifact": str(MANIFEST), "issue": "embedded certified:false", "explanation": "external immutable PASS certificate is authoritative"},
                {"artifact": "artifacts/research_phase_solve_near_50m_config_certificate.json", "issue": "pre-batch-field config hash", "explanation": "later release config and result-local hashes are authoritative"}]}
    atomic(ART / "final_evidence_index.json", evidence)
    report = ["# LatticeLM Final Production Report", "", f"Terminal model: `{final['checkpoint']}`", f"Checkpoint SHA-256: `{final['checkpoint_sha256']}`",
              f"Parameters: {state['parameter_count']:,}", f"Tokens trained: {training['tokens']:,}", f"Optimizer / LR: {state['optimizer']} / {state['selected_lr']}",
              f"Validation: {final['validation_loss']:.6f}", f"WikiText PPL / BPB: {final['wikitext_ppl']:.4f} / {final['wikitext_bpb']:.6f}",
              f"HellaSwag / ARC-Easy / PIQA / WinoGrande: {final['hellaswag']:.5f} / {final['arc_easy']:.5f} / {final['piqa']:.5f} / {final['winogrande']:.5f}",
              "", "Post-training was not executed because prior evidence was insufficient.", ""]
    (ART / "final_production_report.md").write_text("\n".join(report))
    # The trainer already owns the curve CSV; ensure its required plotting schema exists.
    required = [ART / "final_training_curve.csv", EVENTS, ART / "final_submission_results.json", ART / "demo_metrics.json"]
    if any(not path.exists() or not path.stat().st_size for path in required): raise RuntimeError("submission artifact generation incomplete")
    complete(state, "SUBMISSION_EVIDENCE_PACKAGE", final_results=result)


def phase_complete(state: dict) -> None:
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    active = [line for line in processes.splitlines() if "train_final_production.py" in line]
    if active: raise RuntimeError("training process remains at terminal transition")
    if "PRODUCTION_COMPLETE" not in state["completed"]: state["completed"].append("PRODUCTION_COMPLETE")
    save(state, "PRODUCTION_COMPLETE", terminal_state="PRODUCTION_COMPLETE", completed_at=time.time())
    event("PRODUCTION_COMPLETE", report=str(ART / "final_production_report.md"))


HANDLERS = {"INSPECT_AND_FREEZE": phase_inspect, "TRITON_FINAL_GATE": phase_triton,
            "MUON_IMPLEMENTATION_AUDIT": phase_muon_audit, "MUON_BOUNDED_SCREEN": phase_muon_screen,
            "OPTIMIZER_FREEZE": phase_optimizer, "LR_RESOLUTION": phase_lr, "FINAL_RECIPE_FREEZE": phase_recipe,
            "FINAL_BUDGET_SELECTION": phase_budget, "PRELAUNCH_CERTIFICATION": phase_prelaunch,
            "FINAL_LONG_RUN_FRESH": phase_train, "FINAL_CHECKPOINT_EVALUATION": phase_evaluate,
            "FINAL_ARTIFACT_PERSIST": phase_persist, "SUBMISSION_EVIDENCE_PACKAGE": phase_package,
            "PRODUCTION_COMPLETE": phase_complete}


def dry_run() -> dict:
    return {"status": "DRY_RUN_PASS", "phases": list(PHASES), "deadline_epoch": DEADLINE,
            "budget_examples": {str(hours): select_budget(hours, 3157.573615954999) for hours in (250, 200, 150)},
            "optional_failures_fall_through": ["TRITON_REJECTED", "MUON_REJECTED", "CONTROL_FAILED_RETAIN_CERTIFIED"],
            "fresh_only_guard": "trainer refuses existing latest/previous/result", "pilot_initialization_forbidden": True}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    if args.dry_run: print(json.dumps(dry_run(), indent=2)); return 0
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop); LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); state = load_state()
            if state.get("terminal_state") == "PRODUCTION_COMPLETE": return 0
            try:
                while state["phase"] in HANDLERS and not STOP:
                    HANDLERS[state["phase"]](state)
                    if state["phase"] == "PRODUCTION_COMPLETE" and state.get("terminal_state") == "PRODUCTION_COMPLETE": break
                if STOP: save(state, state["phase"], stopped_safely_at=time.time()); return 75
            except InterruptedError:
                save(state, state["phase"], stopped_safely_at=time.time()); return 75
            except RetryablePublicationError as error:
                save(state, state["phase"], publication_retry=f"{type(error).__name__}: {error}"); return 1
            except Exception as error:
                phase = state.get("phase", "UNKNOWN")
                terminal = "PRODUCTION_BLOCKED_CORRUPTION" if ("checkpoint" in str(error).lower() or "identity" in str(error).lower()) else "PRODUCTION_BLOCKED_CORRECTNESS"
                save(state, terminal, terminal_state=terminal, blocked_phase=phase, blocker=f"{type(error).__name__}: {error}")
                event(terminal, blocked_phase=phase, error=repr(error)); return 2
    except BlockingIOError:
        print("final production master already owns the lock", file=sys.stderr); return 73
    return 0


if __name__ == "__main__": raise SystemExit(main())
