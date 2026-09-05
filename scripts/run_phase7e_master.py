"""Autonomous, restartable Phase 7E controller.

The controller intentionally delegates numerical training to the frozen Phase 7D
runner.  It owns provenance checks, deadlines, subprocess recovery, milestone
evaluation, persistence metadata, and an honest time-aware DATA-D fallback.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
STATE = ART / "phase7e_master_state.json"
EVENTS = ART / "phase7e_master_events.jsonl"
LOCK = ART / "phase7e_master.lock"
LOG = ART / "logs/phase7e_master.log"
RECOVERY = ART / "checkpoints/co4-l-data-rich-25m/latest.pt"
RECOVERY_HASH = RECOVERY.with_suffix(".sha256")
CONFIG = ROOT / "configs/phase7d_co4_l.json"
TOKENIZER = ART / "tokenizers/babylm_2026_4k.json"
PHASE7D_COMMIT = "016a7dfb5989c161b4bb8d39a4e8260246761aca"
VALID_HF_REVISION = "4bd98a6f87f5d4594cd7fe0ba3aa28c659e5ae32"
SOFT_HOURS, HARD_HOURS = 11.5, 12.0
STAGES = ["PREFLIGHT", "DATA_C_TO_50M", "EVAL_50M", "DATA_C_POOL_EXPANSION",
          "DATA_C_TO_100M", "EVAL_100M", "DATA_D_PREPARATION",
          "DATA_D_25M", "EVAL_DATA_D_25M", "FINAL_REPORT"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(directory)
    finally: os.close(directory)


def load_state() -> dict[str, Any] | None:
    if not STATE.exists(): return None
    value = json.loads(STATE.read_text())
    if not isinstance(value, dict) or "run_id" not in value: raise ValueError("invalid master state")
    return value


def event(kind: str, **metadata: Any) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"timestamp": now(), "event": kind, **metadata}, sort_keys=True)
    fd = os.open(EVENTS, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
    try: os.write(fd, (line + "\n").encode()); os.fsync(fd)
    finally: os.close(fd)


class MasterLock:
    def __init__(self, path: Path = LOCK): self.path, self.handle = path, None
    def acquire(self, blocking: bool = False) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try: fcntl.flock(self.handle, flags)
        except BlockingIOError: self.handle.close(); self.handle = None; return False
        self.handle.seek(0); self.handle.truncate(); self.handle.write(str(os.getpid()) + "\n"); self.handle.flush()
        return True
    def release(self) -> None:
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN); self.handle.close(); self.handle = None
    def __enter__(self):
        if not self.acquire(): raise BlockingIOError("Phase 7E master already owns the lock")
        return self
    def __exit__(self, *_): self.release()


def initial_state(source_commit: str) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    return {"run_id": f"phase7e-{started:%Y%m%dT%H%M%SZ}", "source_git_commit": source_commit,
            "started_at": started.isoformat(), "soft_deadline": (started + timedelta(hours=SOFT_HOURS)).isoformat(),
            "hard_deadline": (started + timedelta(hours=HARD_HOURS)).isoformat(), "current_stage": "PREFLIGHT",
            "completed_stages": [], "current_model": "Co4-L DATA-C", "current_token_count": 25_000_000,
            "current_data_manifest": "artifacts/phase7a_selected_dataset_manifest.json",
            "latest_checkpoint": str(RECOVERY.relative_to(ROOT)), "checkpoint_sha256": digest(RECOVERY),
            "latest_validation_metrics": {}, "best_validation": None, "retry_counts": {},
            "hf_upload_state": {"revision_25m": VALID_HF_REVISION, "status": "verified_phase7d"},
            "spot_interruption_count": 0, "last_error": None, "stop_requested": False,
            "master_pid": os.getpid(), "trainer_pid": None, "updated_at": now()}


def deadline(state: dict[str, Any], which: str) -> datetime:
    return datetime.fromisoformat(state[f"{which}_deadline"])


def can_start(state: dict[str, Any], seconds_needed: float, reserve: float = 1800) -> bool:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds_needed + reserve) < deadline(state, "hard")


def save(state: dict[str, Any]) -> None:
    state["updated_at"] = now(); atomic_json(STATE, state)


def check_phase7d() -> dict[str, Any]:
    required = [RECOVERY, RECOVERY_HASH, CONFIG, TOKENIZER, ART / "phase7d_control_manifest.json",
                ART / "phase7d_l_25m_metrics.json", ART / "future_data_scaling_plan.md",
                ART / "data/phase7a/babylm_train.int32", ART / "data/phase7a/finewebedu_train.int32",
                ART / "data/phase7a/common_validation.int32"]
    missing = [str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p) for p in required if not p.exists()]
    if missing: raise RuntimeError(f"missing Phase 7D inputs: {missing}")
    expected = RECOVERY_HASH.read_text().strip()
    if digest(RECOVERY) != expected: raise RuntimeError("corrected Co4-L checkpoint checksum mismatch")
    if "archive/phase7d_invalid_unpaired_l" in str(RECOVERY): raise RuntimeError("invalid archived lineage selected")
    import torch
    payload = torch.load(RECOVERY, map_location="cpu", weights_only=False)
    needed = {"model", "optimizer", "scheduler", "python_rng_state", "torch_rng_state",
              "data_stream_state", "tokens_seen", "cumulative_training_seconds", "best_validation_loss"}
    absent = sorted(needed - payload.keys())
    if absent: raise RuntimeError(f"resume fields absent: {absent}")
    if payload.get("lineage") != "co4-l-data-rich-25m" or payload.get("source") != "DATA-C":
        raise RuntimeError("checkpoint is not corrected paired Co4-L DATA-C lineage")
    if int(payload["tokens_seen"]) != 25_000_000: raise RuntimeError("checkpoint is not at 25M")
    manifest = json.loads((ART / "phase7d_control_manifest.json").read_text())
    checks = {"babylm_train.int32": manifest["provenance"]["babylm_train_sha256"],
              "finewebedu_train.int32": manifest["provenance"]["finewebedu_train_sha256"],
              "common_validation.int32": manifest["provenance"]["common_validation_sha256"]}
    for name, expected_hash in checks.items():
        if digest(ART / "data/phase7a" / name) != expected_hash: raise RuntimeError(f"DATA-C hash mismatch: {name}")
    usage = shutil.disk_usage(ROOT)
    if usage.free < 12 * 2**30: raise RuntimeError("less than 12 GiB free; unsafe for rolling checkpoints")
    return {"checkpoint": expected, "free_bytes": usage.free, "tokens": payload["tokens_seen"]}


def run_child(state: dict[str, Any], label: str, command: list[str], retries: int = 2) -> None:
    for attempt in range(retries + 1):
        if load_state().get("stop_requested"): raise InterruptedError("graceful stop requested")
        event("SUBPROCESS_STARTED", stage=state["current_stage"], label=label, attempt=attempt, command=command[1:3])
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=False)
        state["trainer_pid"] = child.pid; save(state)
        code = child.wait(); state["trainer_pid"] = None; save(state)
        if code == 0: return
        state["retry_counts"][label] = attempt + 1; state["last_error"] = f"{label} exited {code}"; save(state)
        event("SUBPROCESS_FAILED", label=label, exit_code=code, attempt=attempt)
        if datetime.now(timezone.utc) >= deadline(state, "hard"): raise TimeoutError("hard deadline reached")
        if attempt == retries: raise RuntimeError(state["last_error"])
        state["spot_interruption_count"] += 1; event("SPOT_RESUME", label=label, retry=attempt + 1)
        time.sleep(min(60, 5 * 2**attempt))


def latest_curve() -> dict[str, str]:
    path = ART / "co4_l_data_rich_curve.csv"
    with path.open() as handle: return list(csv.DictReader(handle))[-1]


def train_to(state: dict[str, Any], target: int) -> None:
    run_child(state, f"train_to_{target}", [sys.executable, "scripts/run_phase7d_training.py", "--model", "l",
              "--resume", "--phase7e", "--target-tokens", str(target)])
    row = latest_curve(); state["current_token_count"] = int(row["training_tokens"])
    state["checkpoint_sha256"] = RECOVERY_HASH.read_text().strip(); state["latest_validation_metrics"] = row
    state["best_validation"] = float(row["common_validation_loss"]); save(state)


def evaluate_milestone(state: dict[str, Any], tokens: int) -> None:
    tag = f"{tokens//1_000_000}m"; wiki = ART / f"wikitext_l_{tag}.json"; gibc = ART / f"gibc_l_{tag}_raw.json"
    run_child(state, f"wikitext_{tag}", [sys.executable, "scripts/evaluate_wikitext103.py", "--checkpoint",
              str(RECOVERY), "--tokenizer", str(TOKENIZER), "--output", str(wiki), "--threads", "4"])
    run_child(state, f"gibc_{tag}", [sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", str(RECOVERY),
              "--tokenizer", str(TOKENIZER), "--output", str(gibc), "--tasks",
              "hellaswag,arc_easy,piqa,winogrande", "--threads", "4"])
    metrics = {"training": latest_curve(), "wikitext": json.loads(wiki.read_text()),
               "gibc_raw": str(gibc.relative_to(ROOT))}
    atomic_json(ART / f"phase7e_l_{tag}_metrics.json", metrics)
    event("GIBC_EVAL_COMPLETED", tokens=tokens, output=str(gibc.relative_to(ROOT)))


def write_placeholder_data_reports(reason: str) -> None:
    atomic_json(ART / "data_d_broad_manifest.json", {"schema_version": 1, "corpus_id": "DATA-D-BROAD-v1",
        "prepared": False, "mixture": {"fineweb_edu": .50, "wikipedia_en": .20, "fineweb_general": .15, "babylm_2026_strict": .15},
        "reason": reason, "architecture_reference": "artifacts/future_data_scaling_plan.md"})
    for name, title in (("data_d_quality_report.md", "DATA-D quality report"),
                        ("data_d_dedup_report.md", "DATA-D deduplication report"),
                        ("data_d_decontamination_report.md", "DATA-D decontamination report")):
        (ART / name).write_text(f"# {title}\n\nDATA-D-BROAD-v1 was not materialized: {reason}. No unverified substitute data was admitted.\n")


def final_report(state: dict[str, Any]) -> None:
    completed = set(state["completed_stages"]); reason = state.get("last_error") or "not reached before the deadline"
    if not (ART / "data_d_broad_manifest.json").exists(): write_placeholder_data_reports(reason)
    (ART / "phase7e_overnight_report.md").write_text(
        "# Phase 7E overnight report\n\nThis report is generated from persisted state; absent measurements are never fabricated.\n\n"
        f"CO4-L 50M REACHED: {'YES' if 'DATA_C_TO_50M' in completed else 'NO'}\n\n"
        f"CO4-L 100M REACHED: {'YES' if 'DATA_C_TO_100M' in completed else 'NO'}\n\n"
        f"DATA-D-BROAD-v1 PREPARED: {'YES' if 'DATA_D_PREPARATION' in completed else 'NO'}\n\n"
        f"DATA-D 25M PILOT COMPLETED: {'YES' if 'DATA_D_25M' in completed else 'NO'}\n\n"
        "REASONING BENCHMARK TRAJECTORY: INCONCLUSIVE pending structured metric collation.\n\n"
        "SHOULD ~24M MODEL BE TESTED NEXT: INCONCLUSIVE. No ~24M model was trained.\n")
    (ART / "phase7e_decision.md").write_text("# Phase 7E decision\n\nDo not automatically train ~24M. Review DATA-C milestones and, if absent, finish the verified DATA-D pilot next.\n")
    for path, fields in ((ART/"phase7e_training_curves.csv", ["model","tokens","common_validation_loss"]),
                         (ART/"phase7e_data_c_scaling.csv", ["tokens","common_validation_loss"]),
                         (ART/"phase7e_gibc_milestones.csv", ["tokens","task","accuracy"]),
                         (ART/"data_d_25m_comparison.csv", ["regime","tokens","metric","value"])):
        if not path.exists():
            with path.open("w", newline="") as handle: csv.writer(handle).writerow(fields)
    event("FINAL_REPORT_WRITTEN")


def execute_stage(state: dict[str, Any], stage: str) -> None:
    if stage == "PREFLIGHT": check_phase7d(); event("PREFLIGHT_PASSED")
    elif stage == "DATA_C_TO_50M": train_to(state, 50_000_000)
    elif stage == "EVAL_50M": evaluate_milestone(state, 50_000_000)
    elif stage == "DATA_C_POOL_EXPANSION":
        run_child(state, "expand_fineweb", [sys.executable, "scripts/expand_phase7e_fineweb.py", "--tokens", "50_000_000"], retries=1)
        state["current_data_manifest"] = "artifacts/phase7e_fineweb_expansion_manifest.json"; save(state)
        event("DATA_POOL_EXPANDED", manifest=state["current_data_manifest"])
    elif stage == "DATA_C_TO_100M": train_to(state, 100_000_000)
    elif stage == "EVAL_100M": evaluate_milestone(state, 100_000_000)
    elif stage in {"DATA_D_PREPARATION", "DATA_D_25M", "EVAL_DATA_D_25M"}:
        raise RuntimeError(f"prerequisite stage unavailable: {stage}")
    elif stage == "FINAL_REPORT": final_report(state)


def run_master(dry_run: bool = False) -> int:
    lock = MasterLock()
    if not lock.acquire():
        print(json.dumps({"status": "already_running", "state": load_state()}, indent=2)); return 0
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        state = load_state()
        if state and state.get("current_stage") == "MASTER_EXITED":
            print("Phase 7E has already exited; refusing to restart silently"); return 0
        if state is None: state = initial_state(source_commit)
        if dry_run:
            result = check_phase7d(); print(json.dumps({"dry_run": True, "resolved_stage": state["current_stage"],
                "checkpoint": str(RECOVERY), "verification": result, "estimated_stages": STAGES}, indent=2)); return 0
        state["master_pid"] = os.getpid(); save(state); event("MASTER_STARTED", run_id=state["run_id"], source_commit=source_commit)
        start_index = STAGES.index(state["current_stage"]) if state["current_stage"] in STAGES else 0
        for stage in STAGES[start_index:]:
            fresh = load_state() or state; state.update(fresh)
            if state.get("stop_requested"): raise InterruptedError("graceful stop requested")
            if datetime.now(timezone.utc) >= deadline(state, "soft") and stage not in {"FINAL_REPORT"}:
                event("SOFT_DEADLINE_REACHED", skipped_stage=stage); break
            state["current_stage"] = stage; save(state); event("STAGE_STARTED", stage=stage)
            try: execute_stage(state, stage)
            except Exception as error:
                state["last_error"] = f"{type(error).__name__}: {error}"; save(state)
                event("STAGE_SKIPPED", stage=stage, reason=state["last_error"])
                break
            state["completed_stages"].append(stage); save(state); event("STAGE_COMPLETED", stage=stage)
        if "FINAL_REPORT" not in state["completed_stages"]:
            state["current_stage"] = "FINAL_REPORT"; save(state); final_report(state); state["completed_stages"].append("FINAL_REPORT")
        state["current_stage"] = "MASTER_EXITED"; save(state); event("MASTER_EXITED", error=state.get("last_error"))
        return 0
    finally: lock.release()


def status() -> int:
    state = load_state(); held = not MasterLock().acquire()
    print(json.dumps({"lock_held": held, "state": state, "log": str(LOG)}, indent=2)); return 0


def request_stop() -> int:
    state = load_state()
    if not state: print("No Phase 7E state exists"); return 0
    state["stop_requested"] = True; save(state); event("STOP_REQUESTED")
    pid = state.get("trainer_pid")
    if pid:
        with contextlib.suppress(ProcessLookupError): os.kill(int(pid), signal.SIGTERM)
    print("Graceful stop requested"); return 0


def smoke(base: Path) -> int:
    global STATE, EVENTS, LOCK, LOG
    base.mkdir(parents=True, exist_ok=True); STATE, EVENTS, LOCK, LOG = (base/x for x in ("state.json","events.jsonl","master.lock","master.log"))
    atomic_json(STATE, {"run_id":"smoke","current_stage":"tiny_training","tokens":0})
    first = load_state(); first["tokens"] = 8; first["next_batch_hash"] = hashlib.sha256(b"synthetic-batch-1").hexdigest(); atomic_json(STATE, first)
    recovered = load_state(); assert recovered["tokens"] == 8 and recovered["next_batch_hash"] == hashlib.sha256(b"synthetic-batch-1").hexdigest()
    shard = base/"fixture.int32"; shard.write_bytes((1).to_bytes(4,"little",signed=True)*8)
    atomic_json(base/"fixture.manifest.json", {"sha256":digest(shard),"tokens":8,"source":"synthetic"})
    recovered["current_stage"]="FINAL_REPORT"; atomic_json(STATE,recovered); event("FINAL_REPORT_WRITTEN", smoke=True)
    print(json.dumps({"smoke":"passed","state":str(STATE),"shard_sha256":digest(shard)})); return 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--status", action="store_true")
    parser.add_argument("--request-stop", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", type=Path)
    args = parser.parse_args()
    if args.status: return status()
    if args.request_stop: return request_stop()
    if args.smoke: return smoke(args.smoke)
    return run_master(args.dry_run)


if __name__ == "__main__": raise SystemExit(main())
