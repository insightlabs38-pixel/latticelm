#!/usr/bin/env python3
"""One-shot, restart-safe production checkpoint evaluation companion.

This process is intentionally external to the production master.  It watches
the production event stream, pins the first completed checkpoint created after
startup, stops the production unit through systemd, evaluates the pinned
checkpoint, and starts the unit again.  Every transition is persisted before
the next irreversible action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RUN = ART / "final_production_run"
EVENTS = ART / "final_training_events.jsonl"
STATE = ART / "production_eval_companion_state.json"
EVENT_LOG = ART / "production_eval_companion_events.jsonl"
SERVICE = "latticelm-final-production-20260917.service"
TOKENIZER = ART / "tokenizers/final_corpus_4k.json"
MANIFEST = ART / "data/data_d_v4/canonical-2250m/manifest.json"
DEFAULT_INTERVAL = 20.0
TERMINAL = {"COMPLETE", "EVALUATION_FAILED_RESUMED", "EVALUATION_COMPANION_BLOCKED_UNSAFE_PAUSE"}
STOP = False


def now() -> float:
    return time.time()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_state() -> dict:
    if not STATE.exists():
        return {}
    return json.loads(STATE.read_text(encoding="utf-8"))


def save(state: dict, status: str | None = None, **fields) -> None:
    if status is not None:
        state["status"] = status
    state.update(fields); state["updated_at"] = now()
    atomic(STATE, state)
    record = {"at": now(), "at_utc": datetime.now(timezone.utc).isoformat(),
              "event": "STATE", "status": state.get("status"), **fields}
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def run(command: list[str], timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=check)


def systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["systemctl", *args], check=check, timeout=1500)


def production_processes() -> list[str]:
    output = run(["ps", "-eo", "pid=,args="], check=True).stdout
    return [line.strip() for line in output.splitlines()
            if ("run_final_production_master.py" in line or "train_final_production.py" in line)
            and "production_eval_companion" not in line]


def service_active() -> bool:
    return systemctl("is-active", "--quiet", SERVICE, check=False).returncode == 0


def latest_checkpoint() -> tuple[Path, dict, str] | None:
    path = RUN / "latest.pt"; sidecar = RUN / "latest.sha256"
    if not (path.is_file() and sidecar.is_file()):
        return None
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    digest = sha256(path)
    after = (path.stat().st_size, path.stat().st_mtime_ns)
    if before != after or digest != sidecar.read_text().strip():
        return None
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = ("schema", "model", "optimizers", "config", "config_sha256", "tokens",
                "manifest_sha256", "tokenizer_sha256", "next_batch_sha256", "validation_loss")
    if any(key not in payload for key in required) or payload["schema"] != "final-production-checkpoint-v1":
        return None
    if before != (path.stat().st_size, path.stat().st_mtime_ns) or sha256(path) != digest:
        return None
    return path, payload, digest


def event_records() -> list[dict]:
    if not EVENTS.exists():
        return []
    records = []
    with EVENTS.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
                if isinstance(item, dict): records.append(item)
            except json.JSONDecodeError:
                continue
    return records


def checkpoint_event_after(started: float, startup_tokens: int) -> dict | None:
    candidates = [item for item in event_records()
                  if item.get("event") == "CHECKPOINT" and float(item.get("at", 0)) >= started
                  and int(item.get("tokens", 0)) > startup_tokens and item.get("sha256")]
    return max(candidates, key=lambda item: float(item["at"])) if candidates else None


def pin_checkpoint(state: dict, event: dict, candidate: tuple[Path, dict, str]) -> dict:
    path, payload, digest = candidate
    if int(payload["tokens"]) != int(event["tokens"]) or digest != event["sha256"]:
        raise RuntimeError("checkpoint event and payload identity mismatch")
    directory = ART / f"production_milestone_eval_{int(payload['tokens'])}"
    directory.mkdir(parents=True, exist_ok=True)
    pinned = directory / "checkpoint.pt"
    if not pinned.exists():
        os.link(path, pinned)
    elif sha256(pinned) != digest:
        raise RuntimeError("existing pinned checkpoint hash mismatch")
    (directory / "checkpoint.sha256").write_text(digest + "\n", encoding="utf-8")
    import torch
    reread = torch.load(pinned, map_location="cpu", weights_only=False)
    if int(reread["tokens"]) != int(payload["tokens"]) or sha256(pinned) != digest:
        raise RuntimeError("pinned checkpoint failed immutable verification")
    identity = {"tokens": int(payload["tokens"]), "step": int(payload["step"]),
                "checkpoint": str(pinned), "source_checkpoint": str(path),
                "checkpoint_sha256": digest, "config_sha256": payload["config_sha256"],
                "tokenizer_sha256": payload["tokenizer_sha256"],
                "manifest_sha256": payload["manifest_sha256"], "lr": payload["config"].get("learning_rate"),
                "validation_loss": payload["validation_loss"], "source_validation": payload["source_validation"],
                "train_loss": payload.get("train_loss"), "checkpoint_event": event,
                "pinned_at": now()}
    atomic(directory / "checkpoint_identity.json", identity)
    save(state, "CHECKPOINT_VERIFIED", checkpoint=identity, output_dir=str(directory))
    return identity


def pause_production(state: dict) -> None:
    save(state, "PRODUCTION_PAUSING", pause_requested_at=now())
    result = systemctl("stop", SERVICE, check=False)
    if result.returncode != 0:
        save(state, "EVALUATION_COMPANION_BLOCKED_UNSAFE_PAUSE", pause_error=result.stderr[-2000:])
        raise RuntimeError("production stop request failed")
    deadline = time.monotonic() + 1500
    while time.monotonic() < deadline and (service_active() or production_processes()):
        time.sleep(2)
    if service_active() or production_processes():
        save(state, "EVALUATION_COMPANION_BLOCKED_UNSAFE_PAUSE", pause_error="production remained active")
        raise RuntimeError("production did not stop cleanly")
    state["paused_at"] = now(); state["pause_transition_seconds"] = state["paused_at"] - state["pause_requested_at"]
    save(state, "PRODUCTION_PAUSED")


def evaluate(state: dict) -> None:
    identity = state["checkpoint"]; output = Path(state["output_dir"]); checkpoint = Path(identity["checkpoint"])
    save(state, "EVALUATING", evaluation_started_at=now())
    commands = [
        [sys.executable, "scripts/evaluate_wikitext103.py", "--checkpoint", str(checkpoint), "--tokenizer", str(TOKENIZER),
         "--output", str(output / "wikitext.json"), "--batch-size", "16", "--threads", "16"],
        [sys.executable, "scripts/evaluate_gibc.py", "--checkpoint", str(checkpoint), "--tokenizer", str(TOKENIZER),
         "--output", str(output / "gibc.json"), "--tasks", "hellaswag,arc_easy,piqa,winogrande",
         "--batch-size", "16", "--threads", "16"],
    ]
    for command in commands:
        result = run(command, timeout=3600, check=False)
        if result.returncode:
            raise RuntimeError(f"evaluation failed ({result.returncode}): {result.stderr[-1500:]}")
    save(state, "EVALUATION_COMPLETE", evaluation_finished_at=now(), evaluation_duration_seconds=now() - state["evaluation_started_at"])


def restart_and_verify(state: dict) -> None:
    save(state, "PRODUCTION_RESTARTING", restart_requested_at=now())
    result = systemctl("start", SERVICE, check=False)
    if result.returncode:
        raise RuntimeError(f"production restart failed: {result.stderr[-2000:]}")
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        processes = production_processes()
        if service_active() and len(processes) == 2:
            break
        time.sleep(2)
    if not service_active() or len(production_processes()) != 2:
        raise RuntimeError("resume did not produce exactly one master and trainer")
    resumed = False; start_tokens = int(state["checkpoint"]["tokens"]); end = time.monotonic() + 300
    while time.monotonic() < end:
        for item in reversed(event_records()):
            if item.get("event") in {"RESUME", "CHECKPOINT"} and int(item.get("tokens", 0)) > start_tokens:
                resumed = True; state["resume_event"] = item; break
        if resumed: break
        time.sleep(5)
    if not resumed:
        raise RuntimeError("production service active but bounded resume progress was not observed")
    state["resume_verified_at"] = now(); state["resume_verification_seconds"] = state["resume_verified_at"] - state["restart_requested_at"]
    save(state, "RESUME_VERIFIED")


def report(state: dict) -> None:
    output = Path(state["output_dir"]); wiki = json.loads((output / "wikitext.json").read_text()); gibc = json.loads((output / "gibc.json").read_text())
    baseline = ART / "final_recipe_experiments/final-muon-100m"
    report = {"checkpoint": state["checkpoint"], "production_resume": state.get("resume_event"),
              "wikitext": wiki, "gibc": gibc, "baseline_100m": {"result": str(baseline / "result.json"), "wikitext": str(baseline / "wikitext.json"), "gibc": str(baseline / "gibc.json")},
              "pause_resume": {k: state.get(k) for k in ("pause_requested_at", "paused_at", "pause_transition_seconds", "evaluation_started_at", "evaluation_finished_at", "evaluation_duration_seconds", "restart_requested_at", "resume_verified_at", "resume_verification_seconds")},
              "comparability": "same corrected LatticeHarnessLM and lm-eval 0.4.13 for GIBC; dedicated WikiText values only compare to dedicated WikiText baseline",
              "generated_at": now()}
    atomic(output / "comparison_100m.json", report)
    lines = [f"# Production milestone evaluation: {state['checkpoint']['tokens']} tokens", "", f"Checkpoint SHA: `{state['checkpoint']['checkpoint_sha256']}`", "", "Production resumed after raw outputs were written.", "", "## Results", "", f"- DATA-D-v4 validation loss: {state['checkpoint']['validation_loss']}", f"- WikiText PPL: {wiki.get('perplexity')}", f"- WikiText BPB: {wiki.get('bits_per_byte')}"]
    for task in ("hellaswag", "arc_easy", "piqa", "winogrande"):
        values = gibc.get("results", {}).get(task, {}); lines.append(f"- {task}: acc={values.get('acc,none')}, acc_norm={values.get('acc_norm,none')}")
    (output / "comparison_100m.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(state, "COMPLETE", report=str(output / "comparison_100m.md"), completed_at=now())


def signal_handler(*_) -> None:
    global STOP
    STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--poll-seconds", type=float, default=DEFAULT_INTERVAL); args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal_handler); signal.signal(signal.SIGINT, signal_handler)
    state = load_state()
    if state.get("status") in TERMINAL:
        return 0
    if not state:
        candidate = latest_checkpoint()
        if candidate is None: raise SystemExit("no complete startup checkpoint")
        path, payload, digest = candidate
        state = {"schema": "production-eval-companion-v1", "status": "WAITING_FOR_NEW_CHECKPOINT", "started_at": now(),
                 "startup_checkpoint": {"tokens": int(payload["tokens"]), "step": int(payload["step"]), "path": str(path), "sha256": digest,
                                         "mtime_ns": path.stat().st_mtime_ns, "size": path.stat().st_size}, "service": SERVICE}
        save(state)
    while not STOP and state.get("status") == "WAITING_FOR_NEW_CHECKPOINT":
        event = checkpoint_event_after(float(state["started_at"]), int(state["startup_checkpoint"]["tokens"]))
        if event:
            candidate = latest_checkpoint()
            if candidate and int(candidate[1]["tokens"]) == int(event["tokens"]):
                identity = pin_checkpoint(state, event, candidate); state["output_dir"] = str(ART / f"production_milestone_eval_{identity['tokens']}")
                save(state, "CHECKPOINT_DETECTED")
                break
        time.sleep(args.poll_seconds)
    if STOP: return 75
    try:
        pause_production(state); evaluate(state); restart_and_verify(state); report(state)
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        if state.get("status") in {"PRODUCTION_PAUSED", "EVALUATING", "EVALUATION_COMPLETE", "PRODUCTION_RESTARTING"} and not service_active():
            try: restart_and_verify(state)
            except Exception as restart_error: state["restart_error"] = repr(restart_error)
        if service_active(): save(state, "EVALUATION_FAILED_RESUMED", failure=state["error"])
        else: save(state, "EVALUATION_COMPANION_BLOCKED_UNSAFE_PAUSE", failure=state["error"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
