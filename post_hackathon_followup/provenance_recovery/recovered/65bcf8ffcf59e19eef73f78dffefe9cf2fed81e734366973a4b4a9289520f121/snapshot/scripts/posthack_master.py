"""Single restartable, deadline-aware controller for the 15-day campaign.

The first matched-control phase began under its original service. This master
adopts it without starting a second trainer, then owns every remaining phase.
Only the Hugging Face archive helper can publish research artifacts.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from posthack_campaign import RESEARCH, ROOT, STATE, atomic_json, event, utc

MASTER_STATE = RESEARCH / "master_state.json"
LOCK = RESEARCH / ".posthack-master.lock"
PHASES = (
    ("phase1", "scripts/posthack_phase1.py"),
    ("phase2", "scripts/posthack_phase2.py"),
    ("phase3", "scripts/posthack_phase3.py"),
    ("phase4", "scripts/posthack_phase4.py"),
    ("phase5", "scripts/posthack_phase5.py"),
    ("phase6", "scripts/posthack_phase6.py"),
    ("phase8", "scripts/posthack_phase8.py"),
    ("phase7", "scripts/posthack_phase7.py"),
)


def read_state() -> dict:
    if MASTER_STATE.exists():
        result = json.loads(MASTER_STATE.read_text())
        for name, _ in PHASES:
            result["phase_status"].setdefault(name, "PENDING")
        atomic_json(MASTER_STATE, result)
        return result
    campaign = json.loads(STATE.read_text())
    result = {"schema": "posthack-master-v1", "created_utc": utc(),
              "working_deadline_utc": campaign["working_deadline_utc"],
              "archive_start_utc": utc(campaign["archive_start_epoch"]),
              "phase_status": {name: "PENDING" for name, _ in PHASES},
              "failures": {}, "last_update_utc": utc(),
              "finalized": False}
    atomic_json(MASTER_STATE, result)
    return result


def persist(state: dict) -> None:
    state["last_update_utc"] = utc()
    atomic_json(MASTER_STATE, state)


def phase1_running() -> bool:
    result = subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                             "latticelm-posthack-phase1.service"],
                            cwd=ROOT, check=False)
    return result.returncode == 0


def phase1_service_enabled() -> bool:
    result = subprocess.run(["systemctl", "--user", "is-enabled", "--quiet",
                             "latticelm-posthack-phase1.service"],
                            cwd=ROOT, check=False)
    return result.returncode == 0


def phase1_complete() -> bool:
    from posthack_phase1 import ARMS
    return all((RESEARCH / "runs" / arm[0] / "hf_archive_receipt.json").exists()
               for arm in ARMS)


def invoke_phase(name: str, script: str, master: dict) -> bool:
    log_path = RESEARCH / f"master-{name}.log"
    master["phase_status"][name] = "RUNNING"
    persist(master)
    event(name.upper(), "MASTER_PHASE_START", script=script)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{utc()}] master launch {script}\n")
        log.flush()
        child = subprocess.Popen([sys.executable, "-u", script], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        master["active_phase_pid"] = child.pid
        persist(master)
        while True:
            try:
                exit_code = child.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                subprocess.run([sys.executable, "scripts/posthack_schedule.py"],
                               cwd=ROOT, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False)
                master["heartbeat_utc"] = utc()
                persist(master)
                campaign = json.loads(STATE.read_text())
                if time.time() >= campaign["archive_start_epoch"]:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        exit_code = child.wait(timeout=600)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        exit_code = child.wait()
                    # The phase script can exit before its signal-aware trainer
                    # has finished a safe checkpoint. Wait for the whole group.
                    cutoff = time.time() + 600
                    while time.time() < cutoff:
                        try:
                            os.killpg(child.pid, 0)
                        except ProcessLookupError:
                            break
                        time.sleep(5)
                    else:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
        log.write(f"[{utc()}] exit={exit_code}\n")
    master.pop("active_phase_pid", None)
    if exit_code:
        failures = master["failures"].setdefault(name, [])
        failures.append({"at": utc(), "exit_code": exit_code,
                         "log": str(log_path.relative_to(ROOT))})
        master["phase_status"][name] = "RETRY_PENDING"
        persist(master)
        event(name.upper(), "MASTER_PHASE_FAILED", exit_code=exit_code,
              attempt=len(failures), log=str(log_path.relative_to(ROOT)))
        return False
    master["phase_status"][name] = "COMPLETED"
    persist(master)
    event(name.upper(), "MASTER_PHASE_COMPLETE", log=str(log_path.relative_to(ROOT)))
    return True


def finalize(master: dict) -> None:
    if master["finalized"]:
        return
    master["phase_status"]["final"] = "RUNNING"
    persist(master)
    log_path = RESEARCH / "master-final.log"
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-u", "scripts/posthack_finalize.py"],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        master["phase_status"]["final"] = "RETRY_PENDING"
        persist(master)
        event("FINAL", "MASTER_FINALIZATION_FAILED", exit_code=result.returncode)
        return
    master["phase_status"]["final"] = "COMPLETED"
    master["finalized"] = True
    persist(master)
    event("FINAL", "MASTER_FINALIZED")


def main() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        master = read_state()
        while True:
            campaign = json.loads(STATE.read_text())
            now = time.time()
            if now >= campaign["archive_start_epoch"]:
                if phase1_running():
                    subprocess.run(["systemctl", "--user", "stop",
                                    "latticelm-posthack-phase1.service"],
                                   cwd=ROOT, check=False)
                    while phase1_running():
                        time.sleep(5)
                for name, _ in PHASES:
                    if master["phase_status"][name] not in ("COMPLETED", "SKIPPED_RESERVE"):
                        master["phase_status"][name] = "SKIPPED_RESERVE"
                persist(master)
                finalize(master)
                if master["finalized"] or now >= campaign["working_deadline_epoch"]:
                    return
                time.sleep(300)
                continue
            if master["finalized"]:
                return
            name, script = next(((name, script) for name, script in PHASES
                                 if master["phase_status"][name] != "COMPLETED"), (None, None))
            if name is None:
                finalize(master)
                if master["finalized"]:
                    return
                time.sleep(300)
                continue
            if name == "phase1" and phase1_complete():
                master["phase_status"][name] = "COMPLETED"
                persist(master)
                event("PHASE1", "MASTER_ADOPTED_COMPLETED_PHASE")
                continue
            if name == "phase1" and phase1_service_enabled():
                if not phase1_running():
                    restart = subprocess.run(["systemctl", "--user", "start",
                                              "latticelm-posthack-phase1.service"],
                                             cwd=ROOT, check=False)
                    if restart.returncode:
                        event("PHASE1", "MASTER_SERVICE_RESTART_FAILED",
                              exit_code=restart.returncode)
                master["phase_status"][name] = "OBSERVING_EXISTING_TRAINER"
                persist(master)
                time.sleep(60)
                continue
            if not Path(ROOT / script).exists():
                master["phase_status"][name] = "WAITING_FOR_IMPLEMENTATION"
                persist(master)
                time.sleep(300)
                continue
            if not invoke_phase(name, script, master):
                failures = len(master["failures"].get(name, []))
                time.sleep(min(1800, 60 * 2 ** min(failures, 5)))
                continue
            subprocess.run([sys.executable, "scripts/posthack_schedule.py"],
                           cwd=ROOT, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)


if __name__ == "__main__":
    main()
