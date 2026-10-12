#!/usr/bin/env python3
"""Isolated, resumable low-alpha follow-up to the completed salvage run."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_posttraining_salvage as salvage
from latticelm.posttraining.state import atomic_json, sha256

OUT = ROOT / "artifacts/posttraining_salvage_followup"
PLAN_PATH = OUT / "experiment_plan.json"
STATE_PATH = OUT / "state.json"
PARENT_STATE_PATH = ROOT / "artifacts/posttraining_salvage/state.json"


def now_iso(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if epoch is None else epoch,
                                  timezone.utc).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text())


def append_event(state: dict, kind: str, **details) -> None:
    record = {"at": now_iso(), "kind": kind, **details}
    state.setdefault("events", []).append(record)
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    salvage.save(state)


def write_plan_copy() -> dict:
    plan = load_json(PLAN_PATH)
    if not PARENT_STATE_PATH.is_file():
        raise RuntimeError(f"parent state missing: {PARENT_STATE_PATH}")
    parent = load_json(PARENT_STATE_PATH)
    if parent.get("phase") != "COMPLETE":
        raise RuntimeError("parent salvage run is not complete")
    if plan["baseline"]["checkpoint_sha256"] != parent["base_sha256"]:
        raise RuntimeError("plan BASE hash differs from completed parent run")
    if plan["deadline_at"] != parent["deadline_at"]:
        raise RuntimeError("plan deadline differs from completed parent run")
    return plan, parent


def initial_state(plan: dict, parent: dict) -> dict:
    keep = {"BASE"}
    for exp in plan["experiments"]:
        keep.add(exp["endpoint"])
        if exp["candidate_id"] in parent["candidates"]:
            keep.add(exp["candidate_id"])
    candidates = {key: parent["candidates"][key] for key in keep}
    if "BASE" not in parent["final_evaluations"]:
        raise RuntimeError("completed parent run has no official BASE evaluation")
    deadline = float(parent["deadline_epoch"])
    left = deadline - time.time()
    if left <= 0:
        raise RuntimeError("follow-up deadline has already passed")
    return {
        "schema": "posttraining-salvage-followup-v1",
        "phase": "READY",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "deadline_epoch": deadline,
        "deadline_monotonic": time.monotonic() + left,
        "boot_id": salvage.boot_id(),
        "start_at": now_iso(),
        "deadline_at": parent["deadline_at"],
        "base_sha256": parent["base_sha256"],
        "tokenizer_sha256": parent["tokenizer_sha256"],
        "manifest_sha256": parent["manifest_sha256"],
        "code_commit_sha": parent["code_commit_sha"],
        "optimizer": parent.get("optimizer"),
        "recovery_lr": parent.get("recovery_lr"),
        "microbatch": parent.get("microbatch"),
        "candidates": candidates,
        "throughput": {},
        "final_evaluations": {"BASE": parent["final_evaluations"]["BASE"]},
        "training_seconds": 0.0,
        "proxy_seconds": 0.0,
        "official_seconds": 0.0,
        "evidence_seconds": 0.0,
        "failures": [],
        "events": [],
        "experiment_plan": str(PLAN_PATH),
        "parent_state": str(PARENT_STATE_PATH),
    }


def verify_inputs(plan: dict, state: dict) -> None:
    if sha256(state["candidates"]["BASE"]["checkpoint"]) != state["base_sha256"]:
        raise RuntimeError("BASE checkpoint SHA mismatch")
    reference = salvage.validate_model(state["candidates"]["BASE"]["checkpoint"])
    endpoints = {exp["endpoint"] for exp in plan["experiments"]}
    for endpoint in sorted(endpoints):
        item = state["candidates"][endpoint]
        if sha256(item["checkpoint"]) != item["checkpoint_sha256"]:
            raise RuntimeError(f"{endpoint} checkpoint SHA mismatch")
        salvage.validate_model(item["checkpoint"], reference)
    for exp in plan["experiments"]:
        cid = exp["candidate_id"]
        if cid in state["candidates"]:
            item = state["candidates"][cid]
            if sha256(item["checkpoint"]) != item["checkpoint_sha256"]:
                raise RuntimeError(f"existing candidate SHA mismatch: {cid}")
            if item.get("alpha") != exp["alpha"] or item.get("lineage") != exp["endpoint"]:
                raise RuntimeError(f"existing candidate metadata mismatch: {cid}")


def init_or_resume(plan: dict, parent: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        state = load_json(STATE_PATH)
        if state.get("schema") != "posttraining-salvage-followup-v1":
            raise RuntimeError("unexpected follow-up state schema")
        if state.get("base_sha256") != parent["base_sha256"]:
            raise RuntimeError("follow-up state BASE differs from parent")
        if state.get("deadline_epoch") != parent["deadline_epoch"]:
            raise RuntimeError("follow-up state deadline differs from parent")
        state["deadline_monotonic"] = time.monotonic() + max(0, state["deadline_epoch"] - time.time())
        state["boot_id"] = salvage.boot_id()
        for key in ("training_seconds", "proxy_seconds", "official_seconds", "evidence_seconds"):
            state.setdefault(key, 0.0)
        salvage.save(state)
        return state
    state = initial_state(plan, parent)
    salvage.save(state)
    return state


def report(state: dict, plan: dict) -> None:
    base = state["final_evaluations"]["BASE"]
    base_mean = sum(base["gibc_raw"].values()) / 4
    rows = [
        "# Post-training salvage follow-up", "",
        f"Started: {state['start_at']}", f"Deadline: {state['deadline_at']}",
        f"Updated: {now_iso()}", "",
        "Candidate metrics are compared with the official BASE evaluation. A proxy improvement is not an official capability claim.", "",
        "| Candidate | Weight | Data D | D vs BASE | Wiki BPB | BPB vs BASE | GIBC mean | GIBC Δ | Proxy 1024 | Retention gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    experiment_map = {e["candidate_id"]: e for e in plan["experiments"]}
    for cid, exp in experiment_map.items():
        official = state["final_evaluations"].get(cid)
        screen = state.get("screen_evaluations", {}).get(cid)
        if official is None and screen is None:
            rows.append(f"| {cid} | {exp['alpha']:.2f} | pending | pending | pending | pending | pending | pending | pending | pending |")
            continue
        item = official or screen
        data_loss = item["data_d"]["loss"]
        wiki_bpb = item["wikitext_bpb"]
        data_ratio = data_loss / base["data_d"]["loss"]
        wiki_ratio = wiki_bpb / base["wikitext_bpb"]
        gates = [
            data_ratio <= 1.20,
            wiki_ratio <= 1.25,
        ]
        if official is not None:
            mean = sum(official["gibc_raw"].values()) / 4
            gibc_text = f"{mean:.4f}"
            gibc_delta = f"{mean-base_mean:+.4f}"
            gate_text = (f"DataD {'PASS' if gates[0] else 'FAIL'}; Wiki {'PASS' if gates[1] else 'FAIL'}; "
                         f"GIBC {'PASS' if mean >= base_mean+0.005 else 'FAIL'}")
        else:
            gibc_text, gibc_delta = "not run", "—"
            gate_text = f"DataD {'PASS' if gates[0] else 'FAIL'}; Wiki {'PASS' if gates[1] else 'FAIL'}; GIBC not run"
        rows.append(
            f"| {cid} | {exp['alpha']:.2f} | {data_loss:.4f} | {data_ratio:.3f}× | "
            f"{wiki_bpb:.4f} | {wiki_ratio:.3f}× | {gibc_text} | {gibc_delta} | "
            f"{item['proxy_1024']['accuracy']:.4f} | {gate_text} |"
        )
    rows += ["", "## Run status", "", f"Phase: `{state['phase']}`", ""]
    for failure in state.get("failures", []):
        suffix = f" (recovered at {failure['resolved_at']})" if failure.get("resolved_at") else " (unresolved)"
        rows.append(f"- Failure at {failure['at']}: {failure['error']}{suffix}")
    tmp = OUT / ".followup_report.md.tmp"
    tmp.write_text("\n".join(rows) + "\n")
    os.replace(tmp, OUT / "followup_report.md")


def ensure_proxy256(state: dict, cid: str) -> None:
    """Adopt a completed proxy artifact after a crash between file and state writes."""
    ensure_proxy(state, cid, 256)


def ensure_proxy(state: dict, cid: str, n: int) -> None:
    candidate = state["candidates"][cid]
    key = f"proxy_{n}"
    if key in candidate:
        salvage.proxy(state, cid, n)  # validates cached metadata
        return
    path = OUT / "candidates" / cid / f"proxy-{n}.json"
    if not path.exists():
        salvage.proxy(state, cid, n)
        return
    result = load_json(path)
    if (result.get("checkpoint_sha256") != candidate["checkpoint_sha256"] or
            result.get("examples") != n or
            len(result.get("fixed_example_ids", [])) != 2*n):
        raise RuntimeError(f"orphan proxy artifact failed integrity checks: {path}")
    for key in ("v2_ranking_accuracy", "v2_mean_margin", "data_d_validation"):
        if not isinstance(result.get(key), (int, float)):
            raise RuntimeError(f"orphan proxy metric missing: {key}")
    candidate[key] = result
    durations = [e.get("seconds", 0.0) for e in state.get("events", [])
                 if e.get("kind") == "command_end" and
                 e.get("label") == f"{cid}-proxy-{n}" and e.get("exit_code") == 0]
    state["proxy_seconds"] += durations[-1] if durations else 0.0
    salvage.save(state)


def screen_eval(state: dict, cid: str) -> dict:
    if cid in state.setdefault("screen_evaluations", {}):
        return state["screen_evaluations"][cid]
    candidate = state["candidates"][cid]
    if sha256(candidate["checkpoint"]) != candidate["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint SHA changed before screening: {cid}")
    root = OUT / "candidates" / cid / "final_eval"
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "data_d.json"
    if not data_path.exists():
        atomic_json(data_path, salvage.official_validation(candidate["checkpoint"]))
    wiki_path = root / "wikitext.json"
    if not wiki_path.exists():
        tmp = root / ".wikitext.json.tmp"
        sec = salvage.run(state, cid+"-wiki", [salvage.PY, "scripts/evaluate_wikitext103.py", "--checkpoint",
            candidate["checkpoint"], "--tokenizer", salvage.TOKENIZER, "--output", tmp,
            "--threads", 16, "--batch-size", 16])
        wiki = load_json(tmp)
        if (wiki.get("checkpoint_sha256") != candidate["checkpoint_sha256"] or
                not isinstance(wiki.get("bits_per_byte"), (int,float))):
            raise RuntimeError(f"invalid WikiText result: {cid}")
        os.replace(tmp, wiki_path)
        state.setdefault("eval_timings", {}).setdefault(cid, {})["wikitext"] = sec
    data = load_json(data_path)
    wiki = load_json(wiki_path)
    if data.get("protocol") is None or not isinstance(wiki.get("perplexity"), (int,float)):
        raise RuntimeError(f"invalid retention screening artifacts: {cid}")
    ensure_proxy(state, cid, 1024)
    proxy = candidate["proxy_1024"]
    result = {
        "candidate_id": cid,
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "lineage": candidate["lineage"],
        "alpha": candidate.get("alpha"),
        "data_d": data,
        "wikitext_ppl": wiki["perplexity"],
        "wikitext_bpb": wiki["bits_per_byte"],
        "proxy_1024": {"accuracy": proxy["v2_ranking_accuracy"],
                        "mean_margin": proxy["v2_mean_margin"],
                        "data_d_fixed_slice_loss": proxy["data_d_validation"]},
    }
    state["screen_evaluations"][cid] = result
    salvage.save(state)
    return result


def complete_gibc(state: dict, cid: str) -> dict:
    if cid in state["final_evaluations"]:
        return state["final_evaluations"][cid]
    candidate = state["candidates"][cid]
    screen = state["screen_evaluations"][cid]
    root = OUT / "candidates" / cid / "final_eval"
    gibc_path = root / "gibc.json"
    if not gibc_path.exists():
        tmp = root / ".gibc.json.tmp"
        sec = salvage.run(state, cid+"-gibc", [salvage.PY, "scripts/evaluate_gibc.py", "--checkpoint",
            candidate["checkpoint"], "--tokenizer", salvage.TOKENIZER, "--output", tmp,
            "--tasks", ",".join(salvage.TASKS), "--threads", 16, "--batch-size", 16])
        raw = load_json(tmp)
        if raw.get("phase6_metadata", {}).get("checkpoint_sha256") != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"GIBC checkpoint identity mismatch: {cid}")
        for task in salvage.TASKS:
            value = raw.get("results", {}).get(task, {}).get("acc,none")
            if not isinstance(value, (int,float)):
                raise RuntimeError(f"GIBC metric missing for {cid}/{task}")
        os.replace(tmp, gibc_path)
        state.setdefault("eval_timings", {}).setdefault(cid, {})["gibc"] = sec
    raw = load_json(gibc_path)
    result = {
        "candidate_id": cid,
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "architecture": salvage.validate_model(candidate["checkpoint"]),
        "lineage": candidate["lineage"],
        "recovery_tokens": candidate.get("recovery_tokens", 0),
        "interpolation_alpha": candidate.get("alpha"),
        "tokenizer_sha256": state["tokenizer_sha256"],
        "dataset_manifest_sha256": state["manifest_sha256"],
        "code_commit_sha": state["code_commit_sha"],
        "data_d": screen["data_d"],
        "wikitext_ppl": screen["wikitext_ppl"],
        "wikitext_bpb": screen["wikitext_bpb"],
        "gibc_raw": {task: raw["results"][task]["acc,none"] for task in salvage.TASKS},
        "gibc_normalized": {task: raw["results"][task].get("acc_norm,none") for task in salvage.TASKS},
        "proxy_1024": screen["proxy_1024"],
        "raw_outputs": {"data_d": str(root/"data_d.json"), "wikitext": str(root/"wikitext.json"),
                        "gibc": str(gibc_path)},
    }
    state["final_evaluations"][cid] = result
    state["official_seconds"] += state.get("eval_timings", {}).get(cid, {}).get("gibc", 0.0)
    salvage.save(state)
    return result


def run_all(plan: dict, state: dict) -> None:
    state["phase"] = "EXECUTING"
    append_event(state, "execution_start", candidate_count=len(plan["experiments"]))
    for exp in plan["experiments"]:
        cid, endpoint, alpha = exp["candidate_id"], exp["endpoint"], exp["alpha"]
        if cid in state.get("screen_evaluations", {}) or cid in state["final_evaluations"]:
            continue
        if salvage.remaining(state) < 60:
            append_event(state, "deadline_stop", candidate_id=cid, remaining_seconds=salvage.remaining(state))
            state["phase"] = "DEADLINE_STOPPED"
            salvage.save(state)
            report(state, plan)
            return
        append_event(state, "candidate_start", candidate_id=cid, endpoint=endpoint, alpha=alpha)
        try:
            if cid not in state["candidates"]:
                salvage.interpolate(state, endpoint, alpha)
            ensure_proxy256(state, cid)
            result = screen_eval(state, cid)
            for failure in state.get("failures", []):
                if failure.get("candidate_id") == cid and not failure.get("resolved_at"):
                    failure["resolved_at"] = now_iso()
            append_event(state, "candidate_screen_complete", candidate_id=cid,
                         data_d_loss=result["data_d"]["loss"], wikitext_bpb=result["wikitext_bpb"],
                         proxy_1024=result["proxy_1024"]["accuracy"])
        except Exception as exc:
            failure = {"at": now_iso(), "candidate_id": cid, "error": repr(exc)}
            state.setdefault("failures", []).append(failure)
            append_event(state, "candidate_failure", **failure)
            report(state, plan)
            raise
        report(state, plan)
    # Full GIBC is deliberately limited to one additional candidate after
    # screening; the 5% candidate already has a complete four-task result.
    base = state["final_evaluations"]["BASE"]
    base_d, base_bpb = base["data_d"]["loss"], base["wikitext_bpb"]
    eligible = []
    for exp in plan["experiments"]:
        cid = exp["candidate_id"]
        screen = state.get("screen_evaluations", {}).get(cid)
        if screen is None or cid in state["final_evaluations"]:
            continue
        if (screen["data_d"]["loss"] <= base_d*1.20 and
                screen["wikitext_bpb"] <= base_bpb*1.25):
            eligible.append((screen["proxy_1024"]["accuracy"],
                             screen["proxy_1024"]["mean_margin"], cid))
    eligible.sort(reverse=True)
    if eligible and salvage.remaining(state) > 900:
        _, _, selected = eligible[0]
        append_event(state, "gibc_candidate_selected", candidate_id=selected,
                     reason="highest reasoning proxy among candidates passing Data D and WikiText gates",
                     additional_full_gibc_budget=1)
        try:
            result = complete_gibc(state, selected)
            append_event(state, "candidate_gibc_complete", candidate_id=selected,
                         raw_gibc_mean=sum(result["gibc_raw"].values())/4)
        except Exception as exc:
            failure = {"at": now_iso(), "candidate_id": selected, "error": repr(exc)}
            state.setdefault("failures", []).append(failure)
            append_event(state, "candidate_failure", **failure)
            report(state, plan)
            raise
    state["phase"] = "COMPLETE"
    state["finished_at"] = now_iso()
    append_event(state, "execution_complete", evaluated=len(state["final_evaluations"]) - 1)
    report(state, plan)


def heartbeat_loop(stop_event: threading.Event) -> None:
    path = OUT / "controller.heartbeat.json"
    private_path = OUT / f".controller.heartbeat.thread-{os.getpid()}.json"
    while not stop_event.wait(5):
        child = getattr(salvage, "CHILD", None)
        payload = {"pid": os.getpid(), "at": now_iso(), "phase": "EXECUTING",
                   "child_pid": child.pid if child is not None and child.poll() is None else None}
        # The shared runner also updates controller.heartbeat.json directly;
        # use a private atomic temp path to avoid colliding with its temp file.
        atomic_json(private_path, payload)
        os.replace(private_path, path)
    private_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan, parent = write_plan_copy()
    # Redirect the imported salvage helpers before any save() call. Setting
    # these after init_or_resume would write follow-up state into the parent.
    salvage.OUT = OUT
    salvage.STATE = STATE_PATH
    lock_path = OUT / "controller.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another follow-up controller holds the lock") from exc
        state = init_or_resume(plan, parent)
        salvage.require_base(state)
        verify_inputs(plan, state)
        report(state, plan)
        if args.preflight_only:
            print(json.dumps({"status": "READY", "candidates": len(plan["experiments"]),
                              "remaining_seconds": round(salvage.remaining(state)),
                              "output": str(OUT)}, indent=2))
            return
        if state.get("phase") == "COMPLETE":
            print("Follow-up already complete.")
            return
        if state.get("phase") not in ("READY", "EXECUTING", "DEADLINE_STOPPED"):
            raise RuntimeError(f"cannot resume follow-up phase {state.get('phase')}")
        state["phase"] = "EXECUTING"
        pid_path = OUT / "controller.pid"
        pid_path.write_text(f"{os.getpid()}\n")
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(target=heartbeat_loop, args=(heartbeat_stop,), daemon=True)
        heartbeat_thread.start()
        signal.signal(signal.SIGINT, salvage.stop)
        signal.signal(signal.SIGTERM, salvage.stop)
        try:
            run_all(plan, state)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=10)
            pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
