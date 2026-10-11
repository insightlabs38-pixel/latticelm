"""Final-week controller: one trainer, durable identities, exact-resume recovery."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from followup_common import (ROOT, RESEARCH, STATE, alive, atomic_bytes, atomic_json, complete_evidence,
                             decision, failure_class, init_state, process_identity, projected_seconds,
                             registrations, sha256, utc, verified_checkpoint)
from followup_common import object_hash,reconcile_rolling_manifest

STOP = False


def valid_certificate(record):
    path = RESEARCH / record["launch_certificate"]
    if not path.exists(): return False
    certificate = json.loads(path.read_text())
    return (certificate.get("passed") is True
            and certificate.get("registration_sha256") == object_hash(record)
            and certificate.get("source_hashes") == record.get("source_hashes", {}))


def disappeared_classification(active,record,return_code,failure):
    if failure:return failure_class(return_code,failure)
    if active['phase']=='RUNNING':
        run=RESEARCH/'runs'/record['experiment_id'];receipt=run/'train_receipt.json'
        if receipt.exists():
            data=json.loads(receipt.read_text())
            if data.get('status')=='INTERRUPTED_INFRA':
                cp=verified_checkpoint(run)
                if cp and cp['sha256']==data['checkpoint_sha256'] and cp['logical_tokens']==data['logical_tokens']:
                    return 'INTERRUPTED_INFRA'
    return failure_class(return_code,failure)


def scarce_time(records,state,now):
    """Switch to prompt section36 when unfinished core work threatens reserve."""
    terminal={'COMPLETED','ARCHIVED','REJECTED','FAILED_CODE','FAILED_DATA','FAILED_SCIENCE'}
    core_seconds=0.;full_rates=[];meaningful_full_complete=False;full_remaining=[]
    for r in records:
        if r.get('smoke'):continue
        name=r['experiment_id'];info=state['runs'].get(name,{})
        status=info.get('status','PLANNED');full=r.get('full_co4')
        modulated=bool(full and 'modulated' in full['block_types'])
        if modulated and info.get('measured_tps'):full_rates.append(info['measured_tps'])
        if modulated and r['token_budget']>=50_000_000:
            if status in ('COMPLETED','ARCHIVED'):meaningful_full_complete=True
            elif status not in terminal:
                left=max(0,r['token_budget']-info.get('saved_tokens',r.get('initial_logical_tokens',0)))
                full_remaining.append(projected_seconds(left,info.get('measured_tps',r['planning_tps']),evaluation_seconds=r.get('evaluation_seconds',3600)))
        if name in ('F32-DENSE-S1337-FRESH300M','F16-D4-S2026-FRESH150M','F8-MOD-S7331-FRESH100M','F8-DIFF-S7331-FRESH100M') and status not in terminal:
            left=max(0,r['token_budget']-info.get('saved_tokens',0))
            core_seconds+=projected_seconds(left,info.get('measured_tps',r['planning_tps']),evaluation_seconds=r.get('evaluation_seconds',1800))
    if meaningful_full_complete:needed_full=0.
    else:needed_full=min(full_remaining) if full_remaining else projected_seconds(50_000_000,max(full_rates,default=800),evaluation_seconds=3600)
    return state['scientific_stop_epoch']-now < core_seconds+needed_full


def choose(records, state, only=None, now=None):
    now = time.time() if now is None else now
    scarce=scarce_time(records,state,now) if not only else False
    def priority(record):
        n=record['experiment_id']
        if scarce:
            if n.startswith(('F32-','CONTROL-R32')):return 0
            if n.startswith(('F8-MOD-S7331','F8-DIFF-S7331')):return 2
            if n.startswith(('F16-D4','CONTROL-R16')):return 4
            if '16M' in n or 'DEPTH' in n or 'EXT200M' in n or 'DIFFERENTIAL' in n:return 5
            if n.startswith('FC') or 'FULLCONTROL' in n:return 3 if 'S2026' in n else 1
        if n.startswith(('F32-','CONTROL-R32')):return 0
        if n.startswith(('F16-D4','CONTROL-R16')):return 1
        if n.startswith(('F8-MOD-S7331','F8-DIFF-S7331')):return 2
        if n.startswith('CONTROL-A8'):return 5
        if 'EXT200M' in n:return 6
        if 'DEPTH' in n:return 7
        if '16M' in n:return 8
        if 'DIFFERENTIAL' in n:return 9
        return 3
    for record in sorted(records,key=priority):
        name = record["experiment_id"]
        if only and name != only: continue
        if not only and record.get("smoke", False): continue
        status = state["runs"].get(name, {}).get("status", "PLANNED")
        if status in ("COMPLETED", "ARCHIVED", "REJECTED", "FAILED_CODE", "FAILED_DATA", "FAILED_SCIENCE"): continue
        if not record.get("ready", False): continue
        gate = record.get("gate_file")
        if gate:
            path = RESEARCH / gate
            if not path.exists(): continue
            result = json.loads(path.read_text())
            if not result.get("passed", result.get("run", False)): continue
        run = RESEARCH / "runs" / name
        if complete_evidence(run, record, require_evaluation=False):
            if now + record.get('evaluation_seconds',900) < state.get('hard_deadline_epoch',state['scientific_stop_epoch'])-600:
                return record, "EVALUATING"
            continue
        if record.get('historical_experiment'):continue  # imports cannot launch training
        if not record.get("smoke", False) and not valid_certificate(record):
            queue_path = RESEARCH / "FOLLOWUP_PREFLIGHT_QUEUE.json"
            queued = json.loads(queue_path.read_text()).get("experiment_ids",[]) if queue_path.exists() else []
            if record.get("auto_preflight",False) or name in queued:
                seconds = 300 + 3*100*record["config"]["batch_size"]*record["config"]["context_length"]/record["planning_tps"]
                if now + seconds <= state["scientific_stop_epoch"]: return record,"VERIFYING"
            continue
        cp = verified_checkpoint(run)
        remaining = record["token_budget"] - (cp["logical_tokens"] if cp else record.get("initial_logical_tokens",0))
        measured = state["runs"].get(name, {}).get("measured_tps", record["planning_tps"])
        estimated = projected_seconds(remaining, measured, evaluation_seconds=record.get("evaluation_seconds", 900))
        if now + estimated <= state["scientific_stop_epoch"]: return record, "RUNNING"
    return None, None


def schedule_report(state, records):
    lines = ["# Follow-up live schedule", "", "Post-submission research. One heavy trainer at a time.", "",
             "Updated: " + utc(), "Hard deadline: " + state["hard_deadline_utc"],
             "Scientific stop: " + utc(state["scientific_stop_epoch"]), "",
             "| Experiment | Status | Saved tokens | Remaining projected hours |", "|---|---|---:|---:|"]
    for r in records:
        info = state["runs"].get(r["experiment_id"], {})
        saved = info.get("saved_tokens", 0); rate = info.get("measured_tps", r["planning_tps"])
        hours = projected_seconds(max(0, r["token_budget"]-saved), rate) / 3600
        lines.append(f"| {r['experiment_id']} | {info.get('status', 'READY' if r.get('ready') else 'IMPLEMENTING')} | {saved:,} | {hours:.2f} |")
    atomic_bytes(RESEARCH / "FOLLOWUP_LIVE_SCHEDULE.md", ("\n".join(lines) + "\n").encode())


def preemption_notice():
    try:
        req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/preempted", headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(req, timeout=1) as response: return response.read().strip() == b"TRUE"
    except Exception: return False  # periodic checkpoints remain authoritative


def launch(record, phase, state):
    name = record["experiment_id"]; run = RESEARCH / "runs" / name; run.mkdir(parents=True, exist_ok=True)
    if phase=='RUNNING':reconcile_rolling_manifest(run)
    script = {"RUNNING":"followup_train.py","EVALUATING":"followup_evaluate.py","VERIFYING":"followup_verify_variant.py"}[phase]
    if phase=='RUNNING':script=record.get('trainer_script',script)
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--experiment-id", name]
    if phase == "RUNNING": cmd += ["--stop-epoch", str(state["scientific_stop_epoch"])]
    with (run / {"RUNNING":"train.log","EVALUATING":"evaluate.log","VERIFYING":"preflight.log"}[phase]).open("a") as log:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(process.pid)
    if not identity: raise RuntimeError("child disappeared before identity capture")
    active = {**identity, "experiment_id": name, "phase": phase, "launch_utc": utc(),
              "working_directory": str(ROOT), "expected_checkpoint_path": str(run / "checkpoint_manifest.json")}
    state["active"] = active; state["runs"].setdefault(name, {})["status"] = phase
    atomic_json(STATE, state); decision("CHILD_LAUNCHED", **active)
    return process


def recover_unrecorded_children(state, records):
    if state.get("active") and alive(state["active"], state["active"]["experiment_id"]): return
    candidates = []
    # Recover a child launched just before a master crash, before state publication.
    for r in records:
        hb = RESEARCH / "runs" / r["experiment_id"] / "heartbeat.json"
        if hb.exists():
            data = json.loads(hb.read_text())
            if alive(data, r["experiment_id"]):
                ready = hb.parent / "test_checkpoint_ready.json"
                marker = hb.parent / "verified_interruption.json"
                if r.get("verification_parent") and ready.exists() and not marker.exists() and data.get("step") == json.loads(ready.read_text())["step"]:
                    os.kill(data["pid"],signal.SIGKILL)
                    atomic_json(marker,{"at":utc(),"kill_step":data["step"],"classification":"INTERRUPTED_INFRA","orphaned_smoke_rendezvous":True})
                    decision("ORPHANED_TEST_RENDEZVOUS_RECOVERED",experiment_id=r["experiment_id"])
                    continue
                candidates.append({**data,'experiment_id':r['experiment_id'], "phase": "EVALUATING" if data.get("phase") == "EVALUATING" else "RUNNING"})
    # Close the launch->heartbeat window using actual command identity.
    known = {r["experiment_id"] for r in records}
    candidate_pids = {c["pid"] for c in candidates}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in candidate_pids: continue
        try: cmd = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        if b"--experiment-id" not in cmd:
            if b"--checkpoint" in cmd and any(p.endswith((b"/evaluate_gibc.py",b"/evaluate_wikitext103.py")) for p in cmd):
                try:
                    checkpoint=Path(cmd[cmd.index(b"--checkpoint")+1].decode())
                    relative=checkpoint.relative_to(RESEARCH/'runs');name=relative.parts[0]
                except (ValueError,IndexError,UnicodeDecodeError):continue
                identity=process_identity(int(proc.name))
                if name in known and identity and identity['process_state']!='Z':
                    candidates.append({**identity,'experiment_id':name,'phase':'EVALUATING','evaluation_worker':True})
            continue
        scripts = [part for part in cmd if part.endswith((b"/followup_train.py", b"/followup_continuation_train.py",b"/followup_evaluate.py",b"/followup_verify_variant.py"))]
        if not scripts: continue
        try: name = cmd[cmd.index(b"--experiment-id") + 1].decode()
        except (IndexError, UnicodeDecodeError): continue
        identity = process_identity(int(proc.name))
        if name in known and identity and identity["process_state"] != "Z":
            phase = "VERIFYING" if scripts[0].endswith(b"followup_verify_variant.py") else "EVALUATING" if scripts[0].endswith(b"followup_evaluate.py") else "RUNNING"
            candidates.append({**identity, "experiment_id": name, "phase": phase})
    if candidates:
        # A verifier owns its smoke child; adopt the verifier rather than count
        # that pair as two scientific trainers.
        verifiers=[c for c in candidates if c["phase"]=="VERIFYING"]
        if len(verifiers)==1:
            parent=verifiers[0]["experiment_id"]
            if all(c in verifiers or next(r for r in records if r["experiment_id"]==c["experiment_id"]).get("verification_parent")==parent for c in candidates):
                candidates=verifiers
        evaluators=[c for c in candidates if c['phase']=='EVALUATING' and not c.get('evaluation_worker')]
        if len(evaluators)==1 and all(c['experiment_id']==evaluators[0]['experiment_id'] and c['phase']=='EVALUATING' for c in candidates):
            candidates=evaluators
        if len(candidates) > 1: raise RuntimeError("multiple active trainers discovered; refusing further launch")
        state["active"] = candidates[0]; atomic_json(STATE, state)
        decision("ORPHAN_TRAINER_ADOPTED", experiment_id=candidates[0]["experiment_id"])


def master(args):
    state = init_state(args.hard_deadline_utc); process = None
    last_report = 0.; last_scientific_report = 0.; notice_seen = False
    archive_process = None
    if state.get('finalized'):return
    if state.get('active') and alive(state['active'],state['active']['experiment_id']):
        decision('MASTER_RECOVERED_LIVE_CHILD',master_pid=os.getpid(),child_pid=state['active']['pid'],experiment_id=state['active']['experiment_id'])
    while True:
        records = registrations(); recover_unrecorded_children(state, records)
        if not args.only:
            archive_identity = state.get("archive_active")
            if archive_identity and not alive(archive_identity,"HF-DELTA"):
                rc = archive_process.poll() if archive_process and archive_process.pid == archive_identity["pid"] else None
                decision("DELTA_ARCHIVE_PROCESS_FINISHED",return_code=rc)
                state["archive_active"] = None
            if not state.get("archive_active") and time.time() - state.get("last_archive_launch_epoch",0) > 1800:
                with (RESEARCH/"archive.log").open("a") as log:
                    archive_process = subprocess.Popen([sys.executable,str(ROOT/"scripts/followup_archive.py"),
                        "--experiment-id","HF-DELTA"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state["archive_active"] = process_identity(archive_process.pid)
                state["last_archive_launch_epoch"] = time.time(); atomic_json(STATE,state)
        active = state.get("active")
        if active:
            name = active["experiment_id"]; record = next(r for r in records if r["experiment_id"] == name)
            run = RESEARCH / "runs" / name
            if alive(active, name):
                hb_path = run / "heartbeat.json"
                if active["phase"] == "RUNNING" and hb_path.exists():
                    hb = json.loads(hb_path.read_text()); age = time.time() - hb["heartbeat_epoch"]
                    info = state["runs"].setdefault(name, {}); info["heartbeat_age_seconds"] = age
                    # Never clear a lock solely because the heartbeat is stale.
                    info["live_tokens"] = hb["logical_tokens"]
                    mp=run/'checkpoint_manifest.json'
                    if mp.exists():
                        saved=json.loads(mp.read_text())['rolling'][0];info['saved_tokens']=saved['logical_tokens']
                    metrics=sorted(run.glob('metrics-*.jsonl'),key=lambda p:p.stat().st_mtime)
                    if metrics:
                        with metrics[-1].open('rb') as f:
                            f.seek(max(0,metrics[-1].stat().st_size-32768));lines=f.read().splitlines()
                        if lines:
                            try:info['measured_tps']=json.loads(lines[-1])['tokens_per_second']
                            except (json.JSONDecodeError,KeyError):pass
                if not notice_seen and preemption_notice():
                    notice_seen = True; decision("PREEMPTION_NOTICE", experiment_id=name)
                    os.kill(active["pid"], signal.SIGUSR1 if active["phase"] == "RUNNING" else signal.SIGTERM)
                active_stop = state["hard_deadline_epoch"] - 600 if active["phase"] == "EVALUATING" else state["scientific_stop_epoch"]
                if STOP or time.time() >= active_stop:
                    os.kill(active["pid"], signal.SIGTERM)
                    if STOP: decision("MASTER_STOP_REQUESTED", experiment_id=name)
                time.sleep(args.poll_seconds)
            else:
                rc = process.poll() if process and process.pid == active["pid"] else None
                failure_path = run / "failure.json"
                failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
                info = state["runs"].setdefault(name, {})
                if active["phase"] == "VERIFYING" and valid_certificate(record):
                    info["status"] = "READY"
                    cert = json.loads((RESEARCH/record["launch_certificate"]).read_text())
                    info["measured_tps"] = cert["smoke_tokens_per_second"]
                    decision("AUTOMATIC_PREFLIGHT_PASSED",experiment_id=name,certificate=record["launch_certificate"])
                elif active["phase"] == "RUNNING" and complete_evidence(run, record, require_evaluation=False):
                    info["status"] = "EVALUATING"
                    receipt = json.loads((run / "train_receipt.json").read_text())
                    info["measured_tps"] = receipt["logical_tokens"] / max(receipt["wall_seconds"], 1e-9)
                    info["saved_tokens"] = receipt["logical_tokens"]
                    decision("TRAINING_ENDPOINT_VERIFIED", experiment_id=name, receipt=receipt)
                    if args.exit_after_trained:
                        state["active"] = None; atomic_json(STATE, state); schedule_report(state, records); return
                elif active["phase"] == "EVALUATING" and complete_evidence(run, record):
                    info["status"] = "COMPLETED"; decision("ENDPOINT_COMPLETED", experiment_id=name)
                else:
                    classification = disappeared_classification(active,record,rc,failure)
                    info["status"] = classification
                    info["infrastructure_interruptions"] = info.get("infrastructure_interruptions", 0) + int(classification == "INTERRUPTED_INFRA")
                    if classification == "INTERRUPTED_INFRA":
                        cp = verified_checkpoint(run); info["saved_tokens"] = cp["logical_tokens"] if cp else 0
                        if cp and cp["wall_seconds"] > 0: info["measured_tps"] = cp["logical_tokens"] / cp["wall_seconds"]
                    else:
                        info["scientific_or_code_failures"] = info.get("scientific_or_code_failures", 0) + 1
                    decision("CHILD_DISAPPEARED", experiment_id=name, classification=classification, return_code=rc, failure=failure)
                state["active"] = None; process = None; atomic_json(STATE, state)
                if info["status"] in ("FAILED_CODE", "FAILED_DATA"):
                    state["campaign_blocker"]={"experiment_id":name,"classification":info["status"],"at":utc()}
                    atomic_json(STATE,state)
                    raise RuntimeError(f"{name}: {info['status']}; inspect failure receipt before retry")
        else:
            if STOP: return
            if state.get("campaign_blocker"):
                if time.time()>=state['scientific_stop_epoch']:
                    from followup_finalize import finalize
                    finalize(state);return
                time.sleep(args.poll_seconds)
                state["heartbeat_utc"]=utc();atomic_json(STATE,state)
                schedule_report(state,records)
                continue
            if not args.only and time.time()<state['scientific_stop_epoch']:
                from followup_funnel import advance
                try:advance(state);records=registrations()
                except Exception as error:
                    state['campaign_blocker']={'classification':'FAILED_CODE','stage':'conditional registration','message':str(error),'at':utc()}
                    atomic_json(STATE,state);decision('CAMPAIGN_BLOCKED',**state['campaign_blocker']);raise
            record, phase = choose(records, state, args.only)
            if record:
                if phase == "RUNNING" and not record.get("smoke", False):
                    if not valid_certificate(record):
                        raise RuntimeError("long training lacks passing launch certificate")
                process = launch(record, phase, state)
            else:
                if args.only: return
                if time.time() >= state["scientific_stop_epoch"]:
                    decision("SCIENTIFIC_STOP", reason="deadline reserve reached")
                    if state.get('archive_active') and alive(state['archive_active'],'HF-DELTA'):
                        time.sleep(args.poll_seconds);continue
                    from followup_finalize import finalize
                    finalize(state);return
                # Future variants may be registered while a long run executes.
                time.sleep(args.poll_seconds)
        state["heartbeat_utc"] = utc(); atomic_json(STATE, state)
        if time.monotonic()-last_report > 30:
            schedule_report(state, records); last_report = time.monotonic()
        if not args.only and time.monotonic()-last_scientific_report>1800:
            from followup_finalize import reports
            reports(state);last_scientific_report=time.monotonic()


def main():
    global STOP
    p = argparse.ArgumentParser(); p.add_argument("--hard-deadline-utc"); p.add_argument("--only")
    p.add_argument("--exit-after-trained", action="store_true"); p.add_argument("--poll-seconds", type=float, default=5)
    args = p.parse_args(); RESEARCH.mkdir(exist_ok=True)
    def stop(*_):
        global STOP; STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with (RESEARCH / ".followup-master.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("duplicate follow-up master refused")
        master(args)


if __name__ == "__main__": main()
