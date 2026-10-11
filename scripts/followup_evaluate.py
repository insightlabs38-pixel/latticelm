"""Common endpoint evaluation, with atomic publication and checkpoint identity."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import signal
import time
from pathlib import Path
from followup_common import ROOT, RESEARCH, atomic_json, registered, sha256, utc, verified_checkpoint,object_hash


def evaluate(name):
    record = registered(name); run = RESEARCH / "runs" / name
    (run/'failure.json').unlink(missing_ok=True)
    if sha256(ROOT/record['tokenizer_path'])!=record['tokenizer_sha256']:raise ValueError('Evaluation tokenizer identity mismatch')
    for rel,digest in record.get('source_hashes',{}).items():
        if sha256(ROOT/rel)!=digest:raise RuntimeError('Evaluation scientific source changed: '+rel)
    item = verified_checkpoint(run)
    if not item or item["logical_tokens"] != record["token_budget"]: raise RuntimeError("endpoint checkpoint is missing")
    checkpoint = run / item["file"]; tokenizer = ROOT / record["tokenizer_path"]
    outputs = []; started=time.perf_counter(); child=None
    def stopped(*_):
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        atomic_json(run/'failure.json',{'classification':'INTERRUPTED_INFRA','at':utc(),'message':'Evaluator stopped by signal'})
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    mode=record.get('evaluation_mode','full')
    pipeline=(
        ("evaluate_wikitext103.py", "wikitext103.json", []),
        ("evaluate_gibc.py", "capability_full.json" if mode=='full' else "capability_screen.json",
         ["--tasks", record.get("evaluation_tasks", "hellaswag,arc_easy,piqa,winogrande")]+([] if mode=='full' else ['--limit','1000'])))
    if mode=='validation_only':pipeline=()
    for script, filename, extra in pipeline:
        output = run / filename
        if output.exists():
            data = json.loads(output.read_text())
            identity = data.get("checkpoint_sha256", data.get("phase6_metadata", {}).get("checkpoint_sha256"))
            if identity != item["sha256"]: raise RuntimeError("stale evaluation " + filename)
        else:
            temporary = run / ("." + filename + ".tmp")
            recovered=False
            if temporary.exists():
                try:
                    candidate=json.loads(temporary.read_text())
                    identity=candidate.get('checkpoint_sha256',candidate.get('phase6_metadata',{}).get('checkpoint_sha256'))
                    recovered=identity==item['sha256'] and (script!='evaluate_gibc.py' or (run/'.capability_samples.json.tmp').exists())
                except json.JSONDecodeError:pass
            command = [sys.executable, str(ROOT / "scripts" / script), "--checkpoint", str(checkpoint),
                       "--tokenizer", str(tokenizer), "--output", str(temporary), "--threads", "16"] + extra
            if script == "evaluate_gibc.py": command += ["--predictions-output", str(run / ".capability_samples.json.tmp")]
            if not recovered:
                with (run / "evaluate.log").open("a") as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    if child.wait()!=0:raise RuntimeError('Evaluation child failed: '+script)
                child=None
            with temporary.open("rb") as handle: os.fsync(handle.fileno())
            os.replace(temporary, output)
            if script == "evaluate_gibc.py": os.replace(run / ".capability_samples.json.tmp", run / "capability_samples.json")
        outputs.append({"file": filename, "sha256": sha256(output), "size_bytes": output.stat().st_size})
    if mode!='validation_only':
        samples=run/'capability_samples.json'
        if not samples.exists():raise RuntimeError('Capability output lacks paired per-item evidence')
        outputs.append({'file':samples.name,'sha256':sha256(samples),'size_bytes':samples.stat().st_size})
    validation = json.loads((run / "latest_validation.json").read_text())
    if validation["checkpoint_sha256"] != item["sha256"]: raise RuntimeError("endpoint source evaluation mismatch")
    receipt = {"at": utc(), "status": "COMPLETED", "checkpoint_sha256": item["sha256"],
               'registration_sha256':object_hash(record),
               "tokenizer_sha256": record["tokenizer_sha256"], "outputs": outputs,
               "capability_scope": {'full':'full official task evaluation','screen':'1000-example intermediate screen','validation_only':'DATA-D validation only; no capability result'}[mode],
               "evaluation_mode":mode,"wall_seconds":time.perf_counter()-started,"source_losses": validation["source_losses"],
               "frozen_mixture_weighted_loss": validation["frozen_mixture_weighted_loss"]}
    atomic_json(run / "evaluation_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--experiment-id", required=True)
    name=p.parse_args().experiment_id
    try:print(json.dumps(evaluate(name)))
    except Exception as error:
        atomic_json(RESEARCH/'runs'/name/'failure.json',{'at':utc(),'classification':'FAILED_DATA' if isinstance(error,ValueError) else 'FAILED_CODE','message':str(error)})
        raise
