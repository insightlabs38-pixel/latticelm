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
from followup_common import ROOT, RESEARCH, atomic_json, registered, sha256, utc, verified_checkpoint,object_hash,sync_dir


def publish_outputs(run,temporary,output,capability=False):
    """Predictions precede the result commit point; tolerate either crash edge."""
    if capability:
        staging=run/'.capability_samples.json.tmp';samples=run/'capability_samples.json'
        source=staging if staging.exists() else samples
        json.loads(source.read_text())
        with source.open('rb') as handle:os.fsync(handle.fileno())
        if source==staging:os.replace(staging,samples)
    with temporary.open('rb') as handle:os.fsync(handle.fileno())
    os.replace(temporary,output);sync_dir(run)


def output_matches(data,item,record):
    metadata=data.get('phase6_metadata',data)
    return metadata.get('checkpoint_sha256')==item['sha256'] and metadata.get('tokenizer_sha256')==record['tokenizer_sha256']


def validate_samples(run,record,result):
    samples=json.loads((run/'capability_samples.json').read_text())
    for task in record.get('evaluation_tasks','hellaswag,arc_easy,piqa,winogrande').split(','):
        rows=samples.get(task)
        if not rows:raise ValueError('Missing paired task samples: '+task)
        keys=[(str(r['doc_id']),r['doc_hash'],r['prompt_hash'],r['target_hash']) for r in rows]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate task samples: '+task)
        expected=result.get('n-samples',{}).get(task,{}).get('effective')
        if expected is not None and len(rows)!=expected:raise ValueError('Task result/sample count mismatch: '+task)


def ensure_source_validation(record,run,item):
    """Recover a crash after final checkpoint commit but before validation JSON."""
    path=run/'latest_validation.json'
    previous=json.loads(path.read_text()) if path.exists() else None
    if previous and previous.get('checkpoint_sha256')==item['sha256']:return previous
    if sha256(ROOT/record['manifest_path'])!=record['sampling_manifest_sha256']:
        raise ValueError('Source validation recovery manifest mismatch')
    import torch
    from followup_train import build_model,evaluate_sources
    from train_final_production import load_arrays
    torch.set_num_threads(record.get('threads',16))
    payload=torch.load(run/item['file'],map_location='cpu',weights_only=False)
    model=build_model(record);model.load_state_dict(payload['model'],strict=True)
    top,_,valid=load_arrays(ROOT/record['manifest_path'])
    losses,paired=evaluate_sources(model,valid,record['config']['context_length'])
    recovered={'at':utc(),'step':item['step'],'logical_tokens':item['logical_tokens'],
        'source_losses':losses,'paired_losses':paired,'checkpoint_sha256':item['sha256'],
        'frozen_mixture_weighted_loss':sum(top['mixture_definition'][s]*losses[s] for s in losses),
        'recovered_after_checkpoint_publication':True}
    atomic_json(run/'SOURCE_VALIDATION_RECOVERY.json',{'at':utc(),'previous_checkpoint_sha256':None if not previous else previous.get('checkpoint_sha256'),
        'recovered_checkpoint_sha256':item['sha256'],'reason':'Checkpoint committed before matching source-validation JSON survived.'})
    atomic_json(path,recovered)
    return recovered


def evaluate(name):
    record = registered(name); run = RESEARCH / "runs" / name
    (run/'failure.json').unlink(missing_ok=True)
    if sha256(ROOT/record['tokenizer_path'])!=record['tokenizer_sha256']:raise ValueError('Evaluation tokenizer identity mismatch')
    for rel,digest in record.get('source_hashes',{}).items():
        if sha256(ROOT/rel)!=digest:raise RuntimeError('Evaluation scientific source changed: '+rel)
    item = verified_checkpoint(run)
    if not item or item["logical_tokens"] != record["token_budget"]: raise RuntimeError("endpoint checkpoint is missing")
    checkpoint = run / item["file"]; tokenizer = ROOT / record["tokenizer_path"]
    mode=record.get('evaluation_mode','full')
    protocol={'schema':'followup-evaluation-protocol-v1','checkpoint_sha256':item['sha256'],
        'tokenizer_sha256':record['tokenizer_sha256'],'evaluation_mode':mode,
        'tasks':record.get('evaluation_tasks','hellaswag,arc_easy,piqa,winogrande'),
        'source_hashes':{p:sha256(ROOT/p) for p in ('scripts/followup_evaluate.py','scripts/evaluate_gibc.py',
                                                'scripts/evaluate_wikitext103.py','src/latticelm/tokenizer.py')}}
    protocol_path=run/'evaluation_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Evaluation protocol changed across attempts; inspect before reusing outputs')
    atomic_json(protocol_path,protocol)
    if record.get('full_co4'):
        import torch
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        diagnostics=payload.get('architecture_state')
        if not diagnostics:raise ValueError('Full-Co4 checkpoint lacks architecture diagnostics')
        atomic_json(run/'architecture_diagnostics.json',diagnostics)
        del payload
    outputs = []; started=time.perf_counter(); child=None
    def stopped(*_):
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        atomic_json(run/'failure.json',{'classification':'INTERRUPTED_INFRA','at':utc(),'message':'Evaluator stopped by signal'})
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    pipeline=(
        ("evaluate_wikitext103.py", "wikitext103.json", []),
        ("evaluate_gibc.py", "capability_full.json" if mode=='full' else "capability_screen.json",
         ["--tasks", record.get("evaluation_tasks", "hellaswag,arc_easy,piqa,winogrande")]+([] if mode=='full' else ['--limit','1000'])))
    if mode=='validation_only':pipeline=()
    for script, filename, extra in pipeline:
        output = run / filename
        existing=output.exists();capability=script=='evaluate_gibc.py'
        if existing:
            data = json.loads(output.read_text())
            if not output_matches(data,item,record): raise ValueError("stale evaluation " + filename)
            if capability and not (run/'capability_samples.json').exists():
                staged=run/'.capability_samples.json.tmp'
                if staged.exists():
                    json.loads(staged.read_text())
                    with staged.open('rb') as handle:os.fsync(handle.fileno())
                    os.replace(staged,run/'capability_samples.json');sync_dir(run)
                else:existing=False  # Redo incomplete publication, without accepting unpaired evidence.
        if not existing:
            temporary = run / ("." + filename + ".tmp")
            recovered=False
            if temporary.exists():
                try:
                    candidate=json.loads(temporary.read_text())
                    recovered=output_matches(candidate,item,record)
                    if capability:
                        samples=run/'.capability_samples.json.tmp'
                        if not samples.exists():samples=run/'capability_samples.json'
                        json.loads(samples.read_text())
                except (json.JSONDecodeError,FileNotFoundError):recovered=False
            command = [sys.executable, str(ROOT / "scripts" / script), "--checkpoint", str(checkpoint),
                       "--tokenizer", str(tokenizer), "--output", str(temporary), "--threads", "16"] + extra
            if script == "evaluate_gibc.py": command += ["--predictions-output", str(run / ".capability_samples.json.tmp")]
            if not recovered:
                with (run / "evaluate.log").open("a") as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    if child.wait()!=0:raise RuntimeError('Evaluation child failed: '+script)
                child=None
            if not output_matches(json.loads(temporary.read_text()),item,record):raise ValueError('Evaluator output identity mismatch')
            publish_outputs(run,temporary,output,capability)
        if capability:validate_samples(run,record,json.loads(output.read_text()))
        outputs.append({"file": filename, "sha256": sha256(output), "size_bytes": output.stat().st_size})
    if mode!='validation_only':
        samples=run/'capability_samples.json'
        if not samples.exists():raise RuntimeError('Capability output lacks paired per-item evidence')
        outputs.append({'file':samples.name,'sha256':sha256(samples),'size_bytes':samples.stat().st_size})
    validation = ensure_source_validation(record,run,item)
    if validation["checkpoint_sha256"] != item["sha256"]: raise RuntimeError("endpoint source evaluation mismatch")
    for path in (run/'latest_validation.json',protocol_path):
        outputs.append({'file':path.name,'sha256':sha256(path),'size_bytes':path.stat().st_size})
    if record.get('full_co4'):
        path=run/'architecture_diagnostics.json'
        outputs.append({'file':path.name,'sha256':sha256(path),'size_bytes':path.stat().st_size})
    receipt = {'schema':'followup-evaluation-receipt-v1',"at": utc(), "status": "COMPLETED", "checkpoint_sha256": item["sha256"],
               'registration_sha256':object_hash(record),
               'evaluation_source_hashes':protocol['source_hashes'],
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
