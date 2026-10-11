"""Selective launch gate: scientific-batch smoke, SIGKILL recovery, evaluator load."""
from __future__ import annotations
import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from followup_common import (ROOT, RESEARCH, STATE, atomic_json, object_hash, register, registered, sha256, utc, verified_checkpoint)


def wait(process, label, timeout=900):
    start = time.monotonic(); last = 0
    while process.poll() is None:
        if time.monotonic()-start > timeout:
            process.terminate(); raise RuntimeError(label + " exceeded smoke time limit")
        if time.monotonic()-last > 30:
            print(label + f": {time.monotonic()-start:.0f}s", flush=True); last = time.monotonic()
        time.sleep(.2)
    if process.returncode: raise RuntimeError(label + f" failed ({process.returncode}); inspect smoke master/train logs")


def launch(name):
    log = (RESEARCH / (name + "-master.log")).open("a")
    process = subprocess.Popen([sys.executable, str(ROOT / "scripts/posthack_followup_master.py"),
        "--only", name, "--exit-after-trained", "--poll-seconds", ".25"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    log.close(); return process


def compare(a, b, path=""):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b): raise RuntimeError("resume tensor mismatch " + path)
    elif isinstance(a,dict):
        if set(a)!=set(b): raise RuntimeError("resume dictionary keys mismatch " + path)
        for k in a: compare(a[k],b[k],path+"/"+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise RuntimeError("resume list length mismatch " + path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+"/"+str(i))
    elif hasattr(a,"shape"):
        import numpy as np
        if not np.array_equal(a,b):raise RuntimeError("resume array mismatch " + path)
    elif a!=b: raise RuntimeError("resume value mismatch " + path)


def verify(name, smoke_steps=100):
    parent = registered(name); step_tokens=parent["config"]["batch_size"]*parent["config"]["context_length"]
    names = ["SMOKE-REFERENCE-"+name, "SMOKE-INTERRUPTED-"+name]
    for index, smoke_name in enumerate(names):
        r=copy.deepcopy(parent);r.update({"experiment_id":smoke_name,"smoke":True,"token_budget":smoke_steps*step_tokens,
            "schedule_token_budget":parent["token_budget"],"checkpoint_tokens":20*step_tokens,"milestone_tokens":[],
            "ready":True,"hypothesis":"Operational validity test, not a quality result"})
        if index:r["test_pause_after_checkpoint_step"]=20
        register(r)
    process=launch(names[0]);wait(process,"Uninterrupted 100-step scientific-batch smoke")
    interrupted=launch(names[1]);run=RESEARCH/"runs"/names[1];rendezvous=run/"test_checkpoint_ready.json"
    started=time.monotonic()
    while not rendezvous.exists():
        if interrupted.poll() is not None: raise RuntimeError("interrupted smoke exited before checkpoint; inspect logs")
        if time.monotonic()-started>180:raise RuntimeError("checkpoint rendezvous timed out")
        time.sleep(.1)
    item=json.loads(rendezvous.read_text())
    if item["step"]!=20 or item["logical_tokens"]!=20*step_tokens:raise RuntimeError("incorrect smoke checkpoint position")
    active=json.loads(STATE.read_text())["active"]
    if active["experiment_id"]!=names[1]:raise RuntimeError("unexpected active child")
    os.kill(active["pid"],signal.SIGKILL);interrupted.kill();interrupted.wait()
    print("SIGKILL trainer and master after committed step 20; restarting through master.",flush=True)
    process=launch(names[1]);wait(process,"Resumed scientific-batch smoke")
    run_dirs=[RESEARCH/"runs"/n for n in names]
    cps=[verified_checkpoint(d) for d in run_dirs]
    a,b=[torch.load(d/c["file"],map_location="cpu",weights_only=False) for d,c in zip(run_dirs,cps)]
    for key in ("model","optimizers","mixture","sampling_cycle","python_rng","numpy_rng","torch_rng","step",
                "logical_tokens","processed_tokens","next_batch_sha256","sampler_diagnostics"):
        compare(a[key],b[key],key)
    attempts=[json.loads(line) for f in run.glob("metrics-*.jsonl") for line in f.read_text().splitlines()]
    starts=[json.loads(line) for line in (RESEARCH/"FOLLOWUP_DECISIONS.jsonl").read_text().splitlines()
            if json.loads(line).get("decision")=="EXACT_RESUME" and json.loads(line).get("experiment_id")==names[1]]
    if not starts or starts[-1]["step"]!=20:raise RuntimeError("master did not resume from exact saved step")
    # The public evaluator must load this checkpoint schema and score real text.
    from evaluate_gibc import LatticeHarnessLM
    torch.set_num_threads(4)
    lm=LatticeHarnessLM(str(run_dirs[0]/cps[0]["file"]),str(ROOT/parent["tokenizer_path"]))
    score=lm.loglikelihood([SimpleNamespace(args=("The sky", " is blue."))])[0][0]
    if not torch.isfinite(torch.tensor(score)):raise RuntimeError("evaluator returned nonfinite score")
    receipts=[json.loads((d/"train_receipt.json").read_text()) for d in run_dirs]
    state=json.loads(STATE.read_text())
    passed={"at":utc(),"passed":True,"experiment_id":name,"registration_sha256":object_hash(parent),
            "source_hashes":parent["source_hashes"],"smoke_steps":smoke_steps,"smoke_batch_size":parent["config"]["batch_size"],
            "smoke_context_length":parent["config"]["context_length"],"kill_checkpoint_step":20,
            "exact_model_optimizer_rng_sampler_equality":True,"evaluator_compatibility":True,"sample_loglikelihood":score,
            "resume_events":starts,"infrastructure_classification":state["runs"][names[1]]["infrastructure_interruptions"],
            "smoke_tokens_per_second":receipts[0]["logical_tokens"]/receipts[0]["wall_seconds"],
            "checkpoint_hashes":[c["sha256"] for c in cps],"selective_verification_policy":"focused on scientific-batch training and exact preemption resume; no broad benchmark run during launch verification"}
    atomic_json(RESEARCH/parent["launch_certificate"],passed)
    print(json.dumps(passed,indent=2),flush=True)
    return passed


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--experiment-id",default="F32-DENSE-S1337-300M");p.add_argument("--smoke-steps",type=int,default=100)
    a=p.parse_args();verify(a.experiment_id,a.smoke_steps)
