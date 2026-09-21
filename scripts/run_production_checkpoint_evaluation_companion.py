#!/usr/bin/env python3
"""Restart-safe evaluator for 1.00B/1.25B/1.50B production milestones.

This external companion uses the audited privileged stop/evaluate/start
protocol, but never changes production decisions. Raw results are fsynced and
production is restarted before comparisons are generated.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts"; RUN=ART/"final_production_run"
EVENTS=ART/"final_training_events.jsonl"; STATE=ART/"production_eval_companion_state.json"
EVENT_LOG=ART/"production_eval_companion_events.jsonl"; SERVICE="latticelm-final-production-20260917.service"
TOKENIZER=ART/"tokenizers/final_corpus_4k.json"; MANIFEST=ART/"data/data_d_v4/canonical-2250m/manifest.json"
CURVE=ART/"final_training_curve.csv"
TARGETS=(1_000_000_000,1_250_000_000,1_500_000_000); LABELS={1_000_000_000:"1000m",1_250_000_000:"1250m",1_500_000_000:"1500m"}
DEFAULT_INTERVAL=20.0; STOP=False

def now(): return time.time()
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name("."+path.name+".tmp")
 with tmp.open("w") as f:json.dump(value,f,indent=2,sort_keys=True,default=str);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def event(kind,**fields):
 EVENT_LOG.parent.mkdir(parents=True,exist_ok=True);row={"at":now(),"at_utc":datetime.now(timezone.utc).isoformat(),"event":kind,**fields}
 with EVENT_LOG.open("a") as f:f.write(json.dumps(row,sort_keys=True,default=str)+"\n");f.flush();os.fsync(f.fileno())
def new_state():return {"schema":"production-eval-companion-v2","status":"WAITING_FOR_MILESTONE","targets":{str(x):{"status":"PENDING"} for x in TARGETS},"completed_targets":[],"active_target":None,"service":SERVICE,"created_at":now(),"updated_at":now()}
def load_state():
 if not STATE.exists():return new_state()
 old=json.loads(STATE.read_text())
 if old.get("schema")=="production-eval-companion-v2":return old
 state=new_state();state["migrated_from_v1_at"]=now()
 if old.get("checkpoint"):state["legacy_evaluations"]=[old["checkpoint"]]
 return state
def save(state,status=None,**fields):
 if status is not None:state["status"]=status
 state.update(fields);state["updated_at"]=now();atomic(STATE,state);event("STATE",status=state.get("status"),active_target=state.get("active_target"),**fields)
def run(command,timeout=None,check=True):return subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=timeout,check=check)
def systemctl(*args,check=True):
 # Existing working privileged mechanism; the post-training master never uses it.
 return run(["sudo","-n","systemctl",*args],check=check,timeout=1500)
def production_processes():
 out=run(["ps","-eo","pid=,args="]).stdout
 return [x.strip() for x in out.splitlines() if ("run_final_production_master.py" in x or "train_final_production.py" in x) and "production_eval_companion" not in x]
def service_active():return systemctl("is-active","--quiet",SERVICE,check=False).returncode==0
def latest_checkpoint():
 path=RUN/"latest.pt";side=RUN/"latest.sha256"
 if not(path.is_file() and side.is_file()):return None
 before=(path.stat().st_size,path.stat().st_mtime_ns);digest=sha256(path);after=(path.stat().st_size,path.stat().st_mtime_ns)
 if before!=after or digest!=side.read_text().strip():return None
 import torch
 p=torch.load(path,map_location="cpu",weights_only=False);required=("schema","model","optimizers","config","config_sha256","tokens","step","manifest_sha256","tokenizer_sha256","next_batch_sha256","validation_loss")
 if any(k not in p for k in required) or p["schema"]!="final-production-checkpoint-v1":return None
 if before!=(path.stat().st_size,path.stat().st_mtime_ns) or sha256(path)!=digest:return None
 return path,p,digest
def event_records():
 rows=[]
 if not EVENTS.exists():return rows
 for line in EVENTS.read_text().splitlines():
  try:
   x=json.loads(line)
   if isinstance(x,dict):rows.append(x)
  except json.JSONDecodeError:pass
 return rows
def first_checkpoint_at_or_after(target,completed_tokens=None):
 done=completed_tokens or set();rows=[x for x in event_records() if x.get("event")=="CHECKPOINT" and int(x.get("tokens",-1))>=target and x.get("sha256") and int(x["tokens"]) not in done]
 return min(rows,key=lambda x:(int(x["tokens"]),float(x.get("at",0)))) if rows else None
def next_target(state):
 done={int(x) for x in state.get("completed_targets",[])};return next((x for x in TARGETS if x not in done),None)
def output_dir(target):return ART/f"production_milestone_eval_{LABELS[target]}"
def pin_checkpoint(state,target,record,candidate):
 path,p,digest=candidate
 if int(p["tokens"])!=int(record["tokens"]) or digest!=record["sha256"]:raise RuntimeError("checkpoint event and payload identity mismatch")
 out=output_dir(target);out.mkdir(parents=True,exist_ok=True);pinned=out/"checkpoint.pt"
 if not pinned.exists():os.link(path,pinned)
 elif sha256(pinned)!=digest:raise RuntimeError("existing pinned checkpoint hash mismatch")
 side=out/"checkpoint.sha256";side.write_text(digest+"\n")
 with side.open("r+") as f:f.flush();os.fsync(f.fileno())
 import torch
 reread=torch.load(pinned,map_location="cpu",weights_only=False)
 if int(reread["tokens"])!=int(p["tokens"]) or sha256(pinned)!=digest:raise RuntimeError("pinned checkpoint failed immutable verification")
 identity={"target_tokens":target,"tokens":int(p["tokens"]),"step":int(p["step"]),"checkpoint":str(pinned),"source_checkpoint":str(path),"checkpoint_sha256":digest,"config_sha256":p["config_sha256"],"tokenizer_sha256":p["tokenizer_sha256"],"manifest_sha256":p["manifest_sha256"],"peak_lr":p["config"].get("learning_rate"),"validation_loss":p["validation_loss"],"source_validation":p.get("source_validation",{}),"train_loss":p.get("train_loss"),"checkpoint_event":record,"pinned_at":now()}
 atomic(out/"checkpoint_identity.json",identity);state.update(active_target=target,checkpoint=identity,output_dir=str(out));state["targets"][str(target)]={"status":"PINNED","checkpoint":identity};save(state,"CHECKPOINT_PINNED");return identity
def pause_production(state):
 save(state,"PRODUCTION_PAUSING",pause_requested_at=now());result=systemctl("stop",SERVICE,check=False)
 if result.returncode:save(state,"BLOCKED_UNSAFE_PAUSE",pause_error=result.stderr[-2000:]);raise RuntimeError("production stop request failed")
 deadline=time.monotonic()+1500
 while time.monotonic()<deadline and (service_active() or production_processes()):time.sleep(2)
 if service_active() or production_processes():save(state,"BLOCKED_UNSAFE_PAUSE",pause_error="production remained active");raise RuntimeError("production did not stop cleanly")
 resume=latest_checkpoint()
 if resume is None or int(resume[1]["tokens"])<int(state["checkpoint"]["tokens"]):raise RuntimeError("safe-stop checkpoint absent or rolled backward")
 state["resume_checkpoint"]={"tokens":int(resume[1]["tokens"]),"step":int(resume[1]["step"]),"sha256":resume[2],"next_batch_sha256":resume[1]["next_batch_sha256"],"train_loss":resume[1].get("train_loss"),"target_tokens":resume[1].get("target_tokens")}
 state["paused_at"]=now();state["pause_transition_seconds"]=state["paused_at"]-state["pause_requested_at"];save(state,"PRODUCTION_PAUSED")
def evaluate(state):
 out=Path(state["output_dir"]);cp=state["checkpoint"]["checkpoint"];save(state,"EVALUATING",evaluation_started_at=now())
 commands=[[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",cp,"--tokenizer",str(TOKENIZER),"--output",str(out/"wikitext.json"),"--batch-size","16","--threads","16"],[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",cp,"--tokenizer",str(TOKENIZER),"--output",str(out/"gibc.json"),"--tasks","hellaswag,arc_easy,piqa,winogrande","--batch-size","16","--threads","16"]]
 for command in commands:
  result=run(command,timeout=3600,check=False)
  if result.returncode:raise RuntimeError(f"evaluation failed ({result.returncode}): {result.stderr[-1500:]}")
 for name in ("wikitext.json","gibc.json"):
  path=out/name;json.loads(path.read_text())
  with path.open("r+") as f:f.flush();os.fsync(f.fileno())
 raw={n:{"path":str(out/n),"sha256":sha256(out/n)} for n in ("wikitext.json","gibc.json")};atomic(out/"raw_results_identity.json",raw);save(state,"RAW_RESULTS_DURABLE",raw_results=raw,evaluation_finished_at=now(),evaluation_duration_seconds=now()-state["evaluation_started_at"])
def expected_lr_scale(tokens,target,warmup=.02,stable=.83,decay=.15):
 f=min(1.0,tokens/max(target,1))
 if f<warmup:return f/warmup
 if f<warmup+stable:return 1.0
 progress=min(1.0,max(0.0,(f-warmup-stable)/decay));return .5*(1+math.cos(math.pi*progress))
def curve_progress(after_tokens):
 if not CURVE.exists():return None
 try:
  with CURVE.open() as f:rows=[x for x in csv.DictReader(f) if int(x.get("tokens",-1))>after_tokens]
  if not rows:return None
  row=rows[-1];return {"event":"CURVE_PROGRESS","tokens":int(row["tokens"]),"target_tokens":int(row["target_tokens"]),"train_loss":float(row["train_loss"]),"effective_tok_s":float(row["effective_tok_s"]),"lr":float(row["lr"]),"lr_scale":float(row["lr_scale"])}
 except (OSError,ValueError,KeyError):return None
def restart_and_verify(state):
 save(state,"PRODUCTION_RESTARTING",restart_requested_at=now());result=systemctl("start",SERVICE,check=False)
 if result.returncode:raise RuntimeError(f"production restart failed: {result.stderr[-2000:]}")
 deadline=time.monotonic()+180
 while time.monotonic()<deadline:
  if service_active() and len(production_processes())==2:break
  time.sleep(2)
 if not service_active() or len(production_processes())!=2:raise RuntimeError("resume did not produce exactly one master and trainer")
 base=state["resume_checkpoint"];resume=progress=None;end=time.monotonic()+360
 while time.monotonic()<end:
  rows=event_records();rs=[x for x in rows if x.get("event")=="RESUME" and int(x.get("tokens",-1))==int(base["tokens"]) and float(x.get("at",0))>=state["restart_requested_at"]];progress=curve_progress(int(base["tokens"]))
  if rs:resume=rs[-1]
  if resume and progress:break
  time.sleep(5)
 if not resume:raise RuntimeError("exact resume token identity not observed")
 if not progress:raise RuntimeError("monotonic resume progress not observed")
 loss=float(progress.get("validation_loss",progress.get("train_loss","nan")));tps=float(progress.get("effective_tok_s",0));prior=float(state["checkpoint"]["checkpoint_event"].get("effective_tok_s",tps))
 if not math.isfinite(loss):raise RuntimeError("resumed production reported nonfinite loss")
 if tps<=0 or (prior>0 and tps<prior*.25):raise RuntimeError("resumed production throughput is abnormal")
 target=int(progress.get("target_tokens",base.get("target_tokens") or 1_900_000_000));expected=expected_lr_scale(int(progress["tokens"]),target)
 if abs(float(progress.get("lr_scale",expected))-expected)>.02:raise RuntimeError("resumed schedule/LR state is unexpected")
 state["resume_event"]=resume;state["resume_progress_event"]=progress;state["resume_schedule_verification"]={"name":"2/83/15 WSD","tokens":int(progress["tokens"]),"target_tokens":target,"expected_lr_scale":expected,"observed_lr_scale":progress.get("lr_scale"),"observed_lr":progress.get("lr")};state["resume_verified_at"]=now();state["resume_verification_seconds"]=now()-state["restart_requested_at"];save(state,"RESUME_VERIFIED")
def comparison_sources(state):
 out=[{"id":"muon-100m","directory":str(ART/"final_recipe_experiments/final-muon-100m")},{"id":"production-750m","directory":str(ART/"production_milestone_eval_750002176")}]
 for target in TARGETS:
  if target>=int(state["active_target"]):break
  if state["targets"].get(str(target),{}).get("status")=="COMPLETE":out.append({"id":LABELS[target],"directory":str(output_dir(target))})
 return out
def report(state):
 out=Path(state["output_dir"]);wiki=json.loads((out/"wikitext.json").read_text());gibc=json.loads((out/"gibc.json").read_text());tasks={}
 for task in ("hellaswag","arc_easy","piqa","winogrande"):
  v=gibc.get("results",{}).get(task,{});tasks[task]={"raw_accuracy":v.get("acc,none"),"normalized_accuracy":v.get("acc_norm,none")}
 comparison={"schema":"production-milestone-comparison-v2","checkpoint":state["checkpoint"],"dedicated_wikitext":{"perplexity":wiki.get("perplexity"),"bits_per_byte":wiki.get("bits_per_byte"),"source":str(out/"wikitext.json")},"lm_eval_tasks":tasks,"lm_eval_wikitext":None,"comparison_sources":comparison_sources(state),"production_resume":{"exact_resume":state.get("resume_event"),"progress":state.get("resume_progress_event"),"schedule":state.get("resume_schedule_verification")},"training_decisions_changed":False,"generated_at":now()};atomic(out/"comparison.json",comparison)
 lines=[f"# Production milestone: {LABELS[int(state['active_target'])]}","",f"Checkpoint tokens: {state['checkpoint']['tokens']}",f"Checkpoint SHA: `{state['checkpoint']['checkpoint_sha256']}`","","Production restarted immediately after raw results became durable.","",f"- Dedicated WikiText PPL: {wiki.get('perplexity')}",f"- Dedicated WikiText BPB: {wiki.get('bits_per_byte')}"]+[f"- {k}: raw={v['raw_accuracy']}, normalized={v['normalized_accuracy']}" for k,v in tasks.items()]+["","Dedicated WikiText is separate from lm-eval metrics. No training decision was changed."]
 path=out/"comparison.md";path.write_text("\n".join(lines)+"\n");target=int(state["active_target"]);state["targets"][str(target)]={"status":"COMPLETE","checkpoint":state["checkpoint"],"raw_results":state["raw_results"],"comparison":str(out/"comparison.json"),"report":str(path),"completed_at":now()}
 if target not in state["completed_targets"]:state["completed_targets"].append(target)
 state["completed_targets"].sort();state["active_target"]=None
 for key in ("checkpoint","output_dir","resume_checkpoint","raw_results"):state.pop(key,None)
 save(state,"COMPLETE" if target==TARGETS[-1] else "WAITING_FOR_MILESTONE",completed_target=target)
def handle_active(state):
 status=state.get("status")
 if status=="CHECKPOINT_PINNED":pause_production(state);status=state["status"]
 if status in {"PRODUCTION_PAUSED","EVALUATING"}:evaluate(state);status=state["status"]
 if status in {"RAW_RESULTS_DURABLE","PRODUCTION_RESTARTING"}:restart_and_verify(state);status=state["status"]
 if status=="RESUME_VERIFIED":report(state)
def signal_handler(*_):
 global STOP;STOP=True
def main():
 p=argparse.ArgumentParser();p.add_argument("--poll-seconds",type=float,default=DEFAULT_INTERVAL);a=p.parse_args();signal.signal(signal.SIGTERM,signal_handler);signal.signal(signal.SIGINT,signal_handler);state=load_state();save(state)
 if state.get("status")=="COMPLETE" and next_target(state) is None:return 0
 try:
  if state.get("active_target") is not None:handle_active(state)
  while not STOP:
   target=next_target(state)
   if target is None:save(state,"COMPLETE",completed_at=now());return 0
   record=first_checkpoint_at_or_after(target,{int(state["targets"][str(x)].get("checkpoint",{}).get("tokens",-1)) for x in TARGETS});candidate=latest_checkpoint() if record else None
   if record and candidate and int(candidate[1]["tokens"])==int(record["tokens"]) and candidate[2]==record["sha256"]:pin_checkpoint(state,target,record,candidate);handle_active(state)
   else:
    deadline=time.monotonic()+a.poll_seconds
    while not STOP and time.monotonic()<deadline:time.sleep(min(1,max(0,deadline-time.monotonic())))
  return 75
 except Exception as error:
  failure=f"{type(error).__name__}: {error}";state["error"]=failure
  if state.get("status") in {"PRODUCTION_PAUSED","EVALUATING","RAW_RESULTS_DURABLE","PRODUCTION_RESTARTING"} and not service_active():
   try:restart_and_verify(state)
   except Exception as restart_error:state["restart_error"]=repr(restart_error)
  target=state.get("active_target")
  if target is not None:state["targets"][str(target)]={**state["targets"].get(str(target),{}),"status":"FAILED","failure":failure}
  save(state,"EVALUATION_FAILED_RESUMED" if service_active() else "BLOCKED_UNSAFE_PAUSE",failure=failure);return 2
if __name__=="__main__":raise SystemExit(main())
