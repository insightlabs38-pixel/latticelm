"""Fourteen-hour autonomous successor for the LatticeReason tournament.

The process owns predecessor waiting, recovery, conservative branch selection,
milestone persistence, reporting, and its own deadlines.  It is intentionally a
single detached state machine: an interactive agent must not babysit it.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
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
from typing import Any

import torch
from huggingface_hub import get_token, hf_hub_download

from latticelm.data_d import sha256_file, verify_top_manifest
from latticelm.hf_storage import export_checkpoint, upload_checkpoint

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts"
STATE=ART/"post_tournament_overnight_state.json";EVENTS=ART/"logs/post_tournament_overnight_events.jsonl"
LOG=ART/"logs/post_tournament_overnight.log";LOCK=ART/"post_tournament_overnight.lock"
PREDECESSOR="latticereason-tournament-20260906.service"
MANIFEST=ART/"data/phase7f_v2r1/canonical-100m/manifest.json"
TOKENIZER=ART/"tokenizers/babylm_2026_4k.json";TOKEN_REPORT=ART/"tokenizers/babylm_2026_4k.report.json"
CURVE=ART/"lattice_reason_training_curves.csv";REPO="insightlabs38-pixel/LatticeLM-research"
TASKS=("hellaswag","arc_easy","piqa","winogrande")
CONTROL={"run_id":"LR-0","lr_fraction":0.0,"training_tokens":25_000_000,"parameters":15_949_760,
 "common_validation_loss":4.464799,"data_d_validation_loss":3.695686,"wikitext_ppl":83.40936,
 "wikitext_bpb":2.03912,"hellaswag":.262896,"arc_easy":.283249,"piqa":.534276,
 "winogrande":.498027,"hf_revision":"1cf7116faf4b2896594239c83008518bcc84c37d"}
TASK_N={"hellaswag":10042,"arc_easy":2376,"piqa":1838,"winogrande":1267}
STOP=False;CHILD:subprocess.Popen|None=None

def now()->str:return datetime.now(timezone.utc).isoformat()
def epoch(value:str)->float:return datetime.fromisoformat(value).timestamp()
def digest(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def atomic(path:Path,value:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
 with tmp.open("w") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def event(kind:str,**fields)->None:
 EVENTS.parent.mkdir(parents=True,exist_ok=True);line=json.dumps({"at":now(),"event":kind,**fields},sort_keys=True)+"\n"
 fd=os.open(EVENTS,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o664)
 try:os.write(fd,line.encode());os.fsync(fd)
 finally:os.close(fd)
def load()->dict|None:
 return json.loads(STATE.read_text()) if STATE.exists() else None
def save(state:dict,stage:str|None=None,**fields)->None:
 if stage is not None:state["current_stage"]=stage
 state.update(fields);state["updated_at"]=now();atomic(STATE,state)
 if stage is not None:event("STAGE",stage=stage)
def stop_handler(*_):
 global STOP;STOP=True
 state=load()
 if state:state["stop_requested"]=True;save(state);event("STOP_REQUESTED")
 if CHILD and CHILD.poll() is None:
  with contextlib.suppress(ProcessLookupError):CHILD.send_signal(signal.SIGTERM)

class MasterLock:
 def __init__(self,path:Path=LOCK):self.path=path;self.handle=None
 def acquire(self)->bool:
  self.path.parent.mkdir(parents=True,exist_ok=True);self.handle=self.path.open("a+")
  try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:self.handle.close();self.handle=None;return False
  self.handle.seek(0);self.handle.truncate();self.handle.write(str(os.getpid())+"\n");self.handle.flush();return True
 def release(self):
  if self.handle:fcntl.flock(self.handle,fcntl.LOCK_UN);self.handle.close();self.handle=None

def initial(soft_hours:float,hard_hours:float)->dict:
 started=datetime.now(timezone.utc);commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 return {"run_id":f"post-tournament-{started:%Y%m%dT%H%M%SZ}","start_time":started.isoformat(),
  "soft_deadline":(started+timedelta(hours=soft_hours)).isoformat(),"hard_deadline":(started+timedelta(hours=hard_hours)).isoformat(),
  "predecessor_unit":PREDECESSOR,"predecessor_state":None,"tournament_result":None,"selected_fraction":None,
  "selected_checkpoint":None,"current_stage":"STARTING","completed_stages":[],"current_tokens":0,
  "latest_checkpoint":None,"source_git_commit":commit,"hf_state":{},"retry_counts":{},"spot_interruptions":0,
  "final_decision":None,"stop_requested":False,"master_pid":os.getpid(),"child_pid":None,"observed_throughput":4866.71}

def inspect_unit(unit:str=PREDECESSOR)->dict:
 fields="ActiveState,SubState,Result,ExecMainStatus,ExecMainExitTimestamp,InactiveExitTimestamp"
 done=subprocess.run(["systemctl","show",unit,f"--property={fields}","--no-pager"],capture_output=True,text=True)
 values={"query_returncode":done.returncode}
 for line in done.stdout.splitlines():
  if "=" in line:k,v=line.split("=",1);values[k]=v
 return values
def wait_for_predecessor(state:dict,interval:float=90,inspector=inspect_unit)->None:
 while True:
  observed=inspector();state["predecessor_state"]={**observed,"observed_at":now()};save(state,"WAITING_FOR_TOURNAMENT")
  if observed.get("ActiveState") not in {"active","activating","reloading"}:break
  if STOP:raise InterruptedError("stop requested while waiting")
  time.sleep(interval)
 event("PREDECESSOR_TERMINAL",**state["predecessor_state"])

def checkpoint(run_id:str,target:int|None=None)->tuple[Path,dict]|None:
 path=ART/"checkpoints"/run_id/"latest.pt";side=path.with_suffix(".sha256")
 if not path.exists() or not side.exists() or digest(path)!=side.read_text().strip():return None
 payload=torch.load(path,map_location="cpu",weights_only=False)
 required={"model","optimizer","scheduler","config","tokens_seen","python_rng_state","torch_rng_state","mixture_scheduler_state","next_batch_sha256","data_manifest_sha256"}
 if required-set(payload):return None
 if target is not None and int(payload["tokens_seen"])!=target:return None
 if payload.get("run_id")!=run_id:return None
 return path,payload
def curve_row(run_id:str,target:int)->dict|None:
 if not CURVE.exists():return None
 with CURVE.open() as f:rows=[r for r in csv.DictReader(f) if r["run_id"]==run_id and int(r["training_tokens"])==target]
 return rows[-1] if rows else None
def metrics_path(run_id:str,target:int)->Path:return ART/"lattice_reason"/run_id.lower()/f"metrics-{target}.json"
def valid_metrics(run_id:str,target:int)->dict|None:
 path=metrics_path(run_id,target)
 if not path.exists():
  legacy=ART/"lattice_reason"/run_id.lower()/"metrics.json"
  path=legacy if legacy.exists() else path
 try:value=json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return None
 return value if value.get("run_id")==run_id and int(value.get("training_tokens",-1))==target and all(k in value for k in TASKS) else None

def remaining(state:dict)->float:return epoch(state["hard_deadline"])-time.time()
def conservative_seconds(state:dict,start:int,target:int,full_eval:bool=True,parameters:int=15_949_760)->float:
 tps=max(500,float(state.get("observed_throughput") or 4866.71))*(15_949_760/parameters)*.82
 return max(0,target-start)/tps+(5400 if full_eval else 1200)+1800
def can_start(state:dict,start:int,target:int,full_eval:bool=True,parameters:int=15_949_760)->bool:
 return time.time()<epoch(state["soft_deadline"]) and conservative_seconds(state,start,target,full_eval,parameters)<remaining(state)

def run_child(state:dict,label:str,command:list[str],retries:int=2)->None:
 global CHILD
 for attempt in range(retries+1):
  if STOP or load().get("stop_requested"):raise InterruptedError("graceful stop requested")
  if remaining(state)<=0:raise TimeoutError("hard deadline reached")
  event("SUBPROCESS_START",label=label,attempt=attempt,command=command[1:3]);CHILD=subprocess.Popen(command,cwd=ROOT)
  state["child_pid"]=CHILD.pid;save(state);code=CHILD.wait();CHILD=None;state["child_pid"]=None;save(state)
  if code==0:return
  if code==75:raise TimeoutError(f"{label} stopped at safe checkpoint")
  state["retry_counts"][label]=attempt+1;save(state);event("SUBPROCESS_FAILED",label=label,attempt=attempt,returncode=code)
  if attempt==retries:raise RuntimeError(f"{label} exhausted retries")
  state["spot_interruptions"]+=1;save(state);event("SPOT_RESUME",label=label);time.sleep(min(30,2**attempt))

def train(state:dict,run_id:str,percent:int,target:int,config:Path|None=None,parameters:int=15_949_760)->dict:
 found=checkpoint(run_id);start=int(found[1]["tokens_seen"]) if found else 0
 if start>target:raise RuntimeError(f"{run_id} checkpoint is beyond requested target")
 if start<target:
  mode="--resume" if found else "--fresh";save(state,"TRAINING",selected_fraction=percent/100,current_tokens=start,latest_checkpoint=str(found[0]) if found else None)
  cmd=[sys.executable,"scripts/train_lattice_reason_mixture.py","--manifest",str(MANIFEST),"--run-id",run_id,"--lr-percent",str(percent),"--target",str(target),mode,"--hard-deadline-epoch",str(epoch(state["hard_deadline"])-900),"--expected-parameters",str(parameters)]
  if config:cmd.extend(["--config",str(config)])
  run_child(state,f"train-{run_id}-{target}",cmd)
 found=checkpoint(run_id,target);row=curve_row(run_id,target)
 if not found or not row:raise RuntimeError(f"invalid exact terminal state for {run_id}@{target}")
 state["observed_throughput"]=float(row["tokens_per_second"]);state["current_tokens"]=target;state["latest_checkpoint"]=str(found[0]);save(state)
 return evaluate_publish(state,run_id,percent,target,found[0],row,parameters)

def verify_remote(bundle:Path,remote:str,revision:str,token:str)->dict:
 sums=(bundle/"SHA256SUMS").read_text().splitlines();verified={}
 for line in sums:
  expected,name=line.split(None,1);name=name.strip();download=Path(hf_hub_download(REPO,f"{remote}/{name}",revision=revision,token=token))
  observed=digest(download)
  if observed!=expected:raise RuntimeError(f"remote SHA-256 mismatch: {name}")
  verified[name]=observed
 return verified
def evaluate_publish(state:dict,run_id:str,percent:int,target:int,native:Path,row:dict,parameters:int)->dict:
 existing=valid_metrics(run_id,target)
 if existing:return existing
 out=ART/"lattice_reason"/run_id.lower();out.mkdir(parents=True,exist_ok=True);copy=out/f"{run_id.lower()}-{target}.pt"
 if not copy.exists():shutil.copy2(native,copy);copy.with_suffix(".sha256").write_text(digest(copy)+"\n")
 wiki=out/f"wikitext-{target}.json";gibc=out/f"gibc-{target}.json";save(state,"EVALUATION",selected_checkpoint=str(copy))
 if not wiki.exists():run_child(state,f"wiki-{run_id}-{target}",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(copy),"--tokenizer",str(TOKENIZER),"--output",str(wiki),"--threads","4"],1)
 if not gibc.exists():run_child(state,f"gibc-{run_id}-{target}",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(copy),"--tokenizer",str(TOKENIZER),"--output",str(gibc),"--tasks",",".join(TASKS),"--threads","4"],1)
 w=json.loads(wiki.read_text());g=json.loads(gibc.read_text());result={"run_id":run_id,"lr_fraction":percent/100,"training_tokens":target,"parameters":parameters,
  "data_d_tokens":int(row["data_d_tokens"]),"lattice_reason_tokens":int(row["lattice_reason_tokens"]),"common_validation_loss":float(row["common_validation_loss"]),
  "data_d_validation_loss":float(row["data_d_validation_loss"]),"lattice_reason_validation_loss":float(row["lattice_reason_validation_loss"]),
  **{k:float(v) for k,v in row.items() if k.startswith("lattice_reason_family_") or k.startswith("lattice_reason_difficulty_")},
  "fineweb_edu_validation_loss":float(row["fineweb_edu_validation_loss"]),"wikipedia_validation_loss":float(row["wikipedia_validation_loss"]),"fineweb_validation_loss":float(row["fineweb_validation_loss"]),
  "wikitext_ppl":float(w["perplexity"]),"wikitext_bpb":float(w["bits_per_byte"]),**{t:float(g["results"][t]["acc,none"]) for t in TASKS},
  "tokens_per_second":float(row["tokens_per_second"]),"wall_seconds":float(row["wall_seconds"]),"checkpoint_sha256":digest(copy),"raw_gibc":str(gibc.relative_to(ROOT))}
 mp=metrics_path(run_id,target);atomic(mp,result);bundle=ART/"cache"/f"{run_id.lower()}-{target}"
 export_checkpoint(copy,bundle,f"{run_id}-{target}","milestone",TOKENIZER,TOKEN_REPORT,ART/"data_d_v2r1_manifest.json",{"tokens_trained":target,"wall_seconds":result["wall_seconds"],"val_loss":result["common_validation_loss"],"val_ppl":math.exp(result["common_validation_loss"])})
 token=get_token()
 if not token:raise RuntimeError("Hugging Face credential unavailable from secure credential store")
 remote=f"experiments/post-tournament/{run_id.lower()}-{target}";revision=upload_checkpoint(bundle,REPO,remote,token);remote_hashes=verify_remote(bundle,remote,revision,token)
 result.update(hf_revision=revision,hf_path=remote,remote_sha256_verified=True,remote_file_sha256=remote_hashes);atomic(mp,result)
 state["hf_state"][f"{run_id}@{target}"]={"revision":revision,"path":remote,"verified":True};save(state);event("MILESTONE_PUBLISHED",run_id=run_id,tokens=target,revision=revision)
 return result

def uncertainty(rows:list[dict])->list[dict]:
 output=[]
 for row in rows:
  if row["run_id"]=="LR-0":continue
  for task in TASKS:
   p=float(row[task]);q=float(CONTROL[task]);n=TASK_N[task];delta=p-q;se=math.sqrt(p*(1-p)/n+q*(1-q)/n);lo,hi=delta-1.96*se,delta+1.96*se
   label="CLEAR" if lo>0 or hi<0 else ("SUGGESTIVE" if abs(delta)>se else "WITHIN NOISE / AMBIGUOUS")
   output.append({"run_id":row["run_id"],"tokens":row["training_tokens"],"task":task,"delta":delta,"ci95_low":lo,"ci95_high":hi,"method":"independent-binomial approximation; raw paired vectors unavailable","classification":label})
 return output
def tournament_choice(rows:list[dict])->tuple[str,dict|None]:
 finals=[r for r in rows if r["training_tokens"]==25_000_000 and r["run_id"]!="LR-0"]
 if not {"LR-10","LR-25"}.issubset({r["run_id"] for r in finals}):return "F",None
 def score(r):
  deltas=[r[t]-CONTROL[t] for t in TASKS];return (sum(d>.003 for d in deltas),sum(deltas),-max(0,r["wikitext_ppl"]/CONTROL["wikitext_ppl"]-1))
 best=max(finals,key=score);signals=uncertainty([best]);clear=sum(x["classification"]=="CLEAR" and x["delta"]>0 for x in signals);suggest=sum(x["classification"]=="SUGGESTIVE" and x["delta"]>0 for x in signals)
 negative=sum(x["classification"]=="CLEAR" and x["delta"]<0 for x in signals);ppl_cost=best["wikitext_ppl"]/CONTROL["wikitext_ppl"]-1
 if negative>=2 or ppl_cost>.10:return "E",best
 if clear>=2 and sum(best[t]-CONTROL[t] for t in TASKS)>.02:return "A",best
 if clear>=1 or suggest>=2:return ("C" if ppl_cost>.05 else "B"),best
 if max(score(r)[0] for r in finals)>=2:return "F",best
 return "D",best

def recover_tournament(state:dict)->tuple[list[dict],str,dict|None]:
 save(state,"VERIFYING_TOURNAMENT");verify_top_manifest(MANIFEST,TOKENIZER);rows=[dict(CONTROL)]
 for run_id,percent in (("LR-10",10),("LR-25",25)):
  value=valid_metrics(run_id,25_000_000)
  if value is None:value=train(state,run_id,percent,25_000_000)
  rows.append(value)
 classification,winner=tournament_choice(rows)
 # Preserve a valid predecessor decision when it used the frozen declared rule.
 prior=ART/"lattice_reason_tournament_state.json"
 if prior.exists():
  with contextlib.suppress(Exception):
   decision=json.loads(prior.read_text()).get("final_decision") or {};declared=str(decision.get("classification",""))[:1]
   if declared in "ABCDEF":classification=declared
 state["tournament_result"]={"classification":classification,"rows":rows,"verified":True};state["selected_fraction"]=winner["lr_fraction"] if winner else None;save(state)
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 write_csv(ART/"lattice_reason_mixture_results.csv",rows)
 summary=["# LatticeReason mixture tournament","",f"LATTICEREASON TRANSFER RESULT: **{classification}**","",f"BEST LR FRACTION AT 25M: **{winner['lr_fraction'] if winner else 'INCONCLUSIVE'}**","","The failed service was recovered without restarting any completed lineage. LR-10 and LR-25 were verified at exact 25M checkpoints before this decision."]
 (ART/"lattice_reason_tournament_report.md").write_text("\n".join(summary)+"\n")
 (ART/"lattice_reason_tournament_decision.md").write_text("\n".join(summary+["",f"Predeclared decision: **{classification}**; later scaling must also pass the uncertainty and time gates."])+"\n")
 return rows,classification,winner

def data_d_fallback(state:dict)->dict:
 path=ART/"checkpoints/co4-l-data-d-v2r1-25m/latest.pt";payload=torch.load(path,map_location="cpu",weights_only=False);start=int(payload["tokens_seen"])
 if start<100_000_000:
  if not can_start(state,start,100_000_000):raise TimeoutError("DATA-D 100M fallback cannot fit safely")
  run_child(state,"data-d-fallback-100m",[sys.executable,"scripts/run_phase7f_v2r1_training.py","--manifest",str(MANIFEST),"--resume","--stop-tokens","100000000","--hard-deadline-epoch",str(epoch(state["hard_deadline"])-900)])
 row=list(csv.DictReader((ART/"data_d_v2r1_training_curve.csv").open()))[-1]
 native=path;out=ART/"data_d_fallback_100m";out.mkdir(exist_ok=True);copy=out/"native.pt"
 if not copy.exists():shutil.copy2(native,copy);copy.with_suffix(".sha256").write_text(digest(copy)+"\n")
 wiki=out/"wikitext.json";gibc=out/"gibc.json"
 if not wiki.exists():run_child(state,"data-d-wiki-100m",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(copy),"--tokenizer",str(TOKENIZER),"--output",str(wiki),"--threads","4"],1)
 if not gibc.exists():run_child(state,"data-d-gibc-100m",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(copy),"--tokenizer",str(TOKENIZER),"--output",str(gibc),"--tasks",",".join(TASKS),"--threads","4"],1)
 w=json.loads(wiki.read_text());g=json.loads(gibc.read_text())
 result={"run_id":"DATA-D-ONLY","lr_fraction":0.0,"training_tokens":int(row["training_tokens"]),"parameters":15_949_760,
  "common_validation_loss":float(row["common_validation_loss"]),"data_d_validation_loss":float(row["data_d_balanced_validation_loss"]),
  "fineweb_edu_validation_loss":float(row["fineweb_edu_validation_loss"]),"wikipedia_validation_loss":float(row["wikipedia_validation_loss"]),"fineweb_validation_loss":float(row["fineweb_validation_loss"]),
  "wikitext_ppl":float(w["perplexity"]),"wikitext_bpb":float(w["bits_per_byte"]),**{t:float(g["results"][t]["acc,none"]) for t in TASKS},"checkpoint_sha256":digest(copy)}
 bundle=ART/"cache/data-d-only-100m";export_checkpoint(copy,bundle,"data-d-only-100m","milestone",TOKENIZER,TOKEN_REPORT,ART/"data_d_v2r1_manifest.json",{"tokens_trained":100_000_000,"wall_seconds":float(row["cumulative_training_seconds"]),"val_loss":result["common_validation_loss"],"val_ppl":math.exp(result["common_validation_loss"])})
 token=get_token()
 if not token:raise RuntimeError("Hugging Face credential unavailable from secure credential store")
 remote="experiments/post-tournament/data-d-only-100m";revision=upload_checkpoint(bundle,REPO,remote,token);result.update(hf_revision=revision,hf_path=remote,remote_file_sha256=verify_remote(bundle,remote,revision,token));atomic(out/"metrics.json",result)
 return result

def final_commit()->None:
 paths=[ART/"post_tournament_overnight_report.md",ART/"post_tournament_overnight_decision.md",ART/"post_tournament_scaling.csv",ART/"post_tournament_benchmark_deltas.csv",ART/"post_tournament_uncertainty.csv",STATE,ART/"lattice_reason_tournament_report.md",ART/"lattice_reason_tournament_decision.md",ART/"lattice_reason_mixture_results.csv",CURVE,ART/"data_d_v2r1_training_curve.csv"]
 paths.extend(p for base in (ART/"lattice_reason",ART/"data_d_fallback_100m") if base.exists() for p in base.rglob("*.json"))
 existing=[str(p.relative_to(ROOT)) for p in paths if p.exists()];subprocess.run(["git","add","--",*existing],cwd=ROOT,check=True)
 staged=subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT)
 if staged.returncode:subprocess.run(["git","commit","-m","Record post-tournament overnight results"],cwd=ROOT,check=True)
 subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True)

def write_csv(path:Path,rows:list[dict])->None:
 keys=[]
 for row in rows:
  for key in row:
   if key not in keys:keys.append(key)
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore",lineterminator="\n");w.writeheader();w.writerows(rows)
def reports(state:dict,rows:list[dict],classification:str,winner:dict|None,fallback:dict|None=None)->None:
 uncertainty_rows=uncertainty(rows);write_csv(ART/"post_tournament_scaling.csv",rows+([fallback] if fallback else []));write_csv(ART/"post_tournament_uncertainty.csv",uncertainty_rows)
 deltas=[]
 for r in rows:
  if r["run_id"]!="LR-0":
   for metric in ("data_d_validation_loss","wikitext_ppl",*TASKS):
    if metric in r and metric in CONTROL:deltas.append({"run_id":r["run_id"],"tokens":r["training_tokens"],"metric":metric,"value":r[metric],"control":CONTROL[metric],"delta":r[metric]-CONTROL[metric]})
 write_csv(ART/"post_tournament_benchmark_deltas.csv",deltas)
 reached100=next((r for r in rows if winner and r["run_id"]==winner["run_id"] and r["training_tokens"]==100_000_000),None);reached200=next((r for r in rows if winner and r["run_id"]==winner["run_id"] and r["training_tokens"]==200_000_000),None);capacity=[r for r in rows if r["run_id"].startswith("LR24-")]
 noise="CLEAR" if any(x["classification"]=="CLEAR" for x in uncertainty_rows) else "NOISY / AMBIGUOUS"
 best=max(rows,key=lambda r:int(r.get("training_tokens",0))) if rows else CONTROL
 lines=["# Post-tournament overnight report","","TOURNAMENT COMPLETED: YES",f"LATTICEREASON TRANSFER RESULT: {classification}",f"BEST LR FRACTION: {winner['lr_fraction'] if winner else 'INCONCLUSIVE'}",f"BENCHMARK GAINS CLEAR OR NOISY: {noise}",f"WINNER REACHED 100M: {'YES' if reached100 else 'NO'}"]
 for label,value in (("100M",reached100),("200M",reached200)):
  if label=="200M":lines.append(f"WINNER REACHED 200M: {'YES' if value else 'NO'}")
  if value:lines.extend([f"","## {label} metrics","",json.dumps(value,indent=2,sort_keys=True)])
 lines.extend([f"~24M CAPACITY TESTED: {'YES' if capacity else 'NO'}",f"DATA-D-ONLY FALLBACK 100M RUN: {'YES' if fallback and fallback.get('training_tokens')==100_000_000 else 'NO'}","",f"BEST CURRENT BASE-PRETRAINING RECIPE: {winner['run_id'] if winner and classification in 'ABC' else 'DATA-D-BROAD-v2r1'}",f"BEST CURRENT MODEL: {best['run_id']} @ {best.get('training_tokens',0)} tokens",f"SHOULD LR MIXTURE SCALE FURTHER: {'YES' if reached200 and classification in 'AB' else ('INCONCLUSIVE' if classification in 'CF' else 'NO')}",f"SHOULD ~24M CAPACITY ADVANCE: {'YES' if capacity and len(capacity)>1 else ('NOT YET' if classification in 'ABCF' else 'NO')}","SHOULD POST-TRAINING BE INVESTIGATED NEXT: AFTER MORE BASE TRAINING","","Top 3 next actions (not launched):","1. Review raw milestone and uncertainty artifacts before allocating another scaling run.","2. Replicate the best base-pretraining result with a second predeclared seed if its gains are clear.","3. Resolve post-training competition eligibility before designing any SFT/RL experiment."])
 (ART/"post_tournament_overnight_report.md").write_text("\n".join(lines)+"\n");(ART/"post_tournament_overnight_decision.md").write_text("# Post-tournament decision\n\n"+"\n".join(lines[2:8])+"\n")
 state["final_decision"]={"classification":classification,"best_lr_fraction":winner["lr_fraction"] if winner else None,"winner_100m":bool(reached100),"winner_200m":bool(reached200),"capacity_tested":bool(capacity),"data_d_fallback_100m":bool(fallback and fallback.get("training_tokens")==100_000_000)};save(state)

def execute(state:dict)->None:
 wait_for_predecessor(state);rows,classification,winner=recover_tournament(state);fallback=None
 if classification in {"A","B","C"} and winner:
  percent=round(float(winner["lr_fraction"])*100);run_id=winner["run_id"];start=max(r["training_tokens"] for r in rows if r["run_id"]==run_id)
  if start<100_000_000 and can_start(state,start,100_000_000):rows.append(train(state,run_id,percent,100_000_000));start=100_000_000
  at100=next((r for r in rows if r["run_id"]==run_id and r["training_tokens"]==100_000_000),None)
  healthy=bool(at100 and at100["data_d_validation_loss"]<winner["data_d_validation_loss"] and at100["lattice_reason_validation_loss"]<=winner["lattice_reason_validation_loss"]*1.03)
  if healthy and can_start(state,start,200_000_000):rows.append(train(state,run_id,percent,200_000_000));start=200_000_000
  if healthy and start>=100_000_000 and can_start(state,0,25_000_000,parameters=24_402_816):
   rid=f"LR24-{percent}";rows.append(train(state,rid,percent,25_000_000,ROOT/"configs/post_tournament_co4_24m.json",24_402_816))
   if can_start(state,25_000_000,50_000_000,parameters=24_402_816):rows.append(train(state,rid,percent,50_000_000,ROOT/"configs/post_tournament_co4_24m.json",24_402_816))
 elif classification in {"D","E"}:
  save(state,"TRANSFER_DIAGNOSTIC");fallback=data_d_fallback(state)
 elif classification=="F" and winner:
  percent=round(float(winner["lr_fraction"])*100);target=50_000_000
  if can_start(state,25_000_000,target):rows.append(train(state,winner["run_id"],percent,target))
 else:fallback=data_d_fallback(state)
 save(state,"FINAL_REPORT");reports(state,rows,classification,winner,fallback)
 state["completed_stages"].append("FINAL_REPORT");save(state,"COMPLETE");final_commit();event("MASTER_COMPLETE")

def dry_run(base:Path|None=None)->int:
 verify_top_manifest(MANIFEST,TOKENIZER);assert tournament_choice([CONTROL,{**CONTROL,"run_id":"LR-10","lr_fraction":.1},{**CONTROL,"run_id":"LR-25","lr_fraction":.25}])[0]=="D"
 start=datetime.now(timezone.utc);fixture={"soft_deadline":(start+timedelta(hours=13)).isoformat(),"hard_deadline":(start+timedelta(hours=14)).isoformat(),"observed_throughput":4866.71}
 assert can_start(fixture,25_000_000,50_000_000);assert not can_start(fixture,0,1_000_000_000)
 print(json.dumps({"dry_run":"PASS","manifest":"verified","branch_selection":"verified","deadline":"verified","resume_fields":sorted(checkpoint("LR-10")[1].keys()) if checkpoint("LR-10") else "fresh-original-stage"},indent=2));return 0
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--soft-hours",type=float,default=13);p.add_argument("--hard-hours",type=float,default=14);p.add_argument("--wait-interval",type=float,default=90);p.add_argument("--dry-run",action="store_true");p.add_argument("--stop",action="store_true");p.add_argument("--status",action="store_true");a=p.parse_args()
 if a.stop:stop_handler();print("graceful stop requested");return 0
 if a.status:print(json.dumps(load(),indent=2));return 0
 if a.dry_run:return dry_run()
 lock=MasterLock()
 if not lock.acquire():print("successor already running");return 0
 signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler);state=load()
 if state and state.get("current_stage") in {"COMPLETE","FAILED"}:print("terminal successor state exists; refusing duplicate");lock.release();return 0
 if state is None:state=initial(a.soft_hours,a.hard_hours)
 try:
  save(state,"STARTING");event("MASTER_STARTED",run_id=state["run_id"]);execute(state);return 0
 except (InterruptedError,TimeoutError) as exc:
  state["last_error"]=f"{type(exc).__name__}: {exc}";save(state,"FINAL_REPORT");reports(state,(state.get("tournament_result") or {}).get("rows",[CONTROL]),(state.get("tournament_result") or {}).get("classification","F"),None);save(state,"STOPPED_SAFE");event("MASTER_STOPPED_SAFE",reason=str(exc));return 0
 except Exception as exc:
  state["last_error"]=f"{type(exc).__name__}: {exc}";save(state,"FAILED");event("MASTER_FAILED",error=state["last_error"]);raise
 finally:lock.release()
if __name__=="__main__":raise SystemExit(main())
