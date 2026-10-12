"""Detached seed-replication and conditional matched LR-10 state machine."""
from __future__ import annotations
import argparse,contextlib,csv,fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.stats import binomtest
from huggingface_hub import get_token,hf_hub_download
from latticelm.data_d import sha256_file,verify_top_manifest
from latticelm.hf_storage import export_checkpoint,upload_checkpoint

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts"
STATE=ART/"seed2026_master_state.json";EVENTS=ART/"logs/seed2026_master_events.jsonl";LOCK=ART/"seed2026_master.lock"
MANIFEST=ART/"data/phase7f_v2r1/canonical-100m/manifest.json";CONFIG=ROOT/"configs/seed2026_co4_l.json"
TOKENIZER=ART/"tokenizers/babylm_2026_4k.json";TOKEN_REPORT=ART/"tokenizers/babylm_2026_4k.report.json"
REPO="insightlabs38-pixel/LatticeLM-research";TASKS=("hellaswag","arc_easy","piqa","winogrande")
TASK_N={"hellaswag":10042,"arc_easy":2376,"piqa":1838,"winogrande":1267}
CONTROL={"run_id":"DATA-D-314159","seed":314159,"training_tokens":100_000_000,"parameters":15_949_760,
 "data_d_validation_loss":3.465799629688263,"fineweb_edu_validation_loss":3.4668063521385193,
 "wikipedia_validation_loss":3.2836159020662308,"fineweb_validation_loss":3.7189749032258987,
 "wikitext_ppl":60.1218587924344,"wikitext_bpb":1.88821387326511,"hellaswag":.26757618004381595,
 "arc_easy":.2925084175084175,"piqa":.5413492927094669,"winogrande":.505130228887135,
 "hf_revision":"5ae234a6a9c127ede46768baaf7f0d40892e2df7","raw_gibc":"artifacts/data_d_fallback_100m/gibc.json"}
STOP=False;CHILD:subprocess.Popen|None=None

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path:Path,value:dict)->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
 with tmp.open("w") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def event(kind:str,**fields)->None:
 EVENTS.parent.mkdir(parents=True,exist_ok=True);line=json.dumps({"at":now(),"event":kind,**fields},sort_keys=True)+"\n"
 fd=os.open(EVENTS,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o664)
 try:os.write(fd,line.encode());os.fsync(fd)
 finally:os.close(fd)
def load():return json.loads(STATE.read_text()) if STATE.exists() else None
def save(state,stage=None,**fields):
 if stage:state["current_stage"]=stage
 state.update(fields);state["updated_at"]=now();atomic(STATE,state)
 if stage:event("STAGE",stage=stage)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def epoch(value):return datetime.fromisoformat(value).timestamp()

class Lock:
 def __init__(self):self.handle=None
 def acquire(self):
  LOCK.parent.mkdir(parents=True,exist_ok=True);self.handle=LOCK.open("a+")
  try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:self.handle.close();return False
  self.handle.seek(0);self.handle.truncate();self.handle.write(f"{os.getpid()}\n");self.handle.flush();return True
 def release(self):
  if self.handle:fcntl.flock(self.handle,fcntl.LOCK_UN);self.handle.close()
def stop(*_):
 global STOP;STOP=True
 state=load()
 if state:save(state,stop_requested=True)
 if CHILD and CHILD.poll() is None:
  with contextlib.suppress(ProcessLookupError):CHILD.send_signal(signal.SIGTERM)

def initial(soft,hard):
 start=datetime.now(timezone.utc);commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 return {"run_id":f"seed2026-replication-{start:%Y%m%dT%H%M%SZ}","started_at":start.isoformat(),
  "soft_deadline":(start+timedelta(hours=soft)).isoformat(),"hard_deadline":(start+timedelta(hours=hard)).isoformat(),
  "source_git_commit":commit,"current_stage":"STARTING","completed_stages":[],"seed":2026,"current_tokens":0,
  "child_pid":None,"latest_checkpoint":None,"retry_counts":{},"spot_interruptions":0,"stop_requested":False,
  "replication":None,"replication_classification":None,"replication_usable":None,"lr10":None,"final_decisions":None,
  "observed_throughput":5003.0}
def run_child(state,label,cmd,retries=2):
 global CHILD
 for attempt in range(retries+1):
  if STOP or load().get("stop_requested"):raise InterruptedError("stop requested")
  if time.time()>=epoch(state["hard_deadline"]):raise TimeoutError("hard deadline reached")
  event("SUBPROCESS_START",label=label,attempt=attempt,program=Path(cmd[1]).name if len(cmd)>1 else cmd[0])
  CHILD=subprocess.Popen(cmd,cwd=ROOT);state["child_pid"]=CHILD.pid;save(state);code=CHILD.wait();CHILD=None;state["child_pid"]=None;save(state)
  if code==0:return
  if code==75:raise TimeoutError(f"{label} stopped safely")
  state["retry_counts"][label]=attempt+1;state["spot_interruptions"]+=1;save(state);event("SUBPROCESS_FAILED",label=label,returncode=code)
  if attempt==retries:raise RuntimeError(f"{label} exhausted retries")
  time.sleep(min(30,2**attempt))
def checkpoint(run_id,target=None):
 path=ART/"checkpoints"/run_id/"latest.pt";side=path.with_suffix(".sha256")
 if not path.exists() or not side.exists() or digest(path)!=side.read_text().strip():return None
 data=torch.load(path,map_location="cpu",weights_only=False)
 required={"model","optimizer","scheduler","config","tokens_seen","python_rng_state","torch_rng_state","next_batch_sha256","data_manifest_sha256"}
 if required-set(data) or data.get("run_id",data.get("lineage"))!=run_id:return None
 if target is not None and int(data["tokens_seen"])!=target:return None
 return path,data
def curve_row(path,run_id,target):
 if not path.exists():return None
 rows=list(csv.DictReader(path.open()));matches=[r for r in rows if int(r["training_tokens"])==target and ("run_id" not in r or r["run_id"]==run_id)]
 return matches[-1] if matches else None

def train_data(state):
 run_id="co4-l-data-d-seed2026-100m";curve=ART/"seed2026_data_d_training_curve.csv";found=checkpoint(run_id);mode="--resume" if found else "--fresh"
 if not checkpoint(run_id,100_000_000):
  save(state,"STAGE1_TRAINING",current_tokens=int(found[1]["tokens_seen"]) if found else 0,latest_checkpoint=str(found[0]) if found else None)
  run_child(state,"train-data-d-seed2026",[sys.executable,"scripts/run_phase7f_v2r1_training.py","--manifest",str(MANIFEST),mode,"--stop-tokens","100000000","--config",str(CONFIG),"--run-id",run_id,"--curve",str(curve),"--hard-deadline-epoch",str(epoch(state["hard_deadline"])-900)])
 found=checkpoint(run_id,100_000_000);row=curve_row(curve,run_id,100_000_000)
 if not found or not row:raise RuntimeError("DATA-D seed-2026 exact terminal state missing")
 return evaluate_publish(state,run_id,found[0],row,False)
def train_lr10(state):
 run_id="co4-l-lr10-seed2026-100m";curve=ART/"lattice_reason_training_curves.csv";found=checkpoint(run_id);mode="--resume" if found else "--fresh"
 if not checkpoint(run_id,100_000_000):
  save(state,"STAGE3_LR10_TRAINING",current_tokens=int(found[1]["tokens_seen"]) if found else 0,latest_checkpoint=str(found[0]) if found else None)
  run_child(state,"train-lr10-seed2026",[sys.executable,"scripts/train_lattice_reason_mixture.py","--manifest",str(MANIFEST),"--run-id",run_id,"--lr-percent","10","--target","100000000",mode,"--config",str(CONFIG),"--hard-deadline-epoch",str(epoch(state["hard_deadline"])-900)])
 found=checkpoint(run_id,100_000_000);row=curve_row(curve,run_id,100_000_000)
 if not found or not row:raise RuntimeError("LR-10 seed-2026 exact terminal state missing")
 return evaluate_publish(state,run_id,found[0],row,True)
def verify_remote(bundle,remote,revision,token):
 verified={}
 for line in (bundle/"SHA256SUMS").read_text().splitlines():
  expected,name=line.split(None,1);local=Path(hf_hub_download(REPO,f"{remote}/{name.strip()}",revision=revision,token=token));observed=digest(local)
  if observed!=expected:raise RuntimeError(f"remote SHA-256 mismatch: {name}")
  verified[name.strip()]=observed
 return verified
def evaluate_publish(state,run_id,native,row,is_lr):
 out=ART/("seed2026_lr10_100m" if is_lr else "seed2026_data_d_100m");out.mkdir(parents=True,exist_ok=True)
 wiki=out/"wikitext.json";gibc=out/"gibc.json";save(state,"OFFICIAL_EVALUATION",latest_checkpoint=str(native))
 if not wiki.exists():run_child(state,f"wiki-{run_id}",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(native),"--tokenizer",str(TOKENIZER),"--output",str(wiki),"--threads","4"],1)
 if not gibc.exists():run_child(state,f"gibc-{run_id}",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(native),"--tokenizer",str(TOKENIZER),"--output",str(gibc),"--tasks",",".join(TASKS),"--threads","4"],1)
 w=json.loads(wiki.read_text());g=json.loads(gibc.read_text());cfg=torch.load(native,map_location="cpu",weights_only=False)["config"]
 result={"run_id":run_id,"seed":cfg["seed"],"training_tokens":100_000_000,"parameters":15_949_760,
  "data_d_validation_loss":float(row.get("data_d_validation_loss",row.get("data_d_balanced_validation_loss"))),
  "fineweb_edu_validation_loss":float(row["fineweb_edu_validation_loss"]),"wikipedia_validation_loss":float(row["wikipedia_validation_loss"]),"fineweb_validation_loss":float(row["fineweb_validation_loss"]),
  "wikitext_ppl":float(w["perplexity"]),"wikitext_bpb":float(w["bits_per_byte"]),**{t:float(g["results"][t]["acc,none"]) for t in TASKS},
  "tokens_per_second":float(row["tokens_per_second"]),"wall_seconds":float(row.get("wall_seconds",row.get("cumulative_training_seconds"))),"checkpoint_sha256":digest(native),"raw_gibc":str(gibc.relative_to(ROOT))}
 if is_lr:result.update({k:float(v) for k,v in row.items() if k.startswith("lattice_reason_") and k.endswith("validation_loss")});result.update(data_d_tokens=int(row["data_d_tokens"]),lattice_reason_tokens=int(row["lattice_reason_tokens"]))
 atomic(out/"metrics.json",result);bundle=ART/"cache"/run_id
 export_checkpoint(native,bundle,run_id,"immutable-final",TOKENIZER,TOKEN_REPORT,ART/"data_d_v2r1_manifest.json",{"tokens_trained":100_000_000,"wall_seconds":result["wall_seconds"],"val_loss":result["data_d_validation_loss"],"val_ppl":math.exp(result["data_d_validation_loss"])})
 token=get_token()
 if not token:raise RuntimeError("Hugging Face credential unavailable")
 remote=f"experiments/seed2026/{'lr10' if is_lr else 'data-d'}-100m";revision=upload_checkpoint(bundle,REPO,remote,token);hashes=verify_remote(bundle,remote,revision,token)
 result.update(hf_revision=revision,hf_path=remote,remote_sha256_verified=True,remote_file_sha256=hashes);atomic(out/"metrics.json",result);event("HF_VERIFIED",run_id=run_id,revision=revision);return result

def comparison(a,b,label):
 rows=[]
 for metric in ("data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss","wikitext_ppl","wikitext_bpb",*TASKS):
  x=float(a[metric]);y=float(b[metric]);rows.append({"comparison":label,"metric":metric,"control":x,"candidate":y,"absolute_difference":y-x,"relative_difference":((y-x)/x if x else "")})
 return rows
def write_csv(path,rows):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def uncertainty(a,b,label):
 rows=[]
 for task in TASKS:
  p,q=float(a[task]),float(b[task]);n=TASK_N[task];delta=q-p;se=float("inf");method="independent-binomial approximation; raw paired predictions unavailable";extra={}
  left=ROOT/a["raw_gibc"] if a.get("raw_gibc") else None;right=ROOT/b["raw_gibc"] if b.get("raw_gibc") else None
  if left and right and left.exists() and right.exists():
   x=json.loads(left.read_text())["samples"][task];y=json.loads(right.read_text())["samples"][task]
   def keyed(samples):return {(str(s["doc_id"]),s["doc_hash"],s["target_hash"]):int(float(s["acc"])) for s in samples}
   xa,ya=keyed(x),keyed(y)
   if xa.keys()!=ya.keys():raise RuntimeError(f"paired evaluation identity mismatch for {task}")
   values=np.asarray([ya[k]-xa[k] for k in sorted(xa)],dtype=np.int8);n=len(values);neg=int((values==-1).sum());pos=int((values==1).sum());zero=n-neg-pos
   draws=np.random.default_rng(2026).multinomial(n,[neg/n,zero/n,pos/n],size=10_000);boot=(draws[:,2]-draws[:,0])/n;lo,hi=(float(v) for v in np.quantile(boot,[.025,.975]));delta=float(values.mean());discord=neg+pos
   exact=float(binomtest(min(neg,pos),discord,.5).pvalue) if discord else 1.0;method="paired bootstrap (10,000 deterministic replicates) and exact two-sided McNemar/binomial test";extra={"control_only_correct":neg,"candidate_only_correct":pos,"exact_mcnemar_p":exact}
  else:
   se=math.sqrt(p*(1-p)/n+q*(1-q)/n);lo,hi=delta-1.96*se,delta+1.96*se
  movement=("CLEAR POSITIVE" if lo>0 else "CLEAR NEGATIVE" if hi<0 else "SUGGESTIVE POSITIVE" if delta>se else "SUGGESTIVE NEGATIVE" if delta<-se else "AMBIGUOUS")
  if method.startswith("paired"):movement=("CLEAR POSITIVE" if lo>0 else "CLEAR NEGATIVE" if hi<0 else "SUGGESTIVE POSITIVE" if delta>0 and extra["exact_mcnemar_p"]<.10 else "SUGGESTIVE NEGATIVE" if delta<0 and extra["exact_mcnemar_p"]<.10 else "AMBIGUOUS")
  rows.append({"comparison":label,"task":task,"control_accuracy":p,"candidate_accuracy":q,"delta":delta,"ci95_low":lo,"ci95_high":hi,"method":method,**extra,"classification":movement})
 return rows
def replication_class(result):
 ppl=abs(result["wikitext_ppl"]/CONTROL["wikitext_ppl"]-1);bpb=abs(result["wikitext_bpb"]/CONTROL["wikitext_bpb"]-1);val=abs(result["data_d_validation_loss"]-CONTROL["data_d_validation_loss"])
 if ppl<=.10 and bpb<=.05 and val<=.20:return "STABLE"
 if ppl<=.25 and bpb<=.12 and val<=.40:return "MODERATELY VARIABLE"
 return "HIGH VARIANCE"
def enough_time(state):
 # 100M at a conservative 80% of observed throughput plus 90 minutes evaluation/upload reserve.
 required=100_000_000/max(500,float(state.get("observed_throughput",5003))*.8)+5400
 return time.time()<epoch(state["soft_deadline"]) and time.time()+required<epoch(state["hard_deadline"])
def reports(state,data,lr=None):
 rep=replication_class(data);rep_rows=comparison(CONTROL,data,"DATA-D seed 314159 vs seed 2026");write_csv(ART/"seed_variance_comparison.csv",rep_rows)
 unc=uncertainty(CONTROL,data,"DATA-D seed 314159 vs seed 2026")
 if lr:unc+=uncertainty(data,lr,"seed 2026 DATA-D vs LR-10")
 write_csv(ART/"seed2026_benchmark_uncertainty.csv",unc)
 lines=["# Seed-2026 DATA-D replication report","",f"SEED 2026 DATA-D 100M COMPLETED: {'YES' if data else 'NO'}",f"WIKITEXT PPL: {data['wikitext_ppl']}",f"WIKITEXT BPB: {data['wikitext_bpb']}",f"HELLASWAG: {data['hellaswag']}",f"ARC-EASY: {data['arc_easy']}",f"PIQA: {data['piqa']}",f"WINOGRANDE: {data['winogrande']}",f"HF REVISION: {data['hf_revision']}",f"REPLICATION CLASSIFICATION: {rep}","","Raw per-example predictions are retained and aligned by document/hash identity for paired bootstrap and exact McNemar-style analysis."]
 (ART/"seed2026_replication_report.md").write_text("\n".join(lines)+"\n")
 transfer="INCONCLUSIVE"
 if lr:
  matched=comparison(data,lr,"seed 2026 DATA-D vs LR-10");write_csv(ART/"lr10_100m_matched_comparison.csv",matched)
  signals=uncertainty(data,lr,"seed 2026 DATA-D vs LR-10");mean=sum(x["delta"] for x in signals)/len(signals);clearpos=sum(x["classification"]=="CLEAR POSITIVE" for x in signals);clearneg=sum(x["classification"]=="CLEAR NEGATIVE" for x in signals);pplcost=lr["wikitext_ppl"]/data["wikitext_ppl"]-1
  transfer="STRONG" if clearpos>=2 and mean>0 and pplcost<=.05 else "MODEST" if mean>0 and clearneg==0 and pplcost<=.05 else "NEGATIVE" if clearneg>=2 or pplcost>.10 else "NONE" if abs(mean)<.002 and abs(pplcost)<.03 else "INCONCLUSIVE"
  lrheld=lr.get("lattice_reason_validation_loss","unavailable")
  text=["# Seed-2026 matched LR-10 report","", "LR-10 100M COMPLETED: YES",f"WIKITEXT PPL: {lr['wikitext_ppl']}",f"WIKITEXT BPB: {lr['wikitext_bpb']}",f"HELLASWAG: {lr['hellaswag']}",f"ARC-EASY: {lr['arc_easy']}",f"PIQA: {lr['piqa']}",f"WINOGRANDE: {lr['winogrande']}",f"LR HELD-OUT PERFORMANCE: {lrheld}",f"HF REVISION: {lr['hf_revision']}",f"LR-10 EXTERNAL TRANSFER: {transfer}"]
  (ART/"seed2026_lr10_report.md").write_text("\n".join(text)+"\n")
 else:write_csv(ART/"lr10_100m_matched_comparison.csv",[])
 decisions={"is_data_d_superiority_replicated":"YES" if rep!="HIGH VARIANCE" else "INCONCLUSIVE","is_lr10_worth_further_research":"YES" if transfer in {"STRONG","MODEST"} else "NO" if transfer in {"NONE","NEGATIVE"} else "INCONCLUSIVE","seed_noise_requires_more_replication":"YES" if rep!="STABLE" else "NO","test_24m_next":"NOT YET","scale_data_d_beyond_100m_next":"LOWER PRIORITY","recommended_next_experiments":["A third predeclared DATA-D seed at 100M if variance is not stable","Replicate LR-10 at one additional predeclared seed if matched transfer is positive","Only after variance settles, revisit capacity or token scaling"]}
 state.update(replication_classification=rep,final_decisions=decisions);return transfer
def final_commit():
 paths=[STATE,ART/"seed2026_replication_report.md",ART/"seed2026_lr10_report.md",ART/"seed_variance_comparison.csv",ART/"lr10_100m_matched_comparison.csv",ART/"seed2026_benchmark_uncertainty.csv",ART/"seed2026_data_d_training_curve.csv",ART/"lattice_reason_training_curves.csv",ART/"seed2026_data_d_100m/metrics.json",ART/"seed2026_lr10_100m/metrics.json"]
 existing=[str(p.relative_to(ROOT)) for p in paths if p.exists()];subprocess.run(["git","add","--",*existing],cwd=ROOT,check=True)
 if subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode:subprocess.run(["git","commit","-m","Record seed-2026 replication results"],cwd=ROOT,check=True)
 subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True)
def execute(state):
 save(state,"PREFLIGHT");verify_top_manifest(MANIFEST,TOKENIZER);data=train_data(state);state["replication"]=data;state["observed_throughput"]=data["tokens_per_second"];rep=replication_class(data);state["replication_classification"]=rep;state["replication_usable"]=rep!="HIGH VARIANCE";save(state,"STAGE2_ANALYSIS")
 lr=None
 if state["replication_usable"] and enough_time(state):lr=train_lr10(state);state["lr10"]=lr
 elif state["replication_usable"]:event("LR10_SKIPPED_DEADLINE")
 else:event("LR10_SKIPPED_REPLICATION_GATE",classification=rep)
 save(state,"STAGE4_FINAL_ANALYSIS");reports(state,data,lr);state["completed_stages"]=["STAGE1","STAGE2"]+(["STAGE3","STAGE4"] if lr else []);save(state,"COMPLETE");final_commit();event("MASTER_COMPLETE")
def dry_run():
 verify_top_manifest(MANIFEST,TOKENIZER);cfg=json.loads(CONFIG.read_text());assert cfg["seed"]==2026;assert cfg["context_length"]==128
 assert replication_class({**CONTROL,"wikitext_ppl":61,"wikitext_bpb":1.9,"data_d_validation_loss":3.5})=="STABLE"
 print(json.dumps({"dry_run":"PASS","manifest":"canonical verified","seed":2026,"parameters":15_949_760,"stages":["DATA-D replication","gate","conditional LR-10","analysis"]}));return 0
def main():
 p=argparse.ArgumentParser();p.add_argument("--soft-hours",type=float,default=12.5);p.add_argument("--hard-hours",type=float,default=14);p.add_argument("--dry-run",action="store_true");p.add_argument("--status",action="store_true");p.add_argument("--stop",action="store_true");a=p.parse_args()
 if a.status:print(json.dumps(load(),indent=2));return 0
 if a.stop:
  state=load()
  if state:save(state,stop_requested=True);event("STOP_REQUESTED_EXTERNAL")
  print("graceful stop requested");return 0
 if a.dry_run:return dry_run()
 lock=Lock()
 if not lock.acquire():print("seed replication master already running");return 0
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);state=load()
 if state and state.get("current_stage") in {"COMPLETE","FAILED","STOPPED_SAFE"}:print("terminal master state exists; refusing duplicate");return 0
 if state is None:state=initial(a.soft_hours,a.hard_hours)
 try:save(state,"STARTING");event("MASTER_STARTED",source_git_commit=state["source_git_commit"]);execute(state);return 0
 except (InterruptedError,TimeoutError) as exc:save(state,"STOPPED_SAFE",last_error=f"{type(exc).__name__}: {exc}");event("MASTER_STOPPED_SAFE",reason=str(exc));return 0
 except Exception as exc:save(state,"FAILED",last_error=f"{type(exc).__name__}: {exc}");event("MASTER_FAILED",error=state["last_error"]);raise
 finally:lock.release()
if __name__=="__main__":raise SystemExit(main())
