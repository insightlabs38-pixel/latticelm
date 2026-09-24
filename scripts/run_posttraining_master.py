#!/usr/bin/env python3
"""Autonomous, restart-safe final LatticeLM post-training master."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";PT=ART/"posttraining";STATE=PT/"master_state.json";PREVIOUS=PT/"master_state.previous.json";LOCK=PT/"master.lock";EVENTS=PT/"events.jsonl"
PRODUCTION_STATE=ART/"final_production_master_state.json";TOKENIZER=ART/"tokenizers/final_corpus_4k.json";TOKENIZER_REPORT=ART/"tokenizers/final_corpus_4k.report.json";MANIFEST=ART/"data/data_d_v4/canonical-2250m/manifest.json"
EXPERIMENT_CUTOFF=datetime(2026,9,30,15,0,tzinfo=ZoneInfo("America/New_York")).timestamp();HARD_CUTOFF=datetime(2026,9,30,16,0,tzinfo=ZoneInfo("America/New_York")).timestamp();SAFETY=float(os.environ.get("LATTICELM_POSTTRAIN_SAFETY","1.30"));FINAL_RESERVE=6*3600
PHASES=("WAIT_FOR_PRODUCTION_COMPLETE","CERTIFY_AND_FREEZE_BASE","POSTTRAINING_PREFLIGHT","LATTICEREASON_V2_BUILD","LATTICEREASON_V2_AUDIT","BASE_CAPABILITY_MAP","ROLLOUT_BACKEND_CERTIFICATION","OPTIMIZER_SCREEN","LR_SCREEN","REPLAY_SCREEN","SFT_TRUNK","RANKING_TOURNAMENT","JOINT_TOURNAMENT","RFT_ELIGIBILITY","RFT_TOURNAMENT","SIMPO_ELIGIBILITY","SIMPO_TOURNAMENT","RLVR_ELIGIBILITY","RLVR_PILOTS","RLVR_EXTENSION","RECOVERY_ANNEAL","INTERPOLATION","ADAPTIVE_RESEARCH","FINAL_CANDIDATE_TOURNAMENT","FINAL_SELECTION","FINAL_EXPORT","FINAL_EVIDENCE_PACKAGE","POSTTRAINING_COMPLETE")
STOP=False;CHILD=None
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
 return h.hexdigest()
def atomic_json(path,value,backup=None,immutable=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);encoded=(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n").encode()
 if immutable and path.exists():
  if path.read_bytes()!=encoded:raise RuntimeError(f"immutable artifact conflict: {path}")
  return
 tmp=path.with_name("."+path.name+".tmp")
 with tmp.open("wb") as f:f.write(encoded);f.flush();os.fsync(f.fileno())
 if backup and path.exists():shutil.copy2(path,backup)
 os.replace(tmp,path)
class Lock:
 def __init__(self,path):self.path=Path(path);self.file=None
 def __enter__(self):
  self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open("a+");fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB);return self
 def __exit__(self,*_):fcntl.flock(self.file,fcntl.LOCK_UN);self.file.close()
def event(kind,**fields):
 PT.mkdir(parents=True,exist_ok=True);row={"at":time.time(),"at_utc":datetime.now(timezone.utc).isoformat(),"event":kind,**fields}
 with EVENTS.open("a") as f:f.write(json.dumps(row,sort_keys=True,default=str)+"\n");f.flush();os.fsync(f.fileno())
def initial():return {"schema":"posttraining-master-state-v1","phase":PHASES[0],"completed":[],"terminal_state":None,"created_at":time.time(),"experiment_cutoff":"2026-09-30T15:00:00-04:00","hard_cutoff":"2026-09-30T16:00:00-04:00","candidates":{"BASE":{"status":"PENDING"}},"gibc_evaluations":0,"throughput":{},"optional_failures":[]}
def load():
 for p in (STATE,PREVIOUS):
  try:
   x=json.loads(p.read_text())
   if x.get("schema")=="posttraining-master-state-v1":return x
  except Exception:pass
 return initial()
def save(s,phase=None,**fields):
 if phase:s["phase"]=phase
 s.update(fields);s["updated_at"]=time.time();atomic_json(STATE,s,PREVIOUS);event("STATE",phase=s["phase"])
def complete(s,phase,**fields):
 if phase not in s["completed"]:s["completed"].append(phase)
 save(s,PHASES[min(PHASES.index(phase)+1,len(PHASES)-1)],**fields)
def signal_handler(*_):
 global STOP
 STOP=True
 if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
 else:raise SystemExit(75)
def production_processes():
 out=subprocess.run(["ps","-eo","args="],text=True,capture_output=True).stdout
 return [x for x in out.splitlines() if ("train_final_production.py" in x or "run_final_production_master.py" in x) and "run_posttraining_master" not in x]
def finalization_reserve(s):
 measured=s.get("finalization_seconds",{});return max(FINAL_RESERVE,sum(float(measured.get(x,default)) for x,default in (("proxy_ranking",3600),("official_evals",10800),("selection",600),("merging",1800),("export",3600),("evidence",1200))))
def research_deadline(s,now=None):
 now=time.time() if now is None else now
 return max(now,min(EXPERIMENT_CUTOFF-finalization_reserve(s),EXPERIMENT_CUTOFF))
def experiment_allowed(predicted_seconds=0,s=None,now=None):
 now=time.time() if now is None else now;limit=research_deadline(s or {},now)
 return now<EXPERIMENT_CUTOFF and now+predicted_seconds*SAFETY<limit
def schedule_decision(s,name,tokens,minimum=1_000_000,standard=None,maximum=None):
 tps=float(s["throughput"].get(name,s["throughput"].get("sft",50.0)));standard=standard or tokens;maximum=maximum or standard;now=time.time();remaining=max(0,research_deadline(s,now)-now);fit=int(remaining*tps/SAFETY);budget=min(maximum,fit,standard) if fit>=minimum else 0
 return {"objective":name,"minimum":minimum,"standard":standard,"maximum":maximum,"selected":budget,"measured_logical_tokens_per_second":tps,"safety_factor":SAFETY,"seconds_to_experiment_cutoff":max(0,EXPERIMENT_CUTOFF-now),"seconds_for_research":remaining,"finalization_reserve":finalization_reserve(s),"estimated_seconds":budget/max(tps,1e-9)*SAFETY}
def command(s,label,args,optional=False,timeout=None):
 global CHILD
 log=PT/"logs"/f"{label}.log";log.parent.mkdir(parents=True,exist_ok=True);event("CHILD_START",label=label,command=[str(x) for x in args]);started=time.perf_counter()
 with log.open("a") as f:
  CHILD=subprocess.Popen([str(x) for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);save(s,child_pid=CHILD.pid,child_label=label);code=CHILD.wait(timeout=timeout);CHILD=None;save(s,child_pid=None)
 elapsed=time.perf_counter()-started;event("CHILD_END",label=label,exit_code=code,seconds=elapsed)
 if code:
  if optional:s["optional_failures"].append({"label":label,"exit_code":code,"log":str(log),"at":time.time()});save(s);return False
  raise RuntimeError(f"required child failed: {label} ({code})")
 return True
def worker(s,cid,parent,method,tokens,optimizer="muon_hybrid",lr=3e-5,replay=.2,optional=False,wall_limit_seconds=None,**options):
 out=PT/"candidates"/cid;prior=json.loads((out/"result.json").read_text()) if (out/"result.json").exists() else None
 if prior and int(prior["training_tokens"])>=tokens:return prior
 plan=schedule_decision(s,method,tokens,min(tokens,100_000),tokens,tokens);event("SCHEDULE_DECISION",candidate_id=cid,plan=plan)
 if wall_limit_seconds is not None:
  fit=int(max(0,wall_limit_seconds)*plan["measured_logical_tokens_per_second"]/SAFETY);plan["selected"]=min(plan["selected"],fit);plan["wall_limit_seconds"]=wall_limit_seconds
 if not plan["selected"]:
  event("EXPERIMENT_SKIPPED_CUTOFF",candidate_id=cid,plan=plan);return None
 parent_sha=sha256(parent);parent_candidate=next((x for x in s["candidates"].values() if x.get("checkpoint_sha256")==parent_sha),None);chain=list(parent_candidate.get("method_chain",[]) if parent_candidate else [])+[method]
 args=[sys.executable,"scripts/train_posttraining_worker.py","--base",parent,"--output",out,"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--method",method,"--tokens",plan["selected"],"--optimizer",optimizer,"--lr",lr,"--replay",replay,"--threads",16,"--stop-epoch",min(EXPERIMENT_CUTOFF,research_deadline(s),time.time()+wall_limit_seconds if wall_limit_seconds is not None else HARD_CUTOFF),"--method-chain",json.dumps(chain)]
 if method in ("sft","recovery") and "compile_mode" not in options:options["compile_mode"]=int(s.get("selected_compile",False))
 if method in ("rlvr-grpo","rlvr-rloo","rft") and s.get("rollout_backend")=="CERTIFIED":args.extend(("--rollout-topology",PT/"rollout_backend_certification.json"))
 for key,value in options.items():args.extend(("--"+key.replace("_","-"),value))
 started=time.perf_counter();ok=command(s,cid,args,optional=optional)
 if not ok:return None
 if not (out/"result.json").exists():
  event("EXPERIMENT_NO_VALID_CANDIDATE",candidate_id=cid,evidence=str(out/"ineligible.json"));return None
 result=json.loads((out/"result.json").read_text());elapsed=max(time.perf_counter()-started,1e-9);delta=result["training_tokens"]-(prior["training_tokens"] if prior else 0);rate=max(delta,0)/elapsed
 if rate>0:s["throughput"][method]=rate
 result["observed_logical_tokens_per_second"]=rate;result["observed_full_tokens_per_second"]=(result.get("processed_tokens",0)-(prior.get("processed_tokens",0) if prior else 0))/elapsed;s["candidates"][cid]={**result,"candidate_id":cid,"optimizer":optimizer,"lr":lr,"replay":replay,"status":result["status"],"method_chain":chain,"requested_tokens":tokens,"elapsed_seconds":elapsed};save(s);return result
def base_path(s):return Path(s["base"]["checkpoint"])
def phase_wait(s):
 try:p=json.loads(PRODUCTION_STATE.read_text())
 except Exception:return False
 if p.get("terminal_state")!="PRODUCTION_COMPLETE":return False
 if production_processes():return False
 complete(s,"WAIT_FOR_PRODUCTION_COMPLETE",production_handoff={"state":str(PRODUCTION_STATE),"sha256":sha256(PRODUCTION_STATE),"observed_at":time.time()});return True
def resolve_final():
 result=json.loads((ART/"final_submission_results.json").read_text());candidates=[result.get("metrics",{}).get("checkpoint"),json.loads((ART/"final_production_run/result.json").read_text()).get("checkpoint")]
 cp=next((Path(x) for x in candidates if x and Path(x).is_file()),None)
 if cp is None:raise RuntimeError("authoritative final checkpoint path is ambiguous")
 return cp,result
def phase_certify(s):
 import torch
 from latticelm.config import LatticeConfig
 from latticelm.model import build_model
 if production_processes():raise RuntimeError("production process remains after semantic handoff")
 cp,result=resolve_final();digest=sha256(cp)
 if digest!=result["checkpoint_sha256"]:raise RuntimeError("final result checkpoint SHA mismatch")
 side=cp.with_suffix(".sha256")
 if side.exists() and side.read_text().strip()!=digest:raise RuntimeError("checkpoint sidecar mismatch")
 payload=torch.load(cp,map_location="cpu",weights_only=False);cfg=LatticeConfig(**payload["config"]);model=build_model(cfg);model.load_state_dict(payload["model"],strict=True);params=sum(x.numel() for x in model.parameters() if x.requires_grad);expected={"architecture":"co4_causal","n_layers":12,"d_model":552,"ffn_hidden":1728,"n_heads":6,"n_kv_heads":2,"context_length":256,"vocab_size":4096,"qk_norm":True,"real_gqa":True}
 if params!=48_636_168 or params>=50_000_000 or any(getattr(cfg,k)!=v for k,v in expected.items()):raise RuntimeError("base architecture/parameter identity failure")
 if payload["tokenizer_sha256"]!=sha256(TOKENIZER) or payload["manifest_sha256"]!=sha256(MANIFEST):raise RuntimeError("base tokenizer/dataset identity failure")
 for value in (payload.get("validation_loss"),payload.get("train_loss")):
  if value is not None and not math.isfinite(float(value)):raise RuntimeError("nonfinite base evidence")
 out=PT/"baseline";out.mkdir(parents=True,exist_ok=True);frozen=out/"base.pt"
 if not frozen.exists():os.link(cp,frozen)
 if sha256(frozen)!=digest:raise RuntimeError("immutable BASE collision")
 (out/"base.sha256").write_text(digest+"\n");identity={"candidate_id":"BASE","checkpoint":str(frozen),"checkpoint_sha256":digest,"source_checkpoint":str(cp),"config":payload["config"],"parameter_count":params,"tokenizer_sha256":sha256(TOKENIZER),"manifest_sha256":sha256(MANIFEST),"parent_lineage":None,"method_chain":[]};atomic_json(out/"identity.json",identity,immutable=True);s["candidates"]["BASE"]={**identity,"status":"CERTIFIED"};complete(s,"CERTIFY_AND_FREEZE_BASE",base=identity)
def phase_preflight(s):
 root=PT/"preflight";root.mkdir(parents=True,exist_ok=True);worker(s,"preflight-sft",base_path(s),"sft",256,"adamw",1e-5,.2);result=worker(s,"preflight-sft",base_path(s),"sft",512,"adamw",1e-5,.2);ranking=worker(s,"preflight-ranking",base_path(s),"ranking",64,"adamw",1e-5,0.)
 if not result:raise RuntimeError("post-training preflight could not run")
 command(s,"preflight-evaluator",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",result["checkpoint"],"--tokenizer",TOKENIZER,"--output",root/"evaluator.json","--tasks","piqa","--limit","1","--threads","1","--batch-size","1"])
 if sha256(base_path(s))!=s["base"]["checkpoint_sha256"]:raise RuntimeError("preflight mutated BASE")
 systems=PT/"systems_benchmark.json";command(s,"posttraining-systems",[sys.executable,"scripts/benchmark_posttraining_systems.py","--checkpoint",base_path(s),"--output",systems],optional=True)
 benchmark=json.loads(systems.read_text()) if systems.exists() else {};s["selected_microbatch"]=benchmark.get("selected_microbatch",1);s["selected_compile"]=benchmark.get("compile",{}).get("selected",False)
 atomic_json(root/"result.json",{"status":"PASS","fresh_optimizer":True,"completion_masks":True,"ranking_branch":bool(ranking),"checkpoint_restart":True,"next_batch_identity":True,"parent_immutable":True,"evaluator_invocation":True,"branch_sha":result["checkpoint_sha256"],"systems_benchmark":benchmark});complete(s,"POSTTRAINING_PREFLIGHT")
def phase_build(s):
 from latticelm.lattice_reason_v2 import generate_example,verify_example
 # Benchmark only after production; choose 150M..1B with disk/time safeguards.
 start=time.perf_counter()
 for i in range(2000):verify_example(generate_example(i))
 eps=2000/max(time.perf_counter()-start,1e-9);estimated_tps=eps*100
 from latticelm.lattice_reason_v2.dataset import adaptive_target
 target=adaptive_target(estimated_tps,4.2,shutil.disk_usage(ROOT).free,max(0,EXPERIMENT_CUTOFF-time.time()));registry=PT/"lattice_reason_v2/contamination_registry.json";command(s,"v2-contamination-registry",[sys.executable,"scripts/build_posttraining_contamination_registry.py","--output",registry]);command(s,"build-v2",[sys.executable,"scripts/build_lattice_reason_v2.py","--output",PT/"lattice_reason_v2","--tokenizer",TOKENIZER,"--registry",registry,"--target-tokens",target]);command(s,"build-v2-natural-bank",[sys.executable,"scripts/build_natural_bank_v2.py","--manifest",MANIFEST,"--tokenizer",TOKENIZER,"--output",PT/"lattice_reason_v2/natural_bank","--examples",100000]);complete(s,"LATTICEREASON_V2_BUILD",reservoir_target=target,generator_examples_per_second=eps,contamination_registry=str(registry),natural_bank=str(PT/"lattice_reason_v2/natural_bank/manifest.json"))
def phase_audit(s):
 from latticelm.lattice_reason_v2 import generate_example
 from latticelm.lattice_reason_v2.audit import shortcut_audit
 rows=[generate_example(i,split="validation") for i in range(4096)];audit=shortcut_audit(rows,tolerance=.15);atomic_json(PT/"lattice_reason_v2/audit.json",audit)
 if audit["status"]!="PASS":raise RuntimeError("v2 shortcut certification failed")
 complete(s,"LATTICEREASON_V2_AUDIT",audit=audit)
def phase_map(s):
 out=PT/"capability_map/base.json";command(s,"base-capability-map",[sys.executable,"scripts/evaluate_lattice_reason_v2.py","--checkpoint",base_path(s),"--tokenizer",TOKENIZER,"--output",out,"--examples",512,"--max-k",8,"--threads",16]);complete(s,"BASE_CAPABILITY_MAP",base_capability_map=str(out))
def phase_rollout(s):
 out=PT/"rollout_backend_certification.json";command(s,"rollout-certification",[sys.executable,"scripts/certify_rollout_backend.py","--checkpoint",base_path(s),"--tokenizer",TOKENIZER,"--output",out],optional=True)
 status="CERTIFIED" if out.exists() and json.loads(out.read_text()).get("status")=="PASS" else "REJECTED_OPTIONAL";complete(s,"ROLLOUT_BACKEND_CERTIFICATION",rollout_backend=status,rollout_topology=json.loads(out.read_text()).get("selected") if out.exists() else None)
def proxy(s,cid,examples=256):
 out=PT/"candidates"/cid/f"proxy-{examples}.json";cp=s["candidates"][cid]["checkpoint"];args=[sys.executable,"scripts/evaluate_posttraining_proxy.py","--checkpoint",cp,"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--output",out,"--examples",examples,"--threads",16]
 if cid!="BASE" and s["candidates"][cid].get("parent_checkpoint_sha256"):
  parent=next((x for x in s["candidates"].values() if x.get("checkpoint_sha256")==s["candidates"][cid]["parent_checkpoint_sha256"]),None)
  if parent:args.extend(("--parent",parent["checkpoint"]))
 command(s,cid+f"-proxy-{examples}",args,optional=True);return json.loads(out.read_text()) if out.exists() else None
def choose_screen(s,prefix,field,default):
 rows=[]
 for cid,x in s["candidates"].items():
  if cid.startswith(prefix) and x.get("status") in ("COMPLETE","SAFE_STOPPED_VALID","TRAINED"):
   x["proxy"]=x.get("proxy") or proxy(s,cid)
   if x["proxy"]:rows.append(x)
 if not rows:return default
 return max(rows,key=lambda x:(x["proxy"]["v2_ranking_accuracy"],x["proxy"]["v2_mean_margin"],-x["proxy"]["data_d_validation"],x.get("observed_logical_tokens_per_second",0)))[field]
def phase_optimizer(s):
 for opt in ("muon_hybrid","adamw"):
  cid="screen-opt-"+opt;result=worker(s,cid,base_path(s),"sft",150_000,opt,3e-5,.2,True,microbatch=s.get("selected_microbatch",1))
  if result:s["candidates"][cid]["proxy"]=proxy(s,cid)
 selected=choose_screen(s,"screen-opt","optimizer","muon_hybrid");worker(s,"screen-opt-"+selected,base_path(s),"sft",500_000,selected,3e-5,.2,True,microbatch=s.get("selected_microbatch",1));complete(s,"OPTIMIZER_SCREEN",selected_optimizer=selected)
def phase_lr(s):
 for lr in (1e-5,3e-5,1e-4):worker(s,f"screen-lr-{lr:g}",base_path(s),"sft",150_000,s["selected_optimizer"],lr,.2,True,microbatch=s.get("selected_microbatch",1))
 selected=choose_screen(s,"screen-lr","lr",3e-5);worker(s,f"screen-lr-{selected:g}",base_path(s),"sft",500_000,s["selected_optimizer"],selected,.2,True,microbatch=s.get("selected_microbatch",1));complete(s,"LR_SCREEN",selected_lr=selected)
def phase_replay(s):
 for replay in (0.,.2,.4):worker(s,f"screen-replay-{int(replay*100)}",base_path(s),"sft",150_000,s["selected_optimizer"],s["selected_lr"],replay,True,microbatch=s.get("selected_microbatch",1))
 selected=choose_screen(s,"screen-replay","replay",.2);worker(s,f"screen-replay-{int(selected*100)}",base_path(s),"sft",500_000,s["selected_optimizer"],s["selected_lr"],selected,True,microbatch=s.get("selected_microbatch",1));complete(s,"REPLAY_SCREEN",selected_replay=selected)
def phase_sft(s):
 cid="sft-trunk";result=s["candidates"].get(cid);history=s.get("sft_dose_response",[]);base_proxy=proxy(s,"BASE");sft_deadline=s.get("sft_branch_deadline") or time.time()+.30*max(0,research_deadline(s)-time.time());s["sft_branch_deadline"]=sft_deadline;save(s)
 for budget in (1_000_000,5_000_000,10_000_000,20_000_000,30_000_000,50_000_000,75_000_000,100_000_000,150_000_000):
  if any(x.get("requested_budget")==budget for x in history):continue
  if time.time()>=sft_deadline:break
  if budget>30_000_000 and not experiment_allowed(budget/max(s["throughput"].get("sft",50),1),s):break
  result=worker(s,cid,base_path(s),"sft",budget,s["selected_optimizer"],s["selected_lr"],s["selected_replay"],True,wall_limit_seconds=sft_deadline-time.time(),microbatch=s.get("selected_microbatch",1))
  if result is None:break
  metrics=proxy(s,cid,1024 if budget>=5_000_000 else 256);history.append({"requested_budget":budget,"tokens":result["training_tokens"],"metrics":metrics});save(s,sft_dose_response=history)
  if result["status"]=="SAFE_STOPPED_VALID":break
  if len(history)>=3 and all(x["metrics"] and x["metrics"].get("examples",0)>=1024 for x in history[-3:]) and all(x["metrics"]["v2_mean_margin"]<=history[-3]["metrics"]["v2_mean_margin"] for x in history[-2:]):break
 complete(s,"SFT_TRUNK",sft_winner=cid if result else "BASE",sft_dose_response=history)
def simple_branch_phase(s,phase,method,cid,parent_key,budget,optional=True):
 parent=base_path(s) if parent_key=="BASE" else Path(s["candidates"].get(parent_key,s["candidates"]["BASE"])["checkpoint"]);result=worker(s,cid,parent,method,budget,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),optional,microbatch=s.get("selected_microbatch",1) if method in ("sft","recovery") else 1);complete(s,phase,**{phase.lower():cid if result else "SKIPPED"})
def strongest_candidate(s,methods=None):
 rows=[]
 for cid,c in s["candidates"].items():
  if c.get("status") not in ("COMPLETE","SAFE_STOPPED_VALID","TRAINED","CERTIFIED"):continue
  if methods and c.get("method") not in methods:continue
  p=c.get("proxy") or proxy(s,cid)
  if p:rows.append((cid,p))
 if not rows:return "BASE"
 return max(rows,key=lambda row:(row[1]["v2_ranking_accuracy"],row[1]["v2_mean_margin"],-row[1]["data_d_validation"]))[0]
def phase_ranking(s):
 parent=s.get("sft_winner","BASE");path=s["candidates"][parent]["checkpoint"];rows=[]
 for mode,natural,margin in (("ranking_raw",.2,0.),("ranking_norm",.5,0.),("ranking_dual",.8,0.),("ranking_dual",.5,.2)):
  cid=f"ranking-screen-{mode}-{int(natural*100)}-{int(margin*10)}"
  result=worker(s,cid,path,"ranking",150_000,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),True,ranking_mode=mode,natural_fraction=natural,margin_weight=margin)
  if result:
   p=proxy(s,cid,256);s["candidates"][cid]["proxy"]=p
   if p:rows.append(cid)
 if rows:
  best=strongest_candidate(s,{"ranking"});parent=s["candidates"][best]["checkpoint"]
  result=worker(s,"ranking",parent,"ranking",5_000_000,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),True,ranking_mode=s["candidates"][best].get("ranking_mode","ranking_dual"),natural_fraction=s["candidates"][best].get("natural_fraction",.5),margin_weight=s["candidates"][best].get("margin_weight",0.))
  if result:s["candidates"]["ranking"]["proxy"]=proxy(s,"ranking",1024)
 complete(s,"RANKING_TOURNAMENT",ranking_screen=rows,ranking_winner="ranking" if "ranking" in s["candidates"] else "SKIPPED")
def phase_joint(s):
 parent=strongest_candidate(s,{"sft","ranking"});path=s["candidates"][parent]["checkpoint"];rows=[]
 for weight in (.25,.75):
  cid=f"joint-screen-{int(weight*100)}";result=worker(s,cid,path,"joint",150_000,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),True,rank_weight=weight)
  if result:
   s["candidates"][cid]["proxy"]=proxy(s,cid,256);rows.append(cid)
 if rows:
  best=strongest_candidate(s,{"joint"});weight=s["candidates"][best]["rank_weight"]
  result=worker(s,"joint",Path(s["candidates"][best]["checkpoint"]),"joint",2_000_000,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),True,rank_weight=weight)
  if result:s["candidates"]["joint"]["proxy"]=proxy(s,"joint",1024)
 complete(s,"JOINT_TOURNAMENT",joint_screen=rows,joint_winner="joint" if "joint" in s["candidates"] else "SKIPPED")
def phase_rft(s):
 parent=strongest_candidate(s,{"sft","ranking","joint"});lineage=[];frontier=Path(s.get("rft_eligibility_map",s["base_capability_map"]))
 for iteration,budget in enumerate((1_000_000,2_500_000,5_000_000),1):
  cid=f"rft-iter-{iteration}";result=worker(s,cid,Path(s["candidates"][parent]["checkpoint"]),"rft",budget,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),True,rollout_k=8 if iteration==1 else 4,frontier_map=frontier)
  if not result:break
  p=proxy(s,cid,1024);s["candidates"][cid]["proxy"]=p;lineage.append({"candidate":cid,"parent":parent,"tokens":result["training_tokens"],"proxy":p});save(s,rft_iterations=lineage)
  if not p or result["status"]=="SAFE_STOPPED_VALID":break
  prior=s["candidates"][parent].get("proxy") or proxy(s,parent,1024)
  if prior and p["v2_mean_margin"]<=prior["v2_mean_margin"] and iteration>=2:break
  mapped=PT/"capability_map"/f"rft-iter-{iteration}.json"
  if experiment_allowed(1800,s) and command(s,f"rft-iter-{iteration}-map",[sys.executable,"scripts/evaluate_lattice_reason_v2.py","--checkpoint",result["checkpoint"],"--tokenizer",TOKENIZER,"--output",mapped,"--examples",256,"--max-k",8,"--threads",16],optional=True):frontier=mapped
  parent=cid
 complete(s,"RFT_TOURNAMENT",rft_iterations=lineage,rft_winner=parent if lineage else "SKIPPED")
def phase_rlvr_pilots(s):
 parent=strongest_candidate(s,{"sft","ranking","joint","rft"});rows=[]
 for method in ("rlvr-grpo","rlvr-rloo"):
  result=worker(s,method,Path(s["candidates"][parent]["checkpoint"]),method,1_000_000,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),True,rollout_k=4)
  if result:s["candidates"][method]["proxy"]=proxy(s,method,1024);rows.append(method)
 complete(s,"RLVR_PILOTS",rlvr_parent=parent,rlvr_pilots=rows)
def phase_rlvr_extension(s):
 rows=s.get("rlvr_pilots",[])
 if not rows:return complete(s,"RLVR_EXTENSION",rlvr_extension="SKIPPED")
 best=max(rows,key=lambda cid:(s["candidates"][cid].get("proxy") or {}).get("v2_mean_margin",-1e9));method=s["candidates"][best]["method"]
 result=worker(s,"rlvr-extension",Path(s["candidates"][best]["checkpoint"]),method,5_000_000,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),True,rollout_k=4)
 if result:s["candidates"]["rlvr-extension"]["proxy"]=proxy(s,"rlvr-extension",1024)
 complete(s,"RLVR_EXTENSION",rlvr_extension="rlvr-extension" if result else "SKIPPED",rlvr_pilot_winner=best)
def phase_capability_eligibility(s,phase):
 methods={"sft","ranking","joint"} if phase=="RFT_ELIGIBILITY" else {"sft","ranking","joint","rft","simpo"};parent=strongest_candidate(s,methods);out=PT/"capability_map"/(phase.lower()+".json")
 if not out.exists():command(s,phase.lower()+"-map",[sys.executable,"scripts/evaluate_lattice_reason_v2.py","--checkpoint",s["candidates"][parent]["checkpoint"],"--tokenizer",TOKENIZER,"--output",out,"--examples",256,"--max-k",8,"--threads",16],optional=True)
 regions=json.loads(out.read_text()).get("regions",{}) if out.exists() else json.loads(Path(s["base_capability_map"]).read_text()).get("regions",{});rates=[x.get("pass_at_8",0) for x in regions.values() if x.get("examples",0)>=3]
 learnable=any(0<rate<.95 for rate in rates);throughput=(s.get("rollout_topology") or {}).get("aggregate_tokens_per_second",0)
 eligible=s.get("rollout_backend")=="CERTIFIED" and learnable and (phase=="RFT_ELIGIBILITY" or throughput>0) and experiment_allowed(1800,s)
 event("CAPABILITY_ELIGIBILITY",phase=phase,parent=parent,eligible=eligible,region_rates=rates,rollout_tokens_per_second=throughput)
 complete(s,phase,**{("rft_eligible" if phase=="RFT_ELIGIBILITY" else "rlvr_eligible"):eligible,phase.lower()+"_parent":parent,phase.lower()+"_map":str(out) if out.exists() else s["base_capability_map"]})
def phase_recovery(s):
 base=proxy(s,"BASE");target=None
 for cid,c in s["candidates"].items():
  if cid=="BASE" or c.get("status") not in ("COMPLETE","SAFE_STOPPED_VALID"):continue
  p=c.get("proxy") or proxy(s,cid)
  if p and base and p["v2_mean_margin"]>base["v2_mean_margin"] and p["data_d_validation"]>base["data_d_validation"]*1.025:
   if target is None or p["v2_mean_margin"]>s["candidates"][target]["proxy"]["v2_mean_margin"]:target=cid
 if target:
  parent=s["candidates"][target]["checkpoint"];history=[]
  for dose in (1_000_000,2_500_000,5_000_000):
   result=worker(s,"recovery",parent,"recovery",dose,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),1.,True,microbatch=s.get("selected_microbatch",1))
   if not result:break
   p=proxy(s,"recovery",1024);history.append({"tokens":result["training_tokens"],"proxy":p});s["candidates"]["recovery"]["proxy"]=p;save(s,recovery_history=history)
   if not p or p["data_d_validation"]<=base["data_d_validation"]*1.025 or p["v2_mean_margin"]<s["candidates"][target]["proxy"]["v2_mean_margin"]:break
 complete(s,"RECOVERY_ANNEAL",recovery_parent=target,recovery_history=s.get("recovery_history",[]))
def phase_interpolation(s):
 import torch
 created=[];families=(("sft","sft"),("ranking","ranking"),("rft","rft"));best={name:strongest_candidate(s,{method}) for name,method in families}
 pairs=[]
 for name,cid in best.items():
  if cid!="BASE":pairs.extend((f"base-{name}-{alpha:.2f}","BASE",cid,alpha) for alpha in (.25,.5,.75))
 for left,right in (("sft","ranking"),("ranking","rft"),("ranking","recovery")):
  a=best.get(left,"BASE");b=best.get(right,"BASE") if right!="recovery" else ("recovery" if "recovery" in s["candidates"] else "BASE")
  if a!="BASE" and b!="BASE" and a!=b:pairs.append((f"specialist-{left}-{right}",a,b,.5))
 for cid,left,right,alpha in pairs:
  out=PT/"candidates"/cid;out.mkdir(parents=True,exist_ok=True);cp=out/"checkpoint.pt";parents=[s["candidates"][x] for x in (left,right)]
  if any(sha256(p["checkpoint"])!=p["checkpoint_sha256"] for p in parents):raise RuntimeError("merge parent SHA mismatch")
  if cp.exists():
   digest=sha256(cp)
   if digest!=(out/"checkpoint.sha256").read_text().strip():raise RuntimeError("merge artifact SHA mismatch")
  else:
   first=torch.load(parents[0]["checkpoint"],map_location="cpu",weights_only=False);second=torch.load(parents[1]["checkpoint"],map_location="cpu",weights_only=False)
   if first["config"]!=second["config"] or first["model"].keys()!=second["model"].keys():raise RuntimeError("merge identity mismatch")
   merged={k:first["model"][k].mul(1-alpha).add(second["model"][k],alpha=alpha) for k in first["model"]}
   if not all(torch.isfinite(v).all() for v in merged.values()):raise RuntimeError("nonfinite merge")
   payload={"schema":"posttraining-merged-checkpoint-v1","config":first["config"],"model":merged,"method":"merge","merge":{"parents":[p["checkpoint_sha256"] for p in parents],"coefficients":[1-alpha,alpha]}}
   tmp=out/".checkpoint.pt.tmp";torch.save(payload,tmp);os.replace(tmp,cp);digest=sha256(cp);(out/"checkpoint.sha256").write_text(digest+"\n")
  s["candidates"][cid]={"candidate_id":cid,"checkpoint":str(cp),"checkpoint_sha256":digest,"status":"TRAINED","method":"interpolation","method_chain":["merge"],"parents":[p["checkpoint_sha256"] for p in parents],"alpha":alpha};created.append(cid);save(s)
 complete(s,"INTERPOLATION",interpolations=created)
def adaptive_choice(s,now=None):
 now=time.time() if now is None else now;remaining=research_deadline(s,now)-now
 if remaining<1800:return {"action":"FINALIZE","reason":"estimated finalization reserve consumes remaining window","remaining_seconds":remaining}
 valid=[(cid,c) for cid,c in s["candidates"].items() if c.get("status") in ("COMPLETE","SAFE_STOPPED_VALID","TRAINED") and c.get("method") in ("sft","ranking","joint","rft") and not cid.startswith(("screen-","ranking-screen-","joint-screen-","preflight-"))]
 by_sha={c.get("checkpoint_sha256"):c for c in s["candidates"].values()}
 valid=[(cid,c) for cid,c in valid if not cid.startswith("adaptive-") or not by_sha.get(c.get("parent_checkpoint_sha256")) or (c.get("proxy") or {}).get("v2_mean_margin",-1e9)>(by_sha[c["parent_checkpoint_sha256"]].get("proxy") or {}).get("v2_mean_margin",-1e9)]
 ranked=sorted(valid,key=lambda x:(x[1].get("proxy") or {}).get("v2_mean_margin",-1e9),reverse=True)
 for cid,c in ranked:
  attempts=sum(x.get("parent")==cid for x in s.get("adaptive_decisions",[]))
  if attempts>=2:continue
  rate=s["throughput"].get(c["method"],0)
  if rate<=0:continue
  dose=max(250_000,min(10_000_000,int(remaining*rate/(SAFETY*3))))
  if dose/rate*SAFETY<remaining:return {"action":"CONTINUE","parent":cid,"method":c["method"],"tokens":dose,"reason":"strongest measured proxy branch with useful remaining dose","remaining_seconds":remaining,"measured_tokens_per_second":rate}
 return {"action":"FINALIZE","reason":"no eligible branch has measured throughput and expected useful continuation","remaining_seconds":remaining}
def phase_adaptive(s):
 decision=adaptive_choice(s);s.setdefault("adaptive_decisions",[]).append({**decision,"at":time.time()});event("ADAPTIVE_DECISION",**decision);save(s)
 if decision["action"]=="FINALIZE":return complete(s,"ADAPTIVE_RESEARCH",no_further_research_reason=decision["reason"])
 cid=f"adaptive-{len(s['adaptive_decisions']):02d}";parent=s["candidates"][decision["parent"]];method=decision["method"]
 result=worker(s,cid,Path(parent["checkpoint"]),method,decision["tokens"],parent.get("optimizer",s.get("selected_optimizer","muon_hybrid")),parent.get("lr",s.get("selected_lr",3e-5)),parent.get("replay",s.get("selected_replay",.2)),True,microbatch=parent.get("microbatch",1) if method=="sft" else 1,ranking_mode=parent.get("ranking_mode","ranking_dual"),natural_fraction=parent.get("natural_fraction",.2),margin_weight=parent.get("margin_weight",0.))
 if result:s["candidates"][cid]["proxy"]=proxy(s,cid,1024)
 else:s["adaptive_decisions"][-1]["failed_or_skipped"]=True
 save(s)
def eval_finalist(s,cid):
 c=s["candidates"][cid];root=PT/"candidates"/cid/"final_eval";root.mkdir(parents=True,exist_ok=True)
 if not(root/"wikitext.json").exists():command(s,cid+"-wiki",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",c["checkpoint"],"--tokenizer",TOKENIZER,"--output",root/"wikitext.json","--threads",16,"--batch-size",16])
 if not(root/"gibc.json").exists():
  if s["gibc_evaluations"]>=8:return None
  command(s,cid+"-gibc",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",c["checkpoint"],"--tokenizer",TOKENIZER,"--output",root/"gibc.json","--predictions-output",root/"predictions.json","--tasks","hellaswag,arc_easy,piqa,winogrande","--threads",16,"--batch-size",16]);s["gibc_evaluations"]+=1;save(s)
 wiki=json.loads((root/"wikitext.json").read_text());g=json.loads((root/"gibc.json").read_text())["results"];payload=torch.load(c["checkpoint"],map_location="cpu",weights_only=False);retention=proxy(s,cid);tasks=("hellaswag","arc_easy","piqa","winogrande");metrics={"candidate_id":cid,"identity_pass":sha256(c["checkpoint"])==c["checkpoint_sha256"],"parameter_count":s["base"]["parameter_count"],"dedicated_wikitext_bpb":wiki["bits_per_byte"],"wikitext_bpb":wiki["bits_per_byte"],"wikitext_ppl":wiki["perplexity"],"data_d_validation":retention["data_d_validation"] if retention else float(payload.get("validation_loss",json.loads((ART/"final_submission_results.json").read_text())["metrics"]["validation_loss"])),**{t:g[t]["acc,none"] for t in tasks},**{t+"_normalized":g[t].get("acc_norm,none") for t in tasks},"lm_eval_wikitext":None,"paired_predictions":str(root/"predictions.json"),"v2_calibration":retention};atomic_json(root/"metrics.json",metrics);c["final_metrics"]=metrics;save(s);return metrics
def phase_tournament(s):
 from latticelm.posttraining.selection import proxy_shortlist
 for cid,c in s["candidates"].items():
  if cid=="BASE" or c.get("status") in ("COMPLETE","SAFE_STOPPED_VALID","TRAINED"):
   if not c.get("proxy"):c["proxy"]=proxy(s,cid,256)
 save(s);ids=proxy_shortlist(s["candidates"],min(8,max(5,int((HARD_CUTOFF-time.time())/3600)*2)));event("FINAL_SHORTLIST",candidate_ids=ids,reason="proxy rank and method diversity within official evaluation budget");metrics=[]
 for cid in ids:
  if time.time()>=HARD_CUTOFF-900:break
  result=eval_finalist(s,cid)
  if result:metrics.append(result)
 complete(s,"FINAL_CANDIDATE_TOURNAMENT",finalist_metrics=metrics)
def phase_selection(s):
 from latticelm.posttraining.selection import select
 if not s.get("finalist_metrics"):raise RuntimeError("no certified finalist metrics")
 decision=select(s["finalist_metrics"]);winner=decision["selected"]["candidate_id"];candidate=s["candidates"][winner];result={"schema":"posttraining-final-selection-v1","selected_candidate_id":winner,"selected_is_base":winner=="BASE","checkpoint_path":candidate["checkpoint"],"checkpoint_sha256":candidate["checkpoint_sha256"],"parent_lineage":candidate.get("parent_checkpoint_sha256"),"method_chain":candidate.get("method_chain",[]),"posttraining_token_budget":candidate.get("training_tokens",0),"optimizer":candidate.get("optimizer"),"learning_rate":candidate.get("lr"),"replay_fraction":candidate.get("replay"),"metrics":decision["selected"],"comparison_to_base":next(x for x in s["finalist_metrics"] if x["candidate_id"]=="BASE"),"selection":decision,"evidence_paths":[str(PT/"candidates"/winner/"final_eval/metrics.json")]};atomic_json(PT/"final_selection.json",result);complete(s,"FINAL_SELECTION",selection=result)
def phase_export(s):
 sel=s["selection"];candidate=s["candidates"][sel["selected_candidate_id"]];metrics={"tokens_trained":sel["posttraining_token_budget"],"wall_seconds":candidate.get("wall_seconds",0),"val_loss":sel["metrics"]["data_d_validation"],"val_ppl":math.exp(sel["metrics"]["data_d_validation"]),"optimizer":sel.get("optimizer") or "none-base-selected","learning_rate":sel.get("learning_rate"),"replay_fraction":sel.get("replay_fraction")};path=PT/"export_metrics.json";atomic_json(path,metrics)
 for attempt in (1,2):
  try:
   command(s,f"final-export-{attempt}",[sys.executable,"scripts/export_release_model.py","--checkpoint",sel["checkpoint_path"],"--output",PT/"final_release","--experiment","latticelm-final-selected","--role","best","--tokenizer",TOKENIZER,"--tokenizer-report",TOKENIZER_REPORT,"--dataset-manifest",MANIFEST,"--metrics",path]);break
  except RuntimeError:
   if attempt==2:raise
 publication=None
 for attempt in (1,2):
  if command(s,f"final-publication-{attempt}",[sys.executable,"scripts/publish_posttraining_release.py","--directory",PT/"final_release","--output",PT/"final_release_publication.json"],optional=True):publication=json.loads((PT/"final_release_publication.json").read_text());break
 complete(s,"FINAL_EXPORT",release=str(PT/"final_release"),publication=publication or {"verified":False,"status":"FAILED_BOUNDED_RETRIES","local_winner_unchanged":True})
def phase_evidence(s):
 atomic_json(PT/"final_metrics.json",s["selection"]["metrics"]);paths=[PT/"final_selection.json",PT/"final_metrics.json",PT/"lattice_reason_v2/audit.json",PT/"rollout_backend_certification.json",PT/"systems_benchmark.json"]+[p for p in (PT/"final_release").glob("*") if p.is_file()];index={str(p.relative_to(ROOT)):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in paths if p.exists()};atomic_json(PT/"final_evidence_index.json",index)
 candidates={cid:{k:c.get(k) for k in ("status","method","method_chain","parent_checkpoint_sha256","checkpoint_sha256","training_tokens","processed_tokens","updates","elapsed_seconds","observed_logical_tokens_per_second","observed_full_tokens_per_second","proxy","final_metrics")} for cid,c in s["candidates"].items()};summary={"created_at":s["created_at"],"completed_at":time.time(),"total_wall_seconds":time.time()-s["created_at"],"unused_before_experiment_cutoff_seconds":max(0,EXPERIMENT_CUTOFF-time.time()),"selected":s["selection"]["selected_candidate_id"],"base_comparison":s["selection"]["comparison_to_base"],"throughput_by_method":s["throughput"],"systems_benchmark":json.loads((PT/"systems_benchmark.json").read_text()) if (PT/"systems_benchmark.json").exists() else None,"candidates":candidates,"adaptive_decisions":s.get("adaptive_decisions",[]),"optional_failures":s["optional_failures"],"finalists":s["finalist_metrics"],"selection":s["selection"]["selection"],"no_further_research_reason":s.get("no_further_research_reason","all selected research paths exhausted")};atomic_json(PT/"final_research_summary.json",summary);(PT/"final_report.md").write_text("# LatticeLM post-training final\n\n"+f"Selected: **{summary['selected']}**\n\nCheckpoint: `{s['selection']['checkpoint_sha256']}`\n\nWall seconds: {summary['total_wall_seconds']:.0f}; unused before experiment cutoff: {summary['unused_before_experiment_cutoff_seconds']:.0f}.\n\n"+f"Research stop reason: {summary['no_further_research_reason']}\n\n"+f"Candidates: {len(candidates)}; official finalists: {len(s['finalist_metrics'])}; optional failures: {len(s['optional_failures'])}.\n\nDetailed lineage, proxy trajectories, throughput and finalist metrics are in `final_research_summary.json`.\n");paths.extend((PT/"final_research_summary.json",PT/"final_report.md"));atomic_json(PT/"final_evidence_index.json",{str(p.relative_to(ROOT)):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in paths if p.exists()});atomic_json(PT/"demo_metrics_selected.json",s["selection"]["metrics"]);atomic_json(PT/"demo_timeline_selected.json",{"completed":s["completed"],"selected":s["selection"]["selected_candidate_id"]});complete(s,"FINAL_EVIDENCE_PACKAGE")
def phase_done(s):
 if not s.get("selection") or not (PT/"final_evidence_index.json").exists() or not (PT/"final_release").exists():raise RuntimeError("finalization evidence incomplete")
 if "POSTTRAINING_COMPLETE" not in s["completed"]:s["completed"].append("POSTTRAINING_COMPLETE")
 save(s,"POSTTRAINING_COMPLETE",terminal_state="POSTTRAINING_COMPLETE",completed_at=time.time())

def dispatch(s):
 phase=s["phase"]
 if phase=="WAIT_FOR_PRODUCTION_COMPLETE":return phase_wait(s)
 if phase=="CERTIFY_AND_FREEZE_BASE":phase_certify(s)
 elif phase=="POSTTRAINING_PREFLIGHT":phase_preflight(s)
 elif phase=="LATTICEREASON_V2_BUILD":phase_build(s)
 elif phase=="LATTICEREASON_V2_AUDIT":phase_audit(s)
 elif phase=="BASE_CAPABILITY_MAP":phase_map(s)
 elif phase=="ROLLOUT_BACKEND_CERTIFICATION":phase_rollout(s)
 elif phase=="OPTIMIZER_SCREEN":phase_optimizer(s)
 elif phase=="LR_SCREEN":phase_lr(s)
 elif phase=="REPLAY_SCREEN":phase_replay(s)
 elif phase=="SFT_TRUNK":phase_sft(s)
 elif phase=="RANKING_TOURNAMENT":phase_ranking(s)
 elif phase=="JOINT_TOURNAMENT":phase_joint(s)
 elif phase=="RFT_ELIGIBILITY":phase_capability_eligibility(s,phase)
 elif phase=="RFT_TOURNAMENT":phase_rft(s) if s.get("rft_eligible") else complete(s,phase,rft_tournament="SKIPPED_INELIGIBLE")
 elif phase=="SIMPO_ELIGIBILITY":complete(s,phase,simpo_eligible=True)
 elif phase=="SIMPO_TOURNAMENT":simple_branch_phase(s,phase,"simpo","simpo",strongest_candidate(s,{"sft","ranking","joint"}),2_500_000)
 elif phase=="RLVR_ELIGIBILITY":phase_capability_eligibility(s,phase)
 elif phase=="RLVR_PILOTS":phase_rlvr_pilots(s) if s.get("rlvr_eligible") else complete(s,phase,rlvr_pilots=[])
 elif phase=="RLVR_EXTENSION":phase_rlvr_extension(s) if s.get("rlvr_eligible") else complete(s,phase,rlvr_extension="SKIPPED_INELIGIBLE")
 elif phase=="RECOVERY_ANNEAL":phase_recovery(s)
 elif phase=="INTERPOLATION":phase_interpolation(s)
 elif phase=="ADAPTIVE_RESEARCH":phase_adaptive(s)
 elif phase=="FINAL_CANDIDATE_TOURNAMENT":phase_tournament(s)
 elif phase=="FINAL_SELECTION":phase_selection(s)
 elif phase=="FINAL_EXPORT":phase_export(s)
 elif phase=="FINAL_EVIDENCE_PACKAGE":phase_evidence(s)
 elif phase=="POSTTRAINING_COMPLETE":phase_done(s)
 return True
def dry_run():
 s=initial();assert PHASES[0]=="WAIT_FOR_PRODUCTION_COMPLETE" and PHASES[-1]=="POSTTRAINING_COMPLETE";assert EXPERIMENT_CUTOFF<HARD_CUTOFF;assert next(iter(s["candidates"]))=="BASE";return {"status":"PASS","phases":len(PHASES),"experiment_cutoff":EXPERIMENT_CUTOFF,"hard_cutoff":HARD_CUTOFF,"passive_wait":True,"base_always_candidate":True}
def main():
 p=argparse.ArgumentParser();p.add_argument("--poll-seconds",type=float,default=60);p.add_argument("--dry-run",action="store_true");a=p.parse_args()
 if a.dry_run:print(json.dumps(dry_run(),indent=2));return 0
 signal.signal(signal.SIGTERM,signal_handler);signal.signal(signal.SIGINT,signal_handler);PT.mkdir(parents=True,exist_ok=True)
 try:
  with Lock(LOCK):
   s=load()
   if not STATE.exists():save(s)
   while not STOP and s.get("terminal_state")!="POSTTRAINING_COMPLETE":
    if time.time()>=HARD_CUTOFF:raise RuntimeError("absolute post-training cutoff reached")
    # Even after the experiment cutoff, semantic handoff and BASE identity
    # certification must occur before jumping over optional research phases.
    if s.get("base") and time.time()>=research_deadline(s) and PHASES.index(s["phase"])<PHASES.index("FINAL_CANDIDATE_TOURNAMENT"):save(s,"FINAL_CANDIDATE_TOURNAMENT",cutoff_transition_at=time.time(),no_further_research_reason="finalization reserve reached")
    progressed=dispatch(s)
    if not progressed:
     deadline=time.monotonic()+a.poll_seconds
     while not STOP and time.monotonic()<deadline:time.sleep(min(1,max(0,deadline-time.monotonic())))
   return 75 if STOP else 0
 except BlockingIOError:return 73
 except Exception as e:event("BLOCKED",phase=locals().get("s",{}).get("phase"),error=f"{type(e).__name__}: {e}");return 2
if __name__=="__main__":raise SystemExit(main())
