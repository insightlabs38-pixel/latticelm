#!/usr/bin/env python3
"""Autonomous, restart-safe final LatticeLM post-training master."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";PT=ART/"posttraining";STATE=PT/"master_state.json";PREVIOUS=PT/"master_state.previous.json";LOCK=PT/"master.lock";EVENTS=PT/"events.jsonl"
PRODUCTION_STATE=ART/"final_production_master_state.json";TOKENIZER=ART/"tokenizers/final_corpus_4k.json";TOKENIZER_REPORT=ART/"tokenizers/final_corpus_4k.report.json";MANIFEST=ART/"data/data_d_v4/canonical-2250m/manifest.json"
EXPERIMENT_CUTOFF=datetime(2026,9,30,15,0,tzinfo=ZoneInfo("America/New_York")).timestamp();HARD_CUTOFF=datetime(2026,9,30,16,0,tzinfo=ZoneInfo("America/New_York")).timestamp();SAFETY=1.35;FINAL_RESERVE=3600
PHASES=("WAIT_FOR_PRODUCTION_COMPLETE","CERTIFY_AND_FREEZE_BASE","POSTTRAINING_PREFLIGHT","LATTICEREASON_V2_BUILD","LATTICEREASON_V2_AUDIT","BASE_CAPABILITY_MAP","ROLLOUT_BACKEND_CERTIFICATION","OPTIMIZER_SCREEN","LR_SCREEN","REPLAY_SCREEN","SFT_TRUNK","RANKING_TOURNAMENT","RFT_ELIGIBILITY","RFT_TOURNAMENT","SIMPO_ELIGIBILITY","SIMPO_TOURNAMENT","RLVR_ELIGIBILITY","RLVR_PILOTS","RLVR_EXTENSION","RECOVERY_ANNEAL","INTERPOLATION","FINAL_CANDIDATE_TOURNAMENT","FINAL_SELECTION","FINAL_EXPORT","FINAL_EVIDENCE_PACKAGE","POSTTRAINING_COMPLETE")
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
def experiment_allowed(predicted_seconds=0):return time.time()<EXPERIMENT_CUTOFF and time.time()+predicted_seconds*SAFETY<EXPERIMENT_CUTOFF
def schedule_decision(s,name,tokens,minimum=1_000_000,standard=None,maximum=None):
 tps=float(s["throughput"].get(name,s["throughput"].get("sft",50.0)));standard=standard or tokens;maximum=maximum or standard;remaining=max(0,EXPERIMENT_CUTOFF-time.time());fit=int(remaining*tps/SAFETY);budget=max(minimum,min(maximum,fit,standard)) if fit>=minimum else 0
 return {"objective":name,"minimum":minimum,"standard":standard,"maximum":maximum,"selected":budget,"measured_tokens_per_second":tps,"safety_factor":SAFETY,"seconds_to_experiment_cutoff":max(0,EXPERIMENT_CUTOFF-time.time())}
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
def worker(s,cid,parent,method,tokens,optimizer="muon_hybrid",lr=3e-5,replay=.2,optional=False):
 out=PT/"candidates"/cid;prior=json.loads((out/"result.json").read_text()) if (out/"result.json").exists() else None
 if prior and int(prior["training_tokens"])>=tokens:return prior
 plan=schedule_decision(s,method,tokens,min(tokens,1_000_000),tokens,tokens)
 if not experiment_allowed(tokens/max(plan["measured_tokens_per_second"],1)):
  event("EXPERIMENT_SKIPPED_CUTOFF",candidate_id=cid,plan=plan);return None
 started=time.perf_counter();ok=command(s,cid,[sys.executable,"scripts/train_posttraining_worker.py","--base",parent,"--output",out,"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--method",method,"--tokens",tokens,"--optimizer",optimizer,"--lr",lr,"--replay",replay,"--threads",16,"--stop-epoch",EXPERIMENT_CUTOFF],optional=optional)
 if not ok:return None
 result=json.loads((out/"result.json").read_text());rate=tokens/max(time.perf_counter()-started,1e-9);s["throughput"][method]=rate;s["candidates"][cid]={**result,"candidate_id":cid,"optimizer":optimizer,"lr":lr,"replay":replay,"status":"TRAINED"};save(s);return result
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
 atomic_json(root/"result.json",{"status":"PASS","fresh_optimizer":True,"completion_masks":True,"ranking_branch":bool(ranking),"checkpoint_restart":True,"next_batch_identity":True,"parent_immutable":True,"evaluator_invocation":True,"branch_sha":result["checkpoint_sha256"]});complete(s,"POSTTRAINING_PREFLIGHT")
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
def proxy(s,cid):
 out=PT/"candidates"/cid/"proxy.json";cp=s["candidates"][cid]["checkpoint"];command(s,cid+"-proxy",[sys.executable,"scripts/evaluate_posttraining_proxy.py","--checkpoint",cp,"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--output",out,"--examples",128,"--threads",16],optional=True);return json.loads(out.read_text()) if out.exists() else None
def choose_screen(s,prefix,field,default):
 rows=[]
 for cid,x in s["candidates"].items():
  if cid.startswith(prefix) and x.get("status")=="TRAINED":
   x["proxy"]=x.get("proxy") or proxy(s,cid)
   if x["proxy"]:rows.append(x)
 if not rows:return default
 base_loss=min(x["proxy"]["data_d_validation"] for x in rows);eligible=[x for x in rows if x["proxy"]["data_d_validation"]<=base_loss*1.03];return max(eligible,key=lambda x:(x["proxy"]["v2_ranking_accuracy"],x["proxy"]["v2_mean_margin"],-x["proxy"]["data_d_validation"]))[field]
def phase_optimizer(s):
 for opt in ("muon_hybrid","adamw"):
  cid="screen-opt-"+opt;result=worker(s,cid,base_path(s),"sft",1_000_000,opt,3e-5,.2,True)
  if result:s["candidates"][cid]["proxy"]=proxy(s,cid)
 selected=choose_screen(s,"screen-opt","optimizer","muon_hybrid");complete(s,"OPTIMIZER_SCREEN",selected_optimizer=selected)
def phase_lr(s):
 for lr in (1e-5,3e-5,1e-4):worker(s,f"screen-lr-{lr:g}",base_path(s),"sft",1_000_000,s["selected_optimizer"],lr,.2,True)
 selected=choose_screen(s,"screen-lr","lr",3e-5);complete(s,"LR_SCREEN",selected_lr=selected)
def phase_replay(s):
 for replay in (0.,.2,.4):worker(s,f"screen-replay-{int(replay*100)}",base_path(s),"sft",1_000_000,s["selected_optimizer"],s["selected_lr"],replay,True)
 selected=choose_screen(s,"screen-replay","replay",.2);complete(s,"REPLAY_SCREEN",selected_replay=selected)
def phase_sft(s):
 cid="sft-trunk";result=None;history=[];base_proxy=proxy(s,"BASE")
 for budget in (1_000_000,5_000_000,10_000_000,20_000_000,30_000_000,50_000_000,75_000_000,100_000_000,150_000_000):
  if budget>30_000_000 and not experiment_allowed(budget/max(s["throughput"].get("sft",50),1)):break
  result=worker(s,cid,base_path(s),"sft",budget,s["selected_optimizer"],s["selected_lr"],s["selected_replay"],True)
  if result is None:break
  out=PT/"candidates"/cid/f"proxy-{budget}.json";command(s,f"{cid}-{budget}-proxy",[sys.executable,"scripts/evaluate_posttraining_proxy.py","--checkpoint",result["checkpoint"],"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--output",out,"--examples",128,"--threads",16],optional=True);metrics=json.loads(out.read_text()) if out.exists() else None;history.append({"tokens":budget,"metrics":metrics})
  if metrics and base_proxy and metrics["data_d_validation"]>base_proxy["data_d_validation"]*1.10:break
  if budget>=30_000_000 and len(history)>=2 and metrics and history[-2]["metrics"] and metrics["v2_ranking_accuracy"]-history[-2]["metrics"]["v2_ranking_accuracy"]<.005:break
 complete(s,"SFT_TRUNK",sft_winner=cid if result else "BASE",sft_dose_response=history)
def simple_branch_phase(s,phase,method,cid,parent_key,budget,optional=True):
 parent=base_path(s) if parent_key=="BASE" else Path(s["candidates"].get(parent_key,s["candidates"]["BASE"])["checkpoint"]);result=worker(s,cid,parent,method,budget,s.get("selected_optimizer","muon_hybrid"),s.get("selected_lr",3e-5),s.get("selected_replay",.2),optional);complete(s,phase,**{phase.lower():cid if result else "SKIPPED"})
def phase_interpolation(s):
 import torch
 parent=s["candidates"].get("ranking",s["candidates"].get("sft-trunk"));created=[]
 if parent:
  base=torch.load(base_path(s),map_location="cpu",weights_only=False);post=torch.load(parent["checkpoint"],map_location="cpu",weights_only=False)
  if base["model"].keys()!=post["model"].keys():raise RuntimeError("interpolation parameter names differ")
  for alpha in (.25,.5,.75):
   cid=f"interp-{alpha:.2f}";out=PT/"candidates"/cid;out.mkdir(parents=True,exist_ok=True);payload={**post,"model":{k:base["model"][k].mul(1-alpha).add(post["model"][k],alpha=alpha) for k in base["model"]},"interpolation":{"base_sha":s["base"]["checkpoint_sha256"],"post_sha":parent["checkpoint_sha256"],"alpha":alpha}};cp=out/"checkpoint.pt";torch.save(payload,cp);(out/"checkpoint.sha256").write_text(sha256(cp)+"\n");s["candidates"][cid]={"candidate_id":cid,"checkpoint":str(cp),"checkpoint_sha256":sha256(cp),"status":"TRAINED","method":"interpolation","alpha":alpha};created.append(cid)
  save(s)
 complete(s,"INTERPOLATION",interpolations=created)
def eval_finalist(s,cid):
 c=s["candidates"][cid];root=PT/"candidates"/cid/"final_eval";root.mkdir(parents=True,exist_ok=True)
 if not(root/"wikitext.json").exists():command(s,cid+"-wiki",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",c["checkpoint"],"--tokenizer",TOKENIZER,"--output",root/"wikitext.json","--threads",16,"--batch-size",16])
 if not(root/"gibc.json").exists():
  if s["gibc_evaluations"]>=8:return None
  command(s,cid+"-gibc",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",c["checkpoint"],"--tokenizer",TOKENIZER,"--output",root/"gibc.json","--predictions-output",root/"predictions.json","--tasks","hellaswag,arc_easy,piqa,winogrande","--threads",16,"--batch-size",16]);s["gibc_evaluations"]+=1;save(s)
 wiki=json.loads((root/"wikitext.json").read_text());g=json.loads((root/"gibc.json").read_text())["results"];payload=torch.load(c["checkpoint"],map_location="cpu",weights_only=False);retention=proxy(s,cid);tasks=("hellaswag","arc_easy","piqa","winogrande");metrics={"candidate_id":cid,"identity_pass":sha256(c["checkpoint"])==c["checkpoint_sha256"],"parameter_count":s["base"]["parameter_count"],"dedicated_wikitext_bpb":wiki["bits_per_byte"],"wikitext_bpb":wiki["bits_per_byte"],"wikitext_ppl":wiki["perplexity"],"data_d_validation":retention["data_d_validation"] if retention else float(payload.get("validation_loss",json.loads((ART/"final_submission_results.json").read_text())["metrics"]["validation_loss"])),**{t:g[t]["acc,none"] for t in tasks},**{t+"_normalized":g[t].get("acc_norm,none") for t in tasks},"lm_eval_wikitext":None,"paired_predictions":str(root/"predictions.json"),"v2_calibration":retention};atomic_json(root/"metrics.json",metrics);c["final_metrics"]=metrics;save(s);return metrics
def phase_tournament(s):
 ids=["BASE"]+[x for x in ("sft-trunk","ranking","rft","simpo","rlvr-grpo","rlvr-rloo","recovery","interp-0.25","interp-0.50","interp-0.75") if x in s["candidates"]];metrics=[]
 for cid in ids:
  if time.time()>=HARD_CUTOFF-900:break
  result=eval_finalist(s,cid)
  if result:metrics.append(result)
 complete(s,"FINAL_CANDIDATE_TOURNAMENT",finalist_metrics=metrics)
def phase_selection(s):
 from latticelm.posttraining.selection import select
 if not s.get("finalist_metrics"):raise RuntimeError("no certified finalist metrics")
 decision=select(s["finalist_metrics"]);winner=decision["selected"]["candidate_id"];candidate=s["candidates"][winner];result={"schema":"posttraining-final-selection-v1","selected_candidate_id":winner,"selected_is_base":winner=="BASE","checkpoint_path":candidate["checkpoint"],"checkpoint_sha256":candidate["checkpoint_sha256"],"parent_lineage":candidate.get("parent_checkpoint_sha256"),"method_chain":candidate.get("method",[]) if isinstance(candidate.get("method",[]),list) else [candidate.get("method")],"posttraining_token_budget":candidate.get("training_tokens",0),"optimizer":candidate.get("optimizer"),"learning_rate":candidate.get("lr"),"replay_fraction":candidate.get("replay"),"metrics":decision["selected"],"comparison_to_base":next(x for x in s["finalist_metrics"] if x["candidate_id"]=="BASE"),"selection":decision,"evidence_paths":[str(PT/"candidates"/winner/"final_eval/metrics.json")]};atomic_json(PT/"final_selection.json",result);complete(s,"FINAL_SELECTION",selection=result)
def phase_export(s):
 sel=s["selection"];metrics={"tokens_trained":sel["posttraining_token_budget"],"wall_seconds":1,"val_loss":sel["metrics"]["data_d_validation"],"val_ppl":math.exp(sel["metrics"]["data_d_validation"]),"optimizer":sel.get("optimizer") or "none-base-selected","learning_rate":sel.get("learning_rate"),"replay_fraction":sel.get("replay_fraction")};path=PT/"export_metrics.json";atomic_json(path,metrics)
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
 atomic_json(PT/"final_metrics.json",s["selection"]["metrics"]);paths=[PT/"final_selection.json",PT/"final_metrics.json",PT/"lattice_reason_v2/audit.json",PT/"rollout_backend_certification.json"]+[p for p in (PT/"final_release").glob("*") if p.is_file()];index={str(p.relative_to(ROOT)):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in paths if p.exists()};atomic_json(PT/"final_evidence_index.json",index);(PT/"final_report.md").write_text(f"# LatticeLM post-training final\n\nSelected: **{s['selection']['selected_candidate_id']}**\n\nCheckpoint: `{s['selection']['checkpoint_sha256']}`\n\nBASE lineage remains immutable.\n");atomic_json(PT/"demo_metrics_selected.json",s["selection"]["metrics"]);atomic_json(PT/"demo_timeline_selected.json",{"completed":s["completed"],"selected":s["selection"]["selected_candidate_id"]});complete(s,"FINAL_EVIDENCE_PACKAGE")
def phase_done(s):
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
 elif phase=="RANKING_TOURNAMENT":simple_branch_phase(s,phase,"ranking","ranking",s.get("sft_winner","BASE"),5_000_000)
 elif phase=="RFT_ELIGIBILITY":
  regions=json.loads(Path(s["base_capability_map"]).read_text()).get("regions",{});complete(s,phase,rft_eligible=s.get("rollout_backend")=="CERTIFIED" and any(x.get("pass_at_8",0)>0 for x in regions.values()))
 elif phase=="RFT_TOURNAMENT":simple_branch_phase(s,phase,"rft","rft",s.get("sft_winner","BASE"),5_000_000) if s.get("rft_eligible") else complete(s,phase,rft_tournament="SKIPPED_INELIGIBLE")
 elif phase=="SIMPO_ELIGIBILITY":complete(s,phase,simpo_eligible=True)
 elif phase=="SIMPO_TOURNAMENT":simple_branch_phase(s,phase,"simpo","simpo",s.get("sft_winner","BASE"),5_000_000)
 elif phase=="RLVR_ELIGIBILITY":
  regions=json.loads(Path(s["base_capability_map"]).read_text()).get("regions",{});complete(s,phase,rlvr_eligible=s.get("rollout_backend")=="CERTIFIED" and any(0<x.get("pass_at_8",0)<.95 for x in regions.values()))
 elif phase=="RLVR_PILOTS":simple_branch_phase(s,phase,"rlvr-grpo","rlvr-grpo",s.get("sft_winner","BASE"),1_000_000) if s.get("rlvr_eligible") else complete(s,phase,rlvr_pilots="SKIPPED_INELIGIBLE")
 elif phase=="RLVR_EXTENSION":simple_branch_phase(s,phase,"rlvr-rloo","rlvr-rloo",s.get("sft_winner","BASE"),5_000_000) if s.get("rlvr_eligible") else complete(s,phase,rlvr_extension="SKIPPED_INELIGIBLE")
 elif phase=="RECOVERY_ANNEAL":simple_branch_phase(s,phase,"recovery","recovery","ranking",5_000_000)
 elif phase=="INTERPOLATION":phase_interpolation(s)
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
    if time.time()>=EXPERIMENT_CUTOFF and s.get("base") and PHASES.index(s["phase"])<PHASES.index("FINAL_CANDIDATE_TOURNAMENT"):save(s,"FINAL_CANDIDATE_TOURNAMENT",cutoff_transition_at=time.time())
    progressed=dispatch(s)
    if not progressed:
     deadline=time.monotonic()+a.poll_seconds
     while not STOP and time.monotonic()<deadline:time.sleep(min(1,max(0,deadline-time.monotonic())))
   return 75 if STOP else 0
 except BlockingIOError:return 73
 except Exception as e:event("BLOCKED",phase=locals().get("s",{}).get("phase"),error=f"{type(e).__name__}: {e}");return 2
if __name__=="__main__":raise SystemExit(main())
