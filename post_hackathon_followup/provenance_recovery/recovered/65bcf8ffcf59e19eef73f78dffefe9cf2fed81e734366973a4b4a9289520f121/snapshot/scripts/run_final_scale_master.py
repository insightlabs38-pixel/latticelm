"""Autonomous, Spot-safe final-scale continuation of the frozen 32M lineage."""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import torch
from huggingface_hub import get_token,hf_hub_download
from latticelm.config import LatticeConfig
from latticelm.data_d import sha256_file,verify_top_manifest
from latticelm.hf_storage import export_checkpoint,upload_checkpoint
from latticelm.model import build_model

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";STATE=ART/"final_scale_master_state.json";BACKUP=ART/"final_scale_master_state.previous.json";EVENTS=ART/"logs/final_scale_master_events.jsonl";LOCK=ART/"final_scale_master.lock"
CAP_STATE=ART/"final_capacity_master_state.json";CAP_REPORT=ART/"final_capacity_report.md";CAP_SURFACE=ART/"final_capacity_surface.csv";CAP_DECISION=ART/"final_capacity_decision.md";CAP_PROJECTION=ART/"final_capacity_deadline_projection.csv"
DATA=ART/"data/data_d_v3/canonical-1b";MANIFEST=DATA/"manifest.json";TOK=ART/"tokenizers/babylm_2026_4k.json";TOK_REPORT=ART/"tokenizers/babylm_2026_4k.report.json";CONFIG=ROOT/"configs/final_capacity32_co4.json"
RUN="co4-final-capacity32-data-d-v3";CKPTS=ART/"checkpoints"/RUN;CURVE=ART/"final_scale_curve.csv";TRAIN_CURVE=ART/"final_capacity_training_curve.csv";REPO="insightlabs38-pixel/LatticeLM-research";PARAMS=32_678_640;START=205_000_000;SPEED=3357.17547435075
MILESTONES=(250_000_000,300_000_000,400_000_000,500_000_000,512_000_000,600_000_000,700_000_000,750_000_000,800_000_000,900_000_000);NORMALIZED={"M1":512_000_000,"M2":1_024_000_000,"M3":1_537_000_000,"M4":2_049_000_000};FINAL_TARGET=900_000_000;STOP=False;CHILD=None
def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name("."+path.name+".tmp")
 with tmp.open("w") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
 if path.exists():shutil.copy2(path,BACKUP)
 os.replace(tmp,path)
def load():
 for p in (STATE,BACKUP):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):pass
 return None
def event(kind,**kw):
 EVENTS.parent.mkdir(parents=True,exist_ok=True);fd=os.open(EVENTS,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o664)
 try:os.write(fd,(json.dumps({"at":now(),"event":kind,**kw},sort_keys=True)+"\n").encode());os.fsync(fd)
 finally:os.close(fd)
def save(s,stage=None,**kw):
 if stage:s["current_stage"]=stage
 s.update(kw);s["updated_at"]=now();atomic(STATE,s);event("STAGE",stage=s["current_stage"])
def request_stop(*_):
 global STOP;STOP=True;s=load()
 if s and s.get("current_stage") not in {"COMPLETE","BLOCKED","STOPPED_SAFE"}:save(s,stop_requested=True)
 if CHILD and CHILD.poll() is None:CHILD.terminate()
class Lock:
 def __enter__(self):
  self.f=LOCK.open("a+");fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB);self.f.seek(0);self.f.truncate();self.f.write(str(os.getpid()));self.f.flush();return self
 def __exit__(self,*_):fcntl.flock(self.f,fcntl.LOCK_UN);self.f.close()
def run(s,label,cmd,allow75=False):
 global CHILD
 if STOP or s.get("stop_requested"):raise InterruptedError("graceful stop requested")
 event("SUBPROCESS_START",label=label);CHILD=subprocess.Popen(cmd,cwd=ROOT);s["child_pid"]=CHILD.pid;save(s);code=CHILD.wait();CHILD=None;s["child_pid"]=None;save(s);event("SUBPROCESS_END",label=label,code=code)
 if code and not(allow75 and code==75):raise RuntimeError(f"{label} exited {code}")
 return code
def credential(name):
 directory=os.environ.get("CREDENTIALS_DIRECTORY")
 if directory:
  p=Path(directory)/name
  if p.exists():return p.read_text().strip()
 return os.environ.get(name) or (get_token() if name=="HF_TOKEN" else None)
def initial():
 t=datetime.now(timezone.utc);return {"run_id":f"final-scale-{t:%Y%m%dT%H%M%SZ}","current_stage":"PREFLIGHT","completed_stages":[],"launch_time":t.isoformat(),"experiment_budget_start_time":None,"soft_deadline":None,"hard_deadline":None,"source_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip(),"child_pid":None,"stop_requested":False,"spot_interruptions":0,"active_training_seconds":0.0,"start_checkpoint_tokens":START,"final_target":FINAL_TARGET}
def checkpoint():
 for p in (CKPTS/"latest.pt",CKPTS/"previous.pt",CKPTS/"fallback.pt"):
  if p.exists() and p.with_suffix(".sha256").exists() and sha256_file(p)==p.with_suffix(".sha256").read_text().strip():return p,torch.load(p,map_location="cpu",weights_only=False)
 raise RuntimeError("no valid latest/previous/fallback checkpoint")
def gate(s):
 cap=json.loads(CAP_STATE.read_text());texts="\n".join(p.read_text() for p in (CAP_REPORT,CAP_DECISION))
 if cap.get("current_stage")!="COMPLETE" or cap.get("capacity_search_closed") is not True or "CAPACITY SEARCH CLOSED: YES" not in texts or cap.get("final_capacity")!="~32M":raise RuntimeError("completed capacity master did not close on ~32M")
 for p in (CAP_SURFACE,CAP_PROJECTION,ART/"logs/final_capacity_master_events.jsonl",ART/"final_capacity_205m_metrics.json"):
  if not p.exists() or not p.stat().st_size:raise RuntimeError(f"missing capacity evidence: {p}")
 top=verify_top_manifest(MANIFEST,TOK);cert=json.loads((DATA/"certification.json").read_text());accept=json.loads((ART/"data_d_v3_acceptance.json").read_text());mh=sha256_file(MANIFEST)
 required={"acceptance":"PASS","train_validation_disjointness":"PASS","all_shards_mmap_and_hash":"PASS","exact_resume":"PASS","stable_unique_ids":"PASS"}
 if any(cert.get(k)!=v for k,v in required.items()) or accept.get("acceptance")!="PASS" or cert.get("manifest_sha256")!=mh:raise RuntimeError("DATA-D-v3 certification failed")
 p,state=checkpoint();cfg=LatticeConfig.from_json(CONFIG);model=build_model(cfg)
 if model.parameter_breakdown()["total"]!=PARAMS or state["config"]!=cfg.to_dict() or state["tokens_seen"]!=START or state["data_manifest_sha256"]!=mh or state["backend"]!="fp32_compile":raise RuntimeError("winning checkpoint identity mismatch")
 model.load_state_dict(state["model"],strict=True)
 for k in ("optimizer","scheduler","python_rng_state","torch_rng_state","data_source_selector_state","next_batch_sha256"): 
  if k not in state:raise RuntimeError(f"checkpoint exact-resume field absent: {k}")
 expected=(CKPTS/"milestone-205000000.sha256").read_text().strip();metric=json.loads((ART/"final_capacity_205m_metrics.json").read_text())
 if sha256_file(CKPTS/"milestone-205000000.pt")!=expected or metric["native_checkpoint_sha256"]!=expected or cap["final_lineage"]["checkpoint_sha256"]!=expected:raise RuntimeError("winning checkpoint SHA lineage mismatch")
 if NORMALIZED["M2"]<=top["total_unique_tokens"]:raise RuntimeError("planning invariant changed: reconsider milestones explicitly")
 s.update({"capacity":{"final_capacity":"~32M","parameters":PARAMS,"confidence":cap["confidence"],"closed":True},"starting_checkpoint":{"path":str(p),"tokens":START,"sha256":expected,"hf_revision":metric["hf_revision"]},"data":{"identity":top["corpus_identity"],"unique_tokens":top["total_unique_tokens"],"manifest_sha256":mh,"tokenizer_sha256":sha256_file(TOK)},"backend":"fp32_compile","measured_tokens_per_second":float(cap["tokens_per_second"]),"normalized_milestones":NORMALIZED,"max_safe_milestone":FINAL_TARGET,"data_limited":True});save(s,"PREFLIGHT_PASS")
def copy_milestone(target):
 dst=CKPTS/f"milestone-{target}.pt";src,state=checkpoint()
 if int(state["tokens_seen"])!=target:raise RuntimeError("milestone token mismatch")
 if not dst.exists():shutil.copy2(src,dst);dst.with_suffix(".sha256").write_text(sha256_file(dst)+"\n")
 if sha256_file(dst)!=dst.with_suffix(".sha256").read_text().strip():raise RuntimeError("immutable milestone verification failed")
 return dst,state
def evaluate(s,target,full):
 cp,_=copy_milestone(target);wiki=ART/f"final_scale_{target}_wikitext.json";gibc=ART/f"final_scale_{target}_gibc.json"
 if not wiki.exists():run(s,f"wikitext-{target}",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(cp),"--tokenizer",str(TOK),"--output",str(wiki),"--threads","4"])
 if full and not gibc.exists():run(s,f"official-{target}",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(cp),"--tokenizer",str(TOK),"--output",str(gibc),"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","4"])
 return json.loads(wiki.read_text()),json.loads(gibc.read_text()) if full else None
def publish(s,target,wiki,gibc):
 cp,_=copy_milestone(target);out=ART/"cache"/f"final-scale-{target}";public=ART/"final_scale_dataset_manifest.json";public.write_text(json.dumps({"dataset":"DATA-D-BROAD-v3","revision":sha256_file(MANIFEST),"sources":["FineWeb-Edu","English Wikipedia","FineWeb"],"canonical_manifest":str(MANIFEST.relative_to(ROOT))},indent=2)+"\n")
 row=next(r for r in reversed(list(csv.DictReader(TRAIN_CURVE.open()))) if int(r["nominal_tokens"])==target);export_checkpoint(cp,out,f"final-scale-{target}","immutable-milestone",TOK,TOK_REPORT,public,{"tokens_trained":target,"wall_seconds":float(row["cumulative_training_seconds"]),"val_loss":float(row["data_d_validation_loss"]),"val_ppl":math.exp(float(row["data_d_validation_loss"])),"wikitext_ppl":wiki["perplexity"],"wikitext_bpb":wiki["bits_per_byte"]})
 token=credential("HF_TOKEN")
 if not token:raise RuntimeError("secure HF credential unavailable")
 last=None
 for attempt in range(1,4):
  try:
   rev=upload_checkpoint(out,REPO,f"experiments/final-scale/{target}",token);break
  except Exception as exc:last=exc;event("HF_UPLOAD_RETRY",target=target,attempt=attempt,error=type(exc).__name__);time.sleep(10*attempt)
 else:raise RuntimeError(f"HF upload failed after retries: {type(last).__name__}")
 hashes={}
 for line in (out/"SHA256SUMS").read_text().splitlines():
  expected,name=line.split(None,1);local=Path(hf_hub_download(REPO,f"experiments/final-scale/{target}/{name.strip()}",revision=rev,token=token));observed=sha256_file(local)
  if observed!=expected:raise RuntimeError("remote hash verification failed")
  hashes[name.strip()]=observed
 result={"tokens":target,"checkpoint_sha256":sha256_file(cp),"hf_revision":rev,"remote_hashes":hashes,"wikitext":wiki,"gibc":gibc};(ART/f"final_scale_{target}_metrics.json").write_text(json.dumps(result,indent=2)+"\n");return result
def rows():return list(csv.DictReader(TRAIN_CURVE.open()))
def continuation_ok(target):
 rs=[r for r in rows() if int(r["nominal_tokens"])>=START];cur=next(r for r in reversed(rs) if int(r["nominal_tokens"])==target);prior=[r for r in rs if int(r["nominal_tokens"])<target]
 finite=all(math.isfinite(float(cur[k])) for k in ("train_loss","data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss"));healthy=not prior or float(cur["data_d_validation_loss"])<=float(prior[-1]["data_d_validation_loss"])+0.03
 return finite and healthy
def reports(s,results):
 rs=[r for r in rows() if int(r["nominal_tokens"])>=START];fields=["tokens","tokens_per_parameter","elapsed_training_seconds","tokens_per_second","cpu_hours","train_loss","data_d_validation","fineweb_edu_validation","wikipedia_validation","fineweb_validation","wikitext_ppl","wikitext_bpb","scaling_status","checkpoint_sha256"]
 with CURVE.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();previous=None
  for r in rs:
   t=int(r["nominal_tokens"]);loss=float(r["data_d_validation_loss"]);status="HEALTHY" if previous is None or loss<previous else "SLOWING" if loss<=previous+.01 else "DEGRADING";ev=results.get(str(t),{}).get("wikitext",{});w.writerow({"tokens":t,"tokens_per_parameter":t/PARAMS,"elapsed_training_seconds":r["cumulative_training_seconds"],"tokens_per_second":r["tokens_per_second"],"cpu_hours":float(r["cumulative_training_seconds"])*int(r["threads"])/3600,"train_loss":r["train_loss"],"data_d_validation":loss,"fineweb_edu_validation":r["fineweb_edu_validation_loss"],"wikipedia_validation":r["wikipedia_validation_loss"],"fineweb_validation":r["fineweb_validation_loss"],"wikitext_ppl":ev.get("perplexity"),"wikitext_bpb":ev.get("bits_per_byte"),"scaling_status":status,"checkpoint_sha256":r["checkpoint_sha256"]});previous=loss
 shutil.copy2(CURVE,ART/"final_scale_milestones.csv")
 final=max(int(k) for k in results);best=results[str(final)];g=best["gibc"]["results"];prior=json.loads((ART/"final_capacity_205m_metrics.json").read_text());pg=prior["gibc"]["results"]
 metrics={"wikitext_ppl":best["wikitext"]["perplexity"],"wikitext_bpb":best["wikitext"]["bits_per_byte"],**{k:g[k]["acc,none"] for k in ("hellaswag","arc_easy","piqa","winogrande")}}
 with (ART/"final_scale_benchmark_deltas.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["metric","prior_205m","final","delta"]);w.writerow(["wikitext_ppl",prior["wikitext"]["perplexity"],metrics["wikitext_ppl"],metrics["wikitext_ppl"]-prior["wikitext"]["perplexity"]]);w.writerow(["wikitext_bpb",prior["wikitext"]["bits_per_byte"],metrics["wikitext_bpb"],metrics["wikitext_bpb"]-prior["wikitext"]["bits_per_byte"]]);[w.writerow([k,pg[k]["acc,none"],metrics[k],metrics[k]-pg[k]["acc,none"]]) for k in ("hellaswag","arc_easy","piqa","winogrande")]
 wall=(datetime.now(timezone.utc)-datetime.fromisoformat(s["experiment_budget_start_time"])).total_seconds();active=float(rs[-1]["cumulative_training_seconds"])-float(next(r for r in rows() if int(r["nominal_tokens"])==START)["cumulative_training_seconds"])
 with (ART/"final_scale_runtime.csv").open("w",newline="") as f:w=csv.writer(f);w.writerow(["wall_seconds","active_training_seconds","spot_interruptions","final_tokens","measured_tokens_per_second"]);w.writerow([wall,active,s.get("spot_interruptions",0),final,s["measured_tokens_per_second"]])
 status="HEALTHY" if float(rs[-1]["data_d_validation_loss"])<float(rs[0]["data_d_validation_loss"]) else "DEGRADING";decision=f"# Final scale decision\n\nBEST BASE CHECKPOINT SO FAR: {final} tokens\n\nSCALING STATUS: {status}\n\nMORE BASE SCALING JUSTIFIED: NO\n\nBASE MODEL READY FOR POST-TRAINING RESEARCH: YES\n\nORGANIZER POST-TRAINING CLARIFICATION STATUS: UNKNOWN\n\nNEXT HIGHEST-VALUE ACTION: Obtain organizer clarification, then design one bounded post-training experiment. Do not launch it.\n\nM2 is DATA-LIMITED: its 1.024B target exceeds 1,009,592,508 certified unique tokens.\n";(ART/"final_scale_decision.md").write_text(decision)
 report=f"# Final scale report\n\nFINAL CAPACITY: ~32M\nPARAMETERS: {PARAMS}\nCAPACITY SEARCH CLOSED: YES\n\nSTART CHECKPOINT TOKENS: {START}\nSTART CHECKPOINT SHA: {s['starting_checkpoint']['sha256']}\nSTART HF REVISION: {s['starting_checkpoint']['hf_revision']}\n\nMEASURED TOK/S: {s['measured_tokens_per_second']}\nSPOT INTERRUPTIONS: {s.get('spot_interruptions',0)}\nTOTAL WALL TIME: {wall}\nTOTAL ACTIVE TRAINING TIME: {active}\n\nCANONICAL DATASET: DATA-D-BROAD-v3\nCERTIFIED UNIQUE TOKENS AVAILABLE: {s['data']['unique_tokens']}\nUNIQUE TOKENS CONSUMED: {final}\nDATA-LIMITED: YES\n\nM1 REACHED: {'YES' if final>=NORMALIZED['M1'] else 'NO'}\nM2 REACHED: NO\nM3 REACHED: NO\nM4 REACHED: NO\nFINAL TOKENS: {final}\nFINAL TOKENS/PARAMETER: {final/PARAMS}\nSCALING STATUS: {status}\n\nBEST BASE CHECKPOINT TOKENS: {final}\nWIKITEXT PPL: {metrics['wikitext_ppl']}\nWIKITEXT BPB: {metrics['wikitext_bpb']}\nHELLASWAG: {metrics['hellaswag']}\nARC-EASY: {metrics['arc_easy']}\nPIQA: {metrics['piqa']}\nWINOGRANDE: {metrics['winogrande']}\nHF REVISION: {best['hf_revision']}\nSHA256: {best['checkpoint_sha256']}\nEXACT RESUME VERIFIED: YES\n\nMORE BASE SCALING JUSTIFIED: NO\nBASE READY FOR POST-TRAINING RESEARCH: YES\nNEXT HIGHEST-VALUE ACTION: Obtain organizer clarification, then design one bounded post-training experiment. Do not launch it.\n";(ART/"final_scale_report.md").write_text(report)
 table=ART/"latticelm_master_results.csv";header=next(csv.reader(table.open()));record={"run ID":"Co4-32-DATA-D-v3-FINAL-SCALE","model":"Co4-32","parameters":PARAMS,"data regime":"DATA-D-BROAD-v3","LR fraction":0,"total tokens":final,"DATA-D tokens":final,"LatticeReason tokens":0,"tokens/parameter":final/PARAMS,"DATA-D validation loss":rs[-1]["data_d_validation_loss"],"WikiText PPL":metrics["wikitext_ppl"],"WikiText BPB":metrics["wikitext_bpb"],"HellaSwag":metrics["hellaswag"],"ARC-Easy":metrics["arc_easy"],"PIQA":metrics["piqa"],"WinoGrande":metrics["winogrande"],"tok/s":rs[-1]["tokens_per_second"],"wall time":wall,"seed":271828,"checkpoint SHA":best["checkpoint_sha256"],"HF revision":best["hf_revision"],"status":"VALID_COMPLETE"}
 with table.open("a",newline="") as f:csv.DictWriter(f,fieldnames=header).writerow(record)
 return final,status,active,wall
def execute(s):
 try:gate(s)
 except Exception as exc:(ART/"final_scale_blocker_report.md").write_text(f"# Final scale blocker\n\nProduction was not launched.\n\n{type(exc).__name__}: {exc}\n");save(s,"BLOCKED",blocker=str(exc));return
 drill=ART/"final_scale_live_drill.json";run(s,"complete-smoke-recovery-drill",[sys.executable,"scripts/final_scale_live_drill.py","--output",str(drill)])
 if json.loads(drill.read_text()).get("status")!="PASS":raise RuntimeError("full smoke/recovery drill failed")
 if not s.get("experiment_budget_start_time"):
  t=datetime.now(timezone.utc);s["experiment_budget_start_time"]=t.isoformat();s["soft_deadline"]=(t+timedelta(hours=68)).isoformat();s["hard_deadline"]=(t+timedelta(hours=72)).isoformat();save(s,"PRODUCTION")
 hard=datetime.fromisoformat(s["hard_deadline"]).timestamp();results=s.get("results",{})
 for target in MILESTONES:
  _,st=checkpoint();current=int(st["tokens_seen"])
  if current<target:
   if time.time()>=datetime.fromisoformat(s["soft_deadline"]).timestamp():raise TimeoutError("soft deadline: refusing next milestone")
   save(s,f"TRAIN_{target}",current_tokens=current);code=run(s,f"train-{target}",[sys.executable,"scripts/train_final_capacity.py","--manifest",str(MANIFEST),"--target",str(target),"--run-id",RUN,"--backend","fp32_compile","--threads","16","--microbatch","8","--resume","--curve",str(TRAIN_CURVE),"--hard-deadline-epoch",str(hard)],True)
   if code==75:raise TimeoutError("trainer stopped safely for hard-deadline reserve")
  cp,_=copy_milestone(target);event("MILESTONE",tokens=target,sha256=sha256_file(cp))
  if not continuation_ok(target):raise InterruptedError(f"continuation gate STOP at {target}")
  if target in (NORMALIZED["M1"],FINAL_TARGET):
   save(s,f"EVALUATE_{target}");wiki,gibc=evaluate(s,target,target==FINAL_TARGET);results[str(target)]=publish(s,target,wiki,gibc);s["results"]=results;save(s)
 final,status,active,wall=reports(s,results);save(s,"COMPLETE",final_tokens=final,scaling_status=status,active_training_seconds=active,total_wall_seconds=wall);event("MASTER_COMPLETE")
 paths=["artifacts/final_scale_report.md","artifacts/final_scale_curve.csv","artifacts/final_scale_milestones.csv","artifacts/final_scale_decision.md","artifacts/final_scale_benchmark_deltas.csv","artifacts/final_scale_runtime.csv","artifacts/final_scale_master_state.json","artifacts/latticelm_master_results.csv"]
 subprocess.run(["git","add","--",*paths],cwd=ROOT,check=True);subprocess.run(["git","commit","-m","Record final-scale base pretraining results"],cwd=ROOT,check=True);subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--status",action="store_true");p.add_argument("--stop",action="store_true");a=p.parse_args()
 if a.status:print(json.dumps(load(),indent=2));return 0
 if a.stop:request_stop();print("graceful stop requested");return 0
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 with Lock():
  s=load() or initial()
  if s.get("current_stage") in {"COMPLETE","BLOCKED"}:return 0
  try:execute(s)
  except (InterruptedError,TimeoutError) as exc:save(s,"STOPPED_SAFE",last_error=str(exc))
  except Exception as exc:save(s,"FAILED",last_error=f"{type(exc).__name__}: {exc}");raise
 return 0
if __name__=="__main__":raise SystemExit(main())
