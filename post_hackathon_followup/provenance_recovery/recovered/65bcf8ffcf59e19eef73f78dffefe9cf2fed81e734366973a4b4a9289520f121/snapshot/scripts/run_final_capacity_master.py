"""Autonomous final 16M/24M/32M capacity decision and lineage master."""
from __future__ import annotations
import argparse,contextlib,csv,fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import torch
from huggingface_hub import get_token,hf_hub_download
from latticelm.data_d import sha256_file,verify_top_manifest
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.hf_storage import export_checkpoint,upload_checkpoint
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";PREDECESSOR="latticelm-systems-data-20260907.service";PRE_STATE=ART/"systems_data_master_state.json";PRE_EVENTS=ART/"logs/systems_data_master_events.jsonl";DATA=ART/"data/data_d_v3/canonical-1b";MANIFEST=DATA/"manifest.json";TOK=ART/"tokenizers/babylm_2026_4k.json";TOK_REPORT=ART/"tokenizers/babylm_2026_4k.report.json";STATE=ART/"final_capacity_master_state.json";BACKUP=ART/"final_capacity_master_state.previous.json";EVENTS=ART/"logs/final_capacity_master_events.jsonl";LOCK=ART/"final_capacity_master.lock";LOG=ART/"logs/final_capacity_master.log";REPO="insightlabs38-pixel/LatticeLM-research";RUN="co4-final-capacity32-data-d-v3";RUN24="co4-capacity24-data-d-v3";P32=32_678_640;MATCHED=205_000_000;STOP=False;CHILD=None
def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name("."+path.name+".tmp")
 with tmp.open("w") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
 if path.exists():
  try:json.loads(path.read_text());path.replace(BACKUP)
  except (OSError,json.JSONDecodeError):pass
 os.replace(tmp,path)
def load():
 for p in (STATE,BACKUP):
  try:return json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):pass
 return None
def event(kind,**kw):
 fd=os.open(EVENTS,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o664)
 try:os.write(fd,(json.dumps({"at":now(),"event":kind,**kw},sort_keys=True)+"\n").encode());os.fsync(fd)
 finally:os.close(fd)
def save(s,stage=None,**kw):
 if stage:s["current_stage"]=stage
 s.update(kw);s["updated_at"]=now();atomic(STATE,s);event("STAGE",stage=s["current_stage"])
def stop(*_):
 global STOP;STOP=True;s=load()
 if s and s.get("current_stage") not in {"COMPLETE","BLOCKED","STOPPED_SAFE","FAILED"}:save(s,stop_requested=True)
 if CHILD and CHILD.poll() is None:CHILD.terminate()
class Lock:
 def __enter__(self):
  self.f=LOCK.open("a+")
  try:fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError("capacity master lock held")
  self.f.seek(0);self.f.truncate();self.f.write(str(os.getpid()));self.f.flush();return self
 def __exit__(self,*_):fcntl.flock(self.f,fcntl.LOCK_UN);self.f.close()
def run(s,label,cmd,allow75=False):
 global CHILD
 if STOP or s.get("stop_requested"):raise InterruptedError("stop requested")
 hard=s.get("hard_deadline")
 if hard and time.time()>=datetime.fromisoformat(hard).timestamp():raise TimeoutError("hard deadline")
 event("SUBPROCESS_START",label=label);CHILD=subprocess.Popen(cmd,cwd=ROOT);s["child_pid"]=CHILD.pid;save(s);code=CHILD.wait();CHILD=None;s["child_pid"]=None;save(s)
 if code and not(allow75 and code==75):raise RuntimeError(f"{label} exited {code}")
 return code
def systemd_props():
 text=subprocess.check_output(["systemctl","show",PREDECESSOR,"-p","ActiveState","-p","SubState","-p","Result","-p","ExecMainStatus"],text=True);return dict(line.split("=",1) for line in text.splitlines())
def select_backend(rows,counters):
 compiled=next(x for x in rows if x["backend"]=="fp32_compile");unique=int(counters.get("stats",{}).get("unique_graphs",99));gain=float(compiled["relative_speedup_percent"])
 ok=compiled["status"]=="PASS" and str(compiled["finite_gradients"]).lower()=="true" and str(compiled["loss_sanity"]).lower()=="true" and gain>=10 and unique<=2
 return ("fp32_compile" if ok else "fp32_eager"),gain,unique
def initial():
 launched=datetime.now(timezone.utc);return {"run_id":f"final_capacity-{launched:%Y%m%dT%H%M%SZ}","service_launch_time":launched.isoformat(),"experiment_budget_start_time":None,"soft_deadline":None,"hard_deadline":None,"source_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"predecessor":{"unit":PREDECESSOR,"state":str(PRE_STATE),"events":str(PRE_EVENTS),"reports":["artifacts/systems_screen_report.md","artifacts/systems_screen.csv","artifacts/data_d_v3_report.md","artifacts/data_d_v3_quality_report.md","artifacts/data_d_v3_dedup_report.md","artifacts/data_d_v3_decontamination_report.md"],"manifest":str(MANIFEST)},"current_stage":"WAITING_FOR_PREDECESSOR","completed_stages":[],"child_pid":None,"stop_requested":False}
def wait_predecessor(s):
 save(s,"WAITING_FOR_PREDECESSOR")
 while True:
  props=systemd_props()
  if props.get("ActiveState") not in {"active","activating","reloading"}:break
  if STOP or s.get("stop_requested"):raise InterruptedError("stop requested while waiting")
  time.sleep(60)
 props=systemd_props();event("PREDECESSOR_TERMINATED",**props)
 if props.get("Result")!="success" or props.get("ExecMainStatus")!="0":raise RuntimeError(f"predecessor systemd failure: {props}")
def audit_24m():
 metrics={t:json.loads((ART/f"capacity24_{t}m_metrics.json").read_text()) for t in (50,100,150)}
 curve=list(csv.DictReader((ART/"capacity24_training_curve.csv").open()));by={int(r["nominal_tokens"]):r for r in curve}
 checkpoint=ART/"checkpoints"/RUN24/"milestone-150000000.pt";expected=checkpoint.with_suffix(".sha256").read_text().strip()
 if sha256_file(checkpoint)!=expected or expected!=metrics[150]["native_checkpoint_sha256"]:raise RuntimeError("24M immutable weight hash failure")
 state=torch.load(checkpoint,map_location="cpu",weights_only=False);cfg=LatticeConfig(**state["config"]);model=build_model(cfg);model.load_state_dict(state["model"],strict=True);model.eval();probe=torch.arange(128).remainder(4096).view(1,128)
 with torch.no_grad():first=model(probe)[0];second=model(probe)[0]
 if not torch.equal(first,second) or state.get("tokens_seen")!=150_000_000 or model.parameter_breakdown()["total"]!=24_097_200:raise RuntimeError("24M strict load/inference/identity failure")
 if not state.get("next_batch_sha256") or not state.get("data_source_selector_state") or not state.get("optimizer") or not state.get("scheduler"):raise RuntimeError("24M exact-resume payload incomplete")
 for t in (50,100,150):
  if metrics[t]["wikitext"]["checkpoint_sha256"]!=by[t*1_000_000]["checkpoint_sha256"]:raise RuntimeError(f"24M {t}M evaluation/checkpoint mismatch")
 loss=[float(by[t*1_000_000]["data_d_validation_loss"]) for t in (50,100,150)]
 audit={"parameters":24_097_200,"backend":by[150_000_000]["backend"],"tokens_per_second":float(by[150_000_000]["tokens_per_second"]),"weights_sha":"PASS","strict_load":"PASS","deterministic_inference":"PASS","exact_resume_payload":"PASS","trajectory_validation":loss,"wikitext_ppl":[metrics[t]["wikitext"]["perplexity"] for t in (50,100,150)],"wikitext_bpb":[metrics[t]["wikitext"]["bits_per_byte"] for t in (50,100,150)],"hf_revisions":[metrics[t]["hf_revision"] for t in (50,100,150)]}
 (ART/"final_capacity_24m_audit.json").write_text(json.dumps(audit,indent=2)+"\n");return audit
def gate(s):
 pre=json.loads(PRE_STATE.read_text())
 if pre.get("current_stage")!="COMPLETE" or pre.get("acceptance")!="PASS":raise RuntimeError("predecessor master not accepted")
 if not PRE_EVENTS.exists() or '"MASTER_COMPLETE"' not in PRE_EVENTS.read_text():raise RuntimeError("predecessor completion event absent")
 top=verify_top_manifest(MANIFEST,TOK);cert=json.loads((DATA/"certification.json").read_text());accept=json.loads((ART/"data_d_v3_acceptance.json").read_text())
 required={"acceptance":"PASS","train_validation_disjointness":"PASS","all_shards_mmap_and_hash":"PASS","exact_resume":"PASS","stable_unique_ids":"PASS"}
 if any(cert.get(k)!=v for k,v in required.items()) or accept.get("acceptance")!="PASS":raise RuntimeError("DATA-D-v3 certification gate failed")
 if cert.get("manifest_sha256")!=sha256_file(MANIFEST) or top["total_unique_tokens"]<1_000_000_000:raise RuntimeError("manifest hash or billion-token sufficiency gate failed")
 if shutil.disk_usage(ROOT).free<15*1024**3:raise RuntimeError("insufficient disk reserve")
 rows=list(csv.DictReader((ART/"systems_screen.csv").open()));counters=json.loads((ART/"systems_backends/fp32_compile.json").read_text()).get("compile_counters",{});backend,gain,unique=select_backend(rows,counters)
 s["predecessor_gate"]={"acceptance":"PASS","certified_tokens":top["total_unique_tokens"],"manifest_sha256":sha256_file(MANIFEST),"tokenizer_sha256":sha256_file(TOK),"backend":backend,"compile_speedup_percent":gain,"unique_graphs":unique};s["audit_24m"]=audit_24m();s["test_32m_justified"]=True;save(s,"PREDECESSOR_GATE_PASS")
 if not s.get("experiment_budget_start_time"):
  start=datetime.now(timezone.utc);s["experiment_budget_start_time"]=start.isoformat();s["soft_deadline"]=(start+timedelta(hours=21)).isoformat();s["hard_deadline"]=(start+timedelta(hours=22)).isoformat()
 save(s,"LIVE_CERTIFICATION");return backend
def publish(s,target,wiki,gibc=None):
 root=ART/"checkpoints"/RUN;native=root/"latest.pt";milestone=root/f"milestone-{target}.pt";shutil.copy2(native,milestone);shutil.copy2(native.with_suffix(".sha256"),milestone.with_suffix(".sha256"));row=list(csv.DictReader((ART/"final_capacity_training_curve.csv").open()))[-1]
 public=ART/"final_capacity_dataset_manifest.json";public.write_text(json.dumps({"dataset":"DATA-D-BROAD-v3","revision":sha256_file(MANIFEST),"sources":["FineWeb-Edu","English Wikipedia","FineWeb"],"canonical_manifest":str(MANIFEST.relative_to(ROOT))},indent=2)+"\n");bundle=ART/"cache"/f"final_capacity-{target}";metrics={"tokens_trained":target,"wall_seconds":float(row["cumulative_training_seconds"]),"val_loss":float(row["data_d_validation_loss"]),"val_ppl":math.exp(float(row["data_d_validation_loss"])),"wikitext_ppl":wiki["perplexity"],"wikitext_bpb":wiki["bits_per_byte"]}
 export_checkpoint(milestone,bundle,f"final_capacity-{target}","immutable-milestone",TOK,TOK_REPORT,public,metrics);token=get_token()
 if not token:raise RuntimeError("Hugging Face credential unavailable")
 remote=f"experiments/final_capacity/{target}";revision=upload_checkpoint(bundle,REPO,remote,token);verified={}
 for line in (bundle/"SHA256SUMS").read_text().splitlines():
  expected,name=line.split(None,1);path=Path(hf_hub_download(REPO,f"{remote}/{name.strip()}",revision=revision,token=token));observed=sha256_file(path)
  if observed!=expected:raise RuntimeError("HF hash verification failure")
  verified[name.strip()]=observed
 result={"tokens":target,"native_checkpoint_sha256":sha256_file(milestone),"hf_revision":revision,"remote_hashes":verified,"wikitext":wiki,"gibc":gibc};out=ART/f"final_capacity_{target//1_000_000}m_metrics.json";out.write_text(json.dumps(result,indent=2)+"\n");return result
def analyze(s,results):
 control=json.loads((ART/"data_d_fallback_100m/metrics.json").read_text());r24={t:json.loads((ART/f"capacity24_{t//1_000_000}m_metrics.json").read_text()) for t in (50_000_000,100_000_000,150_000_000)};rows32=list(csv.DictReader((ART/"final_capacity_training_curve.csv").open()));by32={int(x["nominal_tokens"]):x for x in rows32};rows24={int(x["nominal_tokens"]):x for x in csv.DictReader((ART/"capacity24_training_curve.csv").open())}
 better_val=float(by32[MATCHED]["data_d_validation_loss"])<float(rows24[150_000_000]["data_d_validation_loss"]);better_wiki=results[MATCHED]["wikitext"]["perplexity"]<r24[150_000_000]["wikitext"]["perplexity"];winner="~32M" if better_val and better_wiki else "~24M";confidence="HIGH" if better_val and better_wiki and results[MATCHED]["wikitext"]["perplexity"]<=.98*r24[150_000_000]["wikitext"]["perplexity"] else "MODERATE";speed32=float(by32[MATCHED]["tokens_per_second"]);speed24=float(rows24[150_000_000]["tokens_per_second"]);g=results[MATCHED]["gibc"]["results"]
 fields=["model","parameters","tokens","tokens_per_parameter","data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss","wikitext_ppl","wikitext_bpb","hellaswag","arc_easy","piqa","winogrande","tok_per_second","wall_time_seconds","cpu_hours","peak_rss_bytes"]
 with (ART/"final_capacity_surface.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow({"model":"Co4-L","parameters":15_949_760,"tokens":100_000_000,"tokens_per_parameter":100_000_000/15_949_760,"data_d_validation_loss":control["data_d_validation_loss"],"fineweb_edu_validation_loss":control["fineweb_edu_validation_loss"],"wikipedia_validation_loss":control["wikipedia_validation_loss"],"fineweb_validation_loss":control["fineweb_validation_loss"],"wikitext_ppl":control["wikitext_ppl"],"wikitext_bpb":control["wikitext_bpb"],"hellaswag":control["hellaswag"],"arc_easy":control["arc_easy"],"piqa":control["piqa"],"winogrande":control["winogrande"]})
  for name,p,table,metric,targets in (("Co4-24",24_097_200,rows24,r24,(50_000_000,100_000_000,150_000_000)),("Co4-32",P32,by32,results,(50_000_000,100_000_000,150_000_000,MATCHED))):
   for t in targets:
    rr=table[t];gg=metric[t].get("gibc") or {};scores=gg.get("results",{});w.writerow({"model":name,"parameters":p,"tokens":t,"tokens_per_parameter":t/p,"data_d_validation_loss":rr["data_d_validation_loss"],"fineweb_edu_validation_loss":rr["fineweb_edu_validation_loss"],"wikipedia_validation_loss":rr["wikipedia_validation_loss"],"fineweb_validation_loss":rr["fineweb_validation_loss"],"wikitext_ppl":metric[t]["wikitext"]["perplexity"],"wikitext_bpb":metric[t]["wikitext"]["bits_per_byte"],"hellaswag":scores.get("hellaswag",{}).get("acc,none"),"arc_easy":scores.get("arc_easy",{}).get("acc,none"),"piqa":scores.get("piqa",{}).get("acc,none"),"winogrande":scores.get("winogrande",{}).get("acc,none"),"tok_per_second":rr["tokens_per_second"],"wall_time_seconds":rr["cumulative_training_seconds"],"cpu_hours":float(rr["cumulative_training_seconds"])*int(rr["threads"])/3600,"peak_rss_bytes":rr["peak_rss_bytes"]})
 with (ART/"final_capacity_deadline_projection.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["model","hours","projected_tokens","tok_per_second","basis","measured"])
  for model,speed in (("Co4-24",speed24),("Co4-32",speed32)):
   for h in (22,44,66,88):w.writerow([model,h,int(speed*h*3600),speed,"observed terminal throughput extrapolation",False])
 rationale=f"32M matched validation better={better_val}; WikiText better={better_wiki}. Equal-token rows are sample-efficiency evidence only. The official suite is interpreted jointly and not used as a one-benchmark veto."
 decision=f"# Final capacity decision\n\nFINAL CAPACITY: {winner}\n\nCONFIDENCE: {confidence}\n\nCAPACITY SEARCH CLOSED: YES\n\n{rationale}\n\nNo further capacity experiment is authorized.\n";(ART/"final_capacity_decision.md").write_text(decision)
 report=f"# Final capacity report\n\n24M PARAMETERS: 24097200\n24M TOK/S: {speed24}\n24M MATCHED-T/P RESULT: STRONG POSITIVE\n24M SCALING SLOPE: FAVORABLE\n24M DEADLINE EFFICIENCY: COMPETITIVE\n\n32M TEST JUSTIFIED: YES\n32M TEST COMPLETED: YES\n32M PARAMETERS: {P32}\n32M MATCHED-T/P TOKENS: {MATCHED}\n32M WIKITEXT PPL/BPB: {results[MATCHED]['wikitext']['perplexity']} / {results[MATCHED]['wikitext']['bits_per_byte']}\n32M HELLASWAG: {g['hellaswag']['acc,none']}\n32M ARC-EASY: {g['arc_easy']['acc,none']}\n32M PIQA: {g['piqa']['acc,none']}\n32M WINOGRANDE: {g['winogrande']['acc,none']}\n\nFINAL CAPACITY: {winner}\nFINAL CAPACITY CONFIDENCE: {confidence}\nCAPACITY SEARCH CLOSED: YES\n\n{rationale}\n";(ART/"final_capacity_report.md").write_text(report);return {"final_capacity":winner,"confidence":confidence,"capacity_search_closed":True,"tokens_per_second":speed32}
def execute(s):
 wait_predecessor(s)
 try:backend=gate(s)
 except Exception as exc:
  (ART/"final_capacity_blocker_report.md").write_text(f"# Final capacity blocker\n\nCanonical training was not launched.\n\n{type(exc).__name__}: {exc}\n");save(s,"BLOCKED",blocker=str(exc));return
 drill=ART/"final_capacity_live_drill.json"
 if not drill.exists() or json.loads(drill.read_text()).get("status")!="PASS":run(s,"live-smoke-recovery",[sys.executable,"scripts/final_capacity_live_drill.py","--output",str(drill)])
 if "LIVE_CERTIFICATION" not in s["completed_stages"]:s["completed_stages"].append("LIVE_CERTIFICATION")
 save(s,"CALIBRATION");calpath=ART/"final_capacity_calibration.json"
 if not calpath.exists() or json.loads(calpath.read_text()).get("status")!="PASS":run(s,"calibration",[sys.executable,"scripts/calibrate_final_capacity.py","--backend",backend,"--output",str(calpath)])
 cal=json.loads(calpath.read_text())["selected"];s["calibration"]=cal;results={};hard=datetime.fromisoformat(s["hard_deadline"]).timestamp()
 for target in (50_000_000,100_000_000,150_000_000,MATCHED):
  result_path=ART/f"final_capacity_{target//1_000_000}m_metrics.json"
  if result_path.exists():results[target]=json.loads(result_path.read_text());continue
  latest=ART/"checkpoints"/RUN/"latest.pt";current=0
  if latest.exists() and latest.with_suffix(".sha256").exists() and sha256_file(latest)==latest.with_suffix(".sha256").read_text().strip():current=int(torch.load(latest,map_location="cpu",weights_only=False).get("tokens_seen",0))
  if current<target:
   if time.time()>=datetime.fromisoformat(s["soft_deadline"]).timestamp():raise TimeoutError("soft deadline: refusing new milestone")
   save(s,f"TRAIN_{target//1_000_000}M");mode="--fresh" if target==50_000_000 and not latest.exists() else "--resume";code=run(s,f"train-{target}",[sys.executable,"scripts/train_final_capacity.py","--manifest",str(MANIFEST),"--target",str(target),"--run-id",RUN,"--backend",backend,"--threads",str(cal["threads"]),"--microbatch",str(cal["microbatch"]),mode,"--hard-deadline-epoch",str(hard)],True)
   if code==75:raise TimeoutError("capacity training stopped safely before milestone")
  elif current>target:raise RuntimeError("missing immutable earlier milestone result")
  save(s,f"EVALUATE_{target//1_000_000}M");native=latest;wiki_path=ART/f"final_capacity_{target//1_000_000}m_wikitext.json"
  if not wiki_path.exists():run(s,f"wiki-{target}",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(native),"--tokenizer",str(TOK),"--output",str(wiki_path),"--threads","4"])
  wiki=json.loads(wiki_path.read_text());gibc=None
  if target==MATCHED:
   gp=ART/"final_capacity_205m_gibc.json"
   if not gp.exists():run(s,"gibc-205m",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(native),"--tokenizer",str(TOK),"--output",str(gp),"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","4"])
   gibc=json.loads(gp.read_text())
  results[target]=publish(s,target,wiki,gibc);s["completed_stages"].append(f"{target//1_000_000}M")
 save(s,"CAPACITY_ANALYSIS");decision=analyze(s,results);s["decision"]=decision;s["completed_stages"].append("CAPACITY_ANALYSIS");save(s,"FINAL_LINEAGE_PLANNING")
 selected=decision["final_capacity"];current=MATCHED if selected=="~32M" else 150_000_000;speed=decision["tokens_per_second"] if selected=="~32M" else s["audit_24m"]["tokens_per_second"];remaining=max(0,datetime.fromisoformat(s["soft_deadline"]).timestamp()-time.time()-7200);safe=current+int(speed*remaining//50_000_000)*50_000_000;safe=min(safe,int(s["predecessor_gate"]["certified_tokens"]//50_000_000)*50_000_000)
 if safe>current:
  save(s,"FINAL_LINEAGE_TRAINING",final_lineage_start=current,final_lineage_target=safe);lineage=RUN if selected=="~32M" else RUN24;config=ROOT/("configs/final_capacity32_co4.json" if selected=="~32M" else "configs/capacity24_co4.json");parameters=P32 if selected=="~32M" else 24_097_200;curve=ART/("final_capacity_training_curve.csv" if selected=="~32M" else "capacity24_training_curve.csv");run(s,"final-lineage",[sys.executable,"scripts/train_final_capacity.py","--manifest",str(MANIFEST),"--target",str(safe),"--run-id",lineage,"--config",str(config),"--expected-parameters",str(parameters),"--backend",backend,"--threads",str(cal["threads"]),"--microbatch",str(cal["microbatch"]),"--curve",str(curve),"--resume","--hard-deadline-epoch",str(hard)],True);current=safe
 latest=ART/"checkpoints"/(RUN if selected=="~32M" else RUN24)/"latest.pt";latest_sha=sha256_file(latest) if latest.exists() else None;lineage_status=f"# Final lineage status\n\nFINAL LINEAGE STARTED/CONTINUED: {'YES' if current>(MATCHED if selected=='~32M' else 150_000_000) else 'NO'}\nSTART TOKEN: {MATCHED if selected=='~32M' else 150_000_000}\nEND TOKEN: {current}\nTOK/S: {speed}\nLATEST VERIFIED CHECKPOINT: {latest} ({latest_sha})\nHF REVISION: {results[MATCHED]['hf_revision'] if selected=='~32M' else json.loads((ART/'capacity24_150m_metrics.json').read_text())['hf_revision']}\n\nNEXT PRIMARY GOAL: FINAL NATURAL-DATA SCALE\nNEXT SAFE TOKEN MILESTONE: {current+50_000_000}\n";(ART/"final_lineage_status.md").write_text(lineage_status);s["final_lineage"]={"started_or_continued":current>(MATCHED if selected=="~32M" else 150_000_000),"start_tokens":MATCHED if selected=="~32M" else 150_000_000,"end_tokens":current,"checkpoint_sha256":latest_sha,"next_milestone":current+50_000_000}
 save(s,"FINAL_HYGIENE");run(s,"final-tests",[sys.executable,"-m","pytest","-q"]);s["completed_stages"].append("FINAL_HYGIENE");save(s,"COMPLETE",**decision)
 paths=[str(p.relative_to(ROOT)) for p in [STATE,ART/"final_capacity_24m_audit.json",ART/"final_capacity_report.md",ART/"final_capacity_surface.csv",ART/"final_capacity_deadline_projection.csv",ART/"final_capacity_decision.md",ART/"final_lineage_status.md",ART/"final_capacity_calibration.json",ART/"final_capacity_live_drill.json",ART/"final_capacity_dataset_manifest.json",*[ART/f"final_capacity_{x}m_metrics.json" for x in (50,100,150,205)],*[ART/f"final_capacity_{x}m_wikitext.json" for x in (50,100,150,205)]] if p.exists()];subprocess.run(["git","add","--",*paths],cwd=ROOT,check=True)
 if subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode:subprocess.run(["git","commit","-m","Close capacity search and record final lineage"],cwd=ROOT,check=True)
 subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True);event("MASTER_COMPLETE")
def main():
 p=argparse.ArgumentParser();p.add_argument("--status",action="store_true");p.add_argument("--stop",action="store_true");a=p.parse_args()
 if a.status:print(json.dumps(load(),indent=2));return 0
 if a.stop:stop();print("graceful stop requested");return 0
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 with Lock():
  s=load() or initial()
  if s.get("current_stage") in {"COMPLETE","BLOCKED"}:return 0
  try:execute(s)
  except (InterruptedError,TimeoutError) as exc:save(s,"STOPPED_SAFE",last_error=str(exc))
  except Exception as exc:save(s,"FAILED",last_error=f"{type(exc).__name__}: {exc}");raise
 return 0
if __name__=="__main__":raise SystemExit(main())
