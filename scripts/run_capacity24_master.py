"""Dependency-waiting autonomous 24M capacity master."""
from __future__ import annotations
import argparse,contextlib,csv,fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import torch
from huggingface_hub import get_token,hf_hub_download
from latticelm.data_d import sha256_file,verify_top_manifest
from latticelm.hf_storage import export_checkpoint,upload_checkpoint
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";PREDECESSOR="latticelm-systems-data-20260907.service";PRE_STATE=ART/"systems_data_master_state.json";PRE_EVENTS=ART/"logs/systems_data_master_events.jsonl";DATA=ART/"data/data_d_v3/canonical-1b";MANIFEST=DATA/"manifest.json";TOK=ART/"tokenizers/babylm_2026_4k.json";TOK_REPORT=ART/"tokenizers/babylm_2026_4k.report.json";STATE=ART/"capacity24_master_state.json";BACKUP=ART/"capacity24_master_state.previous.json";EVENTS=ART/"logs/capacity24_master_events.jsonl";LOCK=ART/"capacity24_master.lock";LOG=ART/"logs/capacity24_master.log";REPO="insightlabs38-pixel/LatticeLM-research";RUN="co4-capacity24-data-d-v3";STOP=False;CHILD=None
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
 launched=datetime.now(timezone.utc);return {"run_id":f"capacity24-{launched:%Y%m%dT%H%M%SZ}","service_launch_time":launched.isoformat(),"experiment_budget_start_time":None,"soft_deadline":None,"hard_deadline":None,"source_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"predecessor":{"unit":PREDECESSOR,"state":str(PRE_STATE),"events":str(PRE_EVENTS),"reports":["artifacts/systems_screen_report.md","artifacts/systems_screen.csv","artifacts/data_d_v3_report.md","artifacts/data_d_v3_quality_report.md","artifacts/data_d_v3_dedup_report.md","artifacts/data_d_v3_decontamination_report.md"],"manifest":str(MANIFEST)},"current_stage":"WAITING_FOR_PREDECESSOR","completed_stages":[],"child_pid":None,"stop_requested":False}
def wait_predecessor(s):
 save(s,"WAITING_FOR_PREDECESSOR")
 while True:
  props=systemd_props()
  if props.get("ActiveState") not in {"active","activating","reloading"}:break
  if STOP or s.get("stop_requested"):raise InterruptedError("stop requested while waiting")
  time.sleep(60)
 props=systemd_props();event("PREDECESSOR_TERMINATED",**props)
 if props.get("Result")!="success" or props.get("ExecMainStatus")!="0":raise RuntimeError(f"predecessor systemd failure: {props}")
def gate(s):
 pre=json.loads(PRE_STATE.read_text())
 if pre.get("current_stage")!="COMPLETE" or pre.get("acceptance")!="PASS":raise RuntimeError("predecessor master not accepted")
 if not PRE_EVENTS.exists() or '"MASTER_COMPLETE"' not in PRE_EVENTS.read_text():raise RuntimeError("predecessor completion event absent")
 top=verify_top_manifest(MANIFEST,TOK);cert=json.loads((DATA/"certification.json").read_text());accept=json.loads((ART/"data_d_v3_acceptance.json").read_text())
 required={"acceptance":"PASS","train_validation_disjointness":"PASS","all_shards_mmap_and_hash":"PASS","exact_resume":"PASS","stable_unique_ids":"PASS"}
 if any(cert.get(k)!=v for k,v in required.items()) or accept.get("acceptance")!="PASS":raise RuntimeError("DATA-D-v3 certification gate failed")
 if cert.get("manifest_sha256")!=sha256_file(MANIFEST) or top["total_unique_tokens"]<150_000_000:raise RuntimeError("manifest hash or token sufficiency gate failed")
 if shutil.disk_usage(ROOT).free<15*1024**3:raise RuntimeError("insufficient disk reserve")
 rows=list(csv.DictReader((ART/"systems_screen.csv").open()));counters=json.loads((ART/"systems_backends/fp32_compile.json").read_text()).get("compile_counters",{});backend,gain,unique=select_backend(rows,counters)
 s["predecessor_gate"]={"acceptance":"PASS","certified_tokens":top["total_unique_tokens"],"manifest_sha256":sha256_file(MANIFEST),"tokenizer_sha256":sha256_file(TOK),"backend":backend,"compile_speedup_percent":gain,"unique_graphs":unique};save(s,"PREDECESSOR_GATE_PASS")
 if not s.get("experiment_budget_start_time"):
  start=datetime.now(timezone.utc);s["experiment_budget_start_time"]=start.isoformat();s["soft_deadline"]=(start+timedelta(hours=21)).isoformat();s["hard_deadline"]=(start+timedelta(hours=22)).isoformat()
 save(s,"LIVE_CERTIFICATION");return backend
def publish(s,target,wiki,gibc=None):
 root=ART/"checkpoints"/RUN;native=root/"latest.pt";milestone=root/f"milestone-{target}.pt";shutil.copy2(native,milestone);shutil.copy2(native.with_suffix(".sha256"),milestone.with_suffix(".sha256"));row=list(csv.DictReader((ART/"capacity24_training_curve.csv").open()))[-1]
 public=ART/"capacity24_dataset_manifest.json";public.write_text(json.dumps({"dataset":"DATA-D-BROAD-v3","revision":sha256_file(MANIFEST),"sources":["FineWeb-Edu","English Wikipedia","FineWeb"],"canonical_manifest":str(MANIFEST.relative_to(ROOT))},indent=2)+"\n");bundle=ART/"cache"/f"capacity24-{target}";metrics={"tokens_trained":target,"wall_seconds":float(row["cumulative_training_seconds"]),"val_loss":float(row["data_d_validation_loss"]),"val_ppl":math.exp(float(row["data_d_validation_loss"])),"wikitext_ppl":wiki["perplexity"],"wikitext_bpb":wiki["bits_per_byte"]}
 export_checkpoint(milestone,bundle,f"capacity24-{target}","immutable-milestone",TOK,TOK_REPORT,public,metrics);token=get_token()
 if not token:raise RuntimeError("Hugging Face credential unavailable")
 remote=f"experiments/capacity24/{target}";revision=upload_checkpoint(bundle,REPO,remote,token);verified={}
 for line in (bundle/"SHA256SUMS").read_text().splitlines():
  expected,name=line.split(None,1);path=Path(hf_hub_download(REPO,f"{remote}/{name.strip()}",revision=revision,token=token));observed=sha256_file(path)
  if observed!=expected:raise RuntimeError("HF hash verification failure")
  verified[name.strip()]=observed
 result={"tokens":target,"native_checkpoint_sha256":sha256_file(milestone),"hf_revision":revision,"remote_hashes":verified,"wikitext":wiki,"gibc":gibc};out=ART/f"capacity24_{target//1_000_000}m_metrics.json";out.write_text(json.dumps(result,indent=2)+"\n");return result
def analyze(s,results):
 control=json.loads((ART/"data_d_fallback_100m/metrics.json").read_text());rows=list(csv.DictReader((ART/"capacity24_training_curve.csv").open()));by={int(x["nominal_tokens"]):x for x in rows};p150=results[150_000_000]["wikitext"]["perplexity"];adv=p150<control["wikitext_ppl"] and float(by[150_000_000]["data_d_validation_loss"])<control["data_d_validation_loss"];losses=[float(by[x]["data_d_validation_loss"]) for x in (50_000_000,100_000_000,150_000_000)];slope="FAVORABLE" if losses[2]<losses[1]<losses[0] else "FLAT" if losses[2]<=losses[1] else "UNFAVORABLE";result="STRONG POSITIVE" if adv and slope=="FAVORABLE" else "MODEST POSITIVE" if adv else "NEGATIVE" if slope=="UNFAVORABLE" and not adv else "FLAT";speed=float(by[150_000_000]["tokens_per_second"])
 with (ART/"capacity24_scaling.csv").open("w",newline="") as f:w=csv.writer(f);w.writerow(["model","parameters","tokens","tokens_per_parameter","data_d_validation_loss","wikitext_ppl","wikitext_bpb"]);w.writerow(["Co4-L",15949760,100000000,100000000/15949760,control["data_d_validation_loss"],control["wikitext_ppl"],control["wikitext_bpb"]]);[w.writerow(["Co4-24",24097200,t,t/24097200,by[t]["data_d_validation_loss"],results[t]["wikitext"]["perplexity"],results[t]["wikitext"]["bits_per_byte"]]) for t in (50_000_000,100_000_000,150_000_000)]
 with (ART/"capacity24_deadline_projection.csv").open("w",newline="") as f:w=csv.writer(f);w.writerow(["hours","projected_tokens_24m","basis","measured"]);[w.writerow([h,int(speed*h*3600),"terminal observed throughput extrapolation",False]) for h in (22,44,66,88)]
 g=results[150_000_000]["gibc"]["results"];report=f"# 24M capacity report\n\nEXACT PARAMETER COUNT: 24097200\nCANONICAL BACKEND: {s['predecessor_gate']['backend']}\nMEASURED 24M TOK/S: {speed}\n50M COMPLETED: YES\n100M COMPLETED: YES\n150M COMPLETED: YES\n24M@50M WIKITEXT: {results[50_000_000]['wikitext']['perplexity']}\n24M@100M WIKITEXT: {results[100_000_000]['wikitext']['perplexity']}\n24M@150M WIKITEXT: {p150}\n24M@150M HELLASWAG: {g['hellaswag']['acc,none']}\n24M@150M ARC-EASY: {g['arc_easy']['acc,none']}\n24M@150M PIQA: {g['piqa']['acc,none']}\n24M@150M WINOGRANDE: {g['winogrande']['acc,none']}\nMATCHED-T/P CAPACITY ADVANTAGE: {'YES' if adv else 'NO'}\nSCALING SLOPE: {slope}\nDEADLINE-EFFICIENCY OUTLOOK: projections are trajectory-based estimates, not measured outcomes\n24M CAPACITY RESULT: {result}\n32M CAPACITY TEST RECOMMENDED: {'YES' if result in {'STRONG POSITIVE','MODEST POSITIVE'} else 'NO'}\n";(ART/"capacity24_report.md").write_text(report);(ART/"capacity24_decision.md").write_text(report);return {"matched_advantage":adv,"scaling_slope":slope,"capacity_result":result,"recommend_32m":result in {"STRONG POSITIVE","MODEST POSITIVE"}}
def execute(s):
 wait_predecessor(s)
 try:backend=gate(s)
 except Exception as exc:
  (ART/"capacity24_blocker_report.md").write_text(f"# Capacity24 blocker\n\n24M training was not launched.\n\n{type(exc).__name__}: {exc}\n");save(s,"BLOCKED",blocker=str(exc));return
 drill=ART/"capacity24_live_drill.json"
 if not drill.exists() or json.loads(drill.read_text()).get("status")!="PASS":run(s,"live-smoke-recovery",[sys.executable,"scripts/capacity24_live_drill.py","--output",str(drill)])
 if "LIVE_CERTIFICATION" not in s["completed_stages"]:s["completed_stages"].append("LIVE_CERTIFICATION")
 save(s,"CALIBRATION");calpath=ART/"capacity24_calibration.json"
 if not calpath.exists() or json.loads(calpath.read_text()).get("status")!="PASS":run(s,"calibration",[sys.executable,"scripts/calibrate_capacity24.py","--backend",backend,"--output",str(calpath)])
 cal=json.loads(calpath.read_text())["selected"];s["calibration"]=cal;results={};hard=datetime.fromisoformat(s["hard_deadline"]).timestamp()
 for target in (50_000_000,100_000_000,150_000_000):
  result_path=ART/f"capacity24_{target//1_000_000}m_metrics.json"
  if result_path.exists():results[target]=json.loads(result_path.read_text());continue
  latest=ART/"checkpoints"/RUN/"latest.pt";current=0
  if latest.exists() and latest.with_suffix(".sha256").exists() and sha256_file(latest)==latest.with_suffix(".sha256").read_text().strip():current=int(torch.load(latest,map_location="cpu",weights_only=False).get("tokens_seen",0))
  if current<target:
   if time.time()>=datetime.fromisoformat(s["soft_deadline"]).timestamp():raise TimeoutError("soft deadline: refusing new milestone")
   save(s,f"TRAIN_{target//1_000_000}M");mode="--fresh" if target==50_000_000 and not latest.exists() else "--resume";code=run(s,f"train-{target}",[sys.executable,"scripts/train_capacity24.py","--manifest",str(MANIFEST),"--target",str(target),"--run-id",RUN,"--backend",backend,"--threads",str(cal["threads"]),"--microbatch",str(cal["microbatch"]),mode,"--hard-deadline-epoch",str(hard)],True)
   if code==75:raise TimeoutError("capacity training stopped safely before milestone")
  elif current>target:raise RuntimeError("missing immutable earlier milestone result")
  save(s,f"EVALUATE_{target//1_000_000}M");native=latest;wiki_path=ART/f"capacity24_{target//1_000_000}m_wikitext.json"
  if not wiki_path.exists():run(s,f"wiki-{target}",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(native),"--tokenizer",str(TOK),"--output",str(wiki_path),"--threads","4"])
  wiki=json.loads(wiki_path.read_text());gibc=None
  if target==150_000_000:
   gp=ART/"capacity24_150m_gibc.json"
   if not gp.exists():run(s,"gibc-150m",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(native),"--tokenizer",str(TOK),"--output",str(gp),"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","4"])
   gibc=json.loads(gp.read_text())
  results[target]=publish(s,target,wiki,gibc);s["completed_stages"].append(f"{target//1_000_000}M")
 save(s,"CAPACITY_ANALYSIS");decision=analyze(s,results);s["decision"]=decision;s["completed_stages"].append("CAPACITY_ANALYSIS");save(s,"FINAL_HYGIENE");run(s,"final-tests",[sys.executable,"scripts/systems_data_preflight.py"]);s["completed_stages"].append("FINAL_HYGIENE");save(s,"COMPLETE",**decision)
 paths=[str(p.relative_to(ROOT)) for p in [STATE,ART/"capacity24_report.md",ART/"capacity24_scaling.csv",ART/"capacity24_deadline_projection.csv",ART/"capacity24_decision.md",ART/"capacity24_calibration.json",ART/"capacity24_live_drill.json",ART/"capacity24_dataset_manifest.json",*[ART/f"capacity24_{x}m_metrics.json" for x in (50,100,150)],*[ART/f"capacity24_{x}m_wikitext.json" for x in (50,100,150)]] if p.exists()];subprocess.run(["git","add","--",*paths],cwd=ROOT,check=True)
 if subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode:subprocess.run(["git","commit","-m","Record 24M matched-capacity experiment"],cwd=ROOT,check=True)
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
