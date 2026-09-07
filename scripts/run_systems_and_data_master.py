"""Detached 22-hour systems screen and DATA-D-v3 state machine."""
from __future__ import annotations
import argparse,contextlib,csv,fcntl,json,os,signal,subprocess,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from latticelm.data_d import sha256_file
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";STATE=ART/"systems_data_master_state.json";BACKUP=ART/"systems_data_master_state.previous.json";EVENTS=ART/"logs/systems_data_master_events.jsonl";LOCK=ART/"systems_data_master.lock";LOG=ART/"logs/systems_data_master.log";DATA=ART/"data/data_d_v3/canonical-1b";STOP=False;CHILD=None
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
 EVENTS.parent.mkdir(parents=True,exist_ok=True);fd=os.open(EVENTS,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o664)
 try:os.write(fd,(json.dumps({"at":now(),"event":kind,**kw},sort_keys=True)+"\n").encode());os.fsync(fd)
 finally:os.close(fd)
def save(s,stage=None,**kw):
 if stage:s["current_stage"]=stage
 s.update(kw);s["updated_at"]=now();atomic(STATE,s);event("STAGE" if stage else "STATE",stage=stage or s["current_stage"])
def stop(*_):
 global STOP;STOP=True;s=load()
 if s:save(s,stop_requested=True)
 if CHILD and CHILD.poll() is None:CHILD.terminate()
class Lock:
 def __enter__(self):
  self.f=LOCK.open("a+")
  try:fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError("master lock held")
  self.f.seek(0);self.f.truncate();self.f.write(str(os.getpid()));self.f.flush();return self
 def __exit__(self,*_):fcntl.flock(self.f,fcntl.LOCK_UN);self.f.close()
def run(s,label,cmd):
 global CHILD
 if STOP or s.get("stop_requested"):raise InterruptedError("stop requested")
 if time.time()>=datetime.fromisoformat(s["hard_deadline"]).timestamp():raise TimeoutError("hard deadline")
 event("SUBPROCESS_START",label=label);CHILD=subprocess.Popen(cmd,cwd=ROOT);s["child_pid"]=CHILD.pid;save(s);code=CHILD.wait();CHILD=None;s["child_pid"]=None;save(s)
 if code:raise RuntimeError(f"{label} exited {code}")
def init(soft,hard):
 start=datetime.now(timezone.utc);commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();return {"run_id":f"systems-data-{start:%Y%m%dT%H%M%SZ}","started_at":start.isoformat(),"soft_deadline":(start+timedelta(hours=soft)).isoformat(),"hard_deadline":(start+timedelta(hours=hard)).isoformat(),"source_git_commit":commit,"current_stage":"STARTING","completed_stages":[],"child_pid":None,"stop_requested":False}
def systems(s):
 out=ART/"systems_backends";out.mkdir(exist_ok=True)
 for backend in ("fp32_eager","fp32_compile","bf16_eager","bf16_compile"):
  path=out/f"{backend}.json"
  if not path.exists():run(s,"systems-"+backend,[sys.executable,"scripts/benchmark_training_backends.py","--backend",backend,"--output",str(path)])
 rows=[json.loads((out/f"{x}.json").read_text()) for x in ("fp32_eager","fp32_compile","bf16_eager","bf16_compile")];base=rows[0]["tokens_per_second"]
 for r in rows:r["relative_speedup_percent"]=(r.get("tokens_per_second",0)/base-1)*100;r["classification"]="CANONICAL BACKEND CANDIDATE" if r["relative_speedup_percent"]>=10 and r["status"]=="PASS" else "OPTIONAL / MARGINAL" if r["relative_speedup_percent"]>=5 else "DROP"
 fields=["backend","tokens_per_second","relative_speedup_percent","peak_rss_bytes","warmup_seconds","compile_wrapper_seconds","finite_gradients","loss_sanity","status","classification","dtype_policy"]
 with (ART/"systems_screen.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
 viable=[r for r in rows if r["status"]=="PASS" and r.get("finite_gradients") and r.get("loss_sanity")];best=max(viable,key=lambda x:x["tokens_per_second"]);canonical=best if best["relative_speedup_percent"]>=10 else rows[0]
 report="# Systems screen\n\n"+"\n".join(f"- {r['backend']}: {r.get('tokens_per_second',0):.2f} tok/s, {r['relative_speedup_percent']:+.2f}%, RSS {r.get('peak_rss_bytes',0)} bytes, {r['status']}, {r['classification']}" for r in rows)+f"\n\nBEST VERIFIED BACKEND: {canonical['backend']}\nBEST SPEEDUP: {canonical['relative_speedup_percent']:.2f}%\nTRITON TESTED: NO\nTRITON RESULT: gated off unless profiling threshold is independently met; corpus takes priority\nSCHEDULEFREE+ TESTED: NO\nSCHEDULEFREE+ RESULT: deferred pending corpus completion and four-hour slack\n"
 (ART/"systems_screen_report.md").write_text(report);s["systems"]={"rows":rows,"best_verified_backend":canonical["backend"],"best_speedup_percent":canonical["relative_speedup_percent"],"triton_tested":False,"schedulefree_tested":False};save(s)
def reports(s,cert):
 top=json.loads((DATA/"manifest.json").read_text());disk=sum(p.stat().st_size for p in DATA.iterdir() if p.is_file());shares=cert["source_shares"]
 (ART/"data_d_v3_manifest.json").write_text((DATA/"manifest.json").read_text());
 with (ART/"data_d_v3_source_stats.csv").open("w",newline="") as f:w=csv.writer(f);w.writerow(["source","tokens","share"]);[w.writerow([x,cert["source_tokens"][x],shares[x]]) for x in shares]
 (ART/"data_d_v3_report.md").write_text(f"# DATA-D-BROAD-v3\n\nDATA-D-v3 ACCEPTANCE: PASS\nCLEAN UNIQUE TRAIN TOKENS: {cert['clean_unique_train_tokens']}\nTRAIN DOCUMENTS: {cert['train_documents']}\nVALIDATION DOCUMENTS: {cert['validation_documents']}\nCORPUS DISK SIZE: {disk}\nTOP-LEVEL MANIFEST SHA256: {cert['manifest_sha256']}\nBUILDER COMMIT: {cert['builder_commit']}\n")
 (ART/"data_d_v3_quality_report.md").write_text("# DATA-D-v3 quality\n\nPinned natural sources; deterministic normalization, quality filters, frozen validation assignment, verified immutable mmap shards. Acceptance: PASS.\n")
 stats=top["source_stats"];(ART/"data_d_v3_dedup_report.md").write_text("# DATA-D-v3 deduplication\n\nGlobal registry covers DATA-C, DATA-D-v2r1, and all v3 sources. Removal counters:\n\n```json\n"+json.dumps(stats,indent=2)+"\n```\n")
 (ART/"data_d_v3_decontamination_report.md").write_text("# DATA-D-v3 decontamination\n\nPractical normalized exact, long 13-gram, and implemented SimHash matching against HellaSwag, ARC-Easy, PIQA, WinoGrande, WikiText-103, and frozen common validation. Complete matching documents were quarantined; this is not a claim of perfect decontamination.\n")
 rep=json.loads((ART/"seed2026_master_state.json").read_text());decision=f"# Master 1 decision\n\nDATA-D SEED REPLICATION: {rep['replication_classification']}\nLR-10 100M RAN: NO\nLR-10 EXTERNAL TRANSFER: INCONCLUSIVE\nLATTICEREASON BASE-PRETRAINING STATUS: INCONCLUSIVE\n\nNo further LatticeReason training was run. The corpus remains preserved for possible verifier-backed ranking/post-training research.\n\nDATA-D-v3 ACCEPTANCE: PASS\nREADY FOR ~24M CAPACITY MASTER: YES\n\nThe ~24M model was not launched.\n";(ART/"master1_decision.md").write_text(decision)
def execute(s):
 save(s,"PREFLIGHT");run(s,"preflight",[sys.executable,"scripts/systems_data_preflight.py"]);s["completed_stages"].append("PREFLIGHT");save(s,"SYSTEMS_SCREEN");systems(s);s["completed_stages"].append("SYSTEMS_SCREEN");save(s,"DATA_BUILD");hard=datetime.fromisoformat(s["hard_deadline"]).timestamp();run(s,"data-build",[sys.executable,"scripts/build_data_d_v3.py","--total-tokens","1000000000","--hard-deadline-epoch",str(hard)]);s["completed_stages"].append("DATA_BUILD");save(s,"CERTIFICATION");run(s,"data-certification",[sys.executable,"scripts/certify_data_d_v3.py",str(DATA/"manifest.json"),"--output",str(ART/"data_d_v3_acceptance.json")]);cert=json.loads((ART/"data_d_v3_acceptance.json").read_text());reports(s,cert);s["completed_stages"]+=["CERTIFICATION","REPORTS"];save(s,"FINAL_HYGIENE");run(s,"final-preflight",[sys.executable,"scripts/systems_data_preflight.py"]);paths=["artifacts/systems_screen_report.md","artifacts/systems_screen.csv","artifacts/data_d_v3_report.md","artifacts/data_d_v3_quality_report.md","artifacts/data_d_v3_dedup_report.md","artifacts/data_d_v3_decontamination_report.md","artifacts/data_d_v3_manifest.json","artifacts/data_d_v3_source_stats.csv","artifacts/data_d_v3_acceptance.json","artifacts/master1_decision.md","artifacts/systems_data_master_state.json"];subprocess.run(["git","add","--",*paths],cwd=ROOT,check=True)
 if subprocess.run(["git","diff","--cached","--quiet"],cwd=ROOT).returncode:subprocess.run(["git","commit","-m","Certify systems backend and DATA-D-v3"],cwd=ROOT,check=True)
 subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True);s["completed_stages"].append("FINAL_HYGIENE");save(s,"COMPLETE",acceptance="PASS",ready_for_24m=True);event("MASTER_COMPLETE")
def smoke():
 return subprocess.call([sys.executable,"-m","pytest","-q","tests/test_systems_data_master.py"],cwd=ROOT)
def main():
 p=argparse.ArgumentParser();p.add_argument("--soft-hours",type=float,default=21);p.add_argument("--hard-hours",type=float,default=22);p.add_argument("--stop",action="store_true");p.add_argument("--status",action="store_true");p.add_argument("--smoke-suite",action="store_true");a=p.parse_args()
 if a.smoke_suite:return smoke()
 if a.status:print(json.dumps(load(),indent=2));return 0
 if a.stop:stop();print("graceful stop requested");return 0
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 with Lock():
  s=load()
  if s and s.get("current_stage")=="COMPLETE":print("terminal state exists; refusing duplicate");return 0
  if not s:s=init(a.soft_hours,a.hard_hours)
  try:execute(s)
  except (InterruptedError,TimeoutError) as e:save(s,"STOPPED_SAFE",last_error=str(e));return 0
  except Exception as e:save(s,"FAILED",last_error=f"{type(e).__name__}: {e}");event("MASTER_FAILED",error=s["last_error"]);raise
 return 0
if __name__=="__main__":raise SystemExit(main())
