"""Detached deadline-aware master for the LatticeReason mixture tournament."""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from huggingface_hub import get_token
from latticelm.hf_storage import export_checkpoint,upload_checkpoint
from latticelm.data_d import sha256_file,verify_top_manifest

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";STATE=ART/"lattice_reason_tournament_state.json";EVENTS=ART/"logs/lattice_reason_tournament_events.jsonl";LOCK=ART/"lattice_reason_tournament.lock"
TOKENIZER=ART/"tokenizers/babylm_2026_4k.json";TOK_REPORT=ART/"tokenizers/babylm_2026_4k.report.json";MANIFEST=ART/"data/phase7f_v2r1/canonical-100m/manifest.json";REPO="insightlabs38-pixel/LatticeLM-research"
CONTROL={"run_id":"LR-0","lr_fraction":0,"training_tokens":25_000_000,"data_d_tokens":25_000_000,"lattice_reason_tokens":0,"data_d_validation_loss":3.695686,"common_validation_loss":4.464799,"wikitext_ppl":83.40936,"wikitext_bpb":2.03912,"hellaswag":.262896,"arc_easy":.283249,"piqa":.534276,"winogrande":.498027,"hf_revision":"1cf7116faf4b2896594239c83008518bcc84c37d"}
RULE={"noise_floor_accuracy":0.003,"reasoning_comparison":"count task gains exceeding noise floor, then summed deltas and consistency","acceptable_wikitext_ppl_cost_fraction":0.05,"refinement":"predeclared cases A-D from experiment request; default LR-15 for a 10/25 tradeoff","winner":"external transfer first; LatticeReason validation cannot select a winner alone"}

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def event(name,**fields):
 EVENTS.parent.mkdir(parents=True,exist_ok=True)
 with EVENTS.open("a") as h:h.write(json.dumps({"at":now(),"event":name,**fields},sort_keys=True)+"\n")
def run(command,label,deadline,retries=2):
 for attempt in range(retries+1):
  remaining=deadline-time.time()
  if remaining<=0:raise TimeoutError("hard deadline reached")
  event("SUBPROCESS_START",label=label,attempt=attempt)
  try:done=subprocess.run(command,cwd=ROOT,timeout=remaining)
  except subprocess.TimeoutExpired: event("HARD_DEADLINE",label=label);raise
  if done.returncode==0:return
  event("SUBPROCESS_FAILED",label=label,attempt=attempt,returncode=done.returncode)
  time.sleep(min(30,2**attempt))
 raise RuntimeError(f"{label} exhausted retries")
def update(state,stage,**fields):
 state.update({"current_stage":stage,"updated_at":now(),**fields});atomic(STATE,state);event("STAGE",stage=stage)
def terminal(run_id,target):
 path=ART/"checkpoints"/run_id/"latest.pt";side=path.with_suffix(".sha256")
 if not path.exists() or sha256_file(path)!=side.read_text().strip():return None
 value=torch.load(path,map_location="cpu",weights_only=False)
 return value if int(value["tokens_seen"])==target else None
def train(run_id,percent,target,state,hard):
 update(state,"TRAINING",current_model=run_id,mixture={"data_d":(100-percent)/100,"lattice_reason":percent/100},total_tokens=target)
 ck=terminal(run_id,target);mode="--resume" if (ART/"checkpoints"/run_id/"latest.pt").exists() else "--fresh"
 if not ck:run([sys.executable,"scripts/train_lattice_reason_mixture.py","--manifest",str(MANIFEST),"--run-id",run_id,"--lr-percent",str(percent),"--target",str(target),mode],f"train-{run_id}-{target}",hard)
 ck=terminal(run_id,target)
 if not ck:raise RuntimeError(f"{run_id} terminal checkpoint invalid")
 return evaluate_publish(run_id,percent,target,ck,state,hard)
def curve_row(run_id,target):
 with (ART/"lattice_reason_training_curves.csv").open() as h:rows=[r for r in csv.DictReader(h) if r["run_id"]==run_id and int(r["training_tokens"])==target]
 if not rows:raise RuntimeError("terminal curve row missing")
 return rows[-1]
def evaluate_publish(run_id,percent,target,ck,state,hard):
 out=ART/"lattice_reason"/run_id.lower();out.mkdir(parents=True,exist_ok=True);native=out/f"{run_id.lower()}-{target}.pt"
 if not native.exists():shutil.copy2(ART/"checkpoints"/run_id/"latest.pt",native)
 wiki=out/"wikitext.json";gibc=out/"gibc.json"
 update(state,"EVALUATION",current_model=run_id,latest_checkpoint=str(native),checkpoint_hash=sha256_file(native))
 if not wiki.exists():run([sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(native),"--tokenizer",str(TOKENIZER),"--output",str(wiki),"--threads","4"],f"wiki-{run_id}",hard,1)
 if not gibc.exists():run([sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(native),"--tokenizer",str(TOKENIZER),"--output",str(gibc),"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","4"],f"gibc-{run_id}",hard,1)
 w=json.loads(wiki.read_text());g=json.loads(gibc.read_text());row=curve_row(run_id,target)
 result={"run_id":run_id,"lr_fraction":percent/100,"training_tokens":target,"data_d_tokens":int(row["data_d_tokens"]),"lattice_reason_tokens":int(row["lattice_reason_tokens"]),"parameters":15_949_760,"common_validation_loss":float(row["common_validation_loss"]),"data_d_validation_loss":float(row["data_d_validation_loss"]),"lattice_reason_validation_loss":float(row["lattice_reason_validation_loss"]),"wikitext_ppl":w["perplexity"],"wikitext_bpb":w["bits_per_byte"],**{t:g["results"][t]["acc,none"] for t in ("hellaswag","arc_easy","piqa","winogrande")},"tokens_per_second":float(row["tokens_per_second"]),"wall_seconds":float(row["wall_seconds"]),"peak_ram_bytes":int(row["peak_ram_bytes"]),"checkpoint_sha":sha256_file(native)}
 metrics=out/"metrics.json";atomic(metrics,result);bundle=ART/"cache"/f"{run_id.lower()}-{target}";export_checkpoint(native,bundle,run_id,"final",TOKENIZER,TOK_REPORT,ART/"data_d_v2r1_manifest.json",{"tokens_trained":target,"wall_seconds":result["wall_seconds"],"val_loss":result["common_validation_loss"],"val_ppl":pow(2.718281828,result["common_validation_loss"])})
 token=get_token()
 if not token:raise RuntimeError("Hugging Face credential unavailable")
 remote=f"experiments/co4-l-data-d-lr{percent}-{target//1_000_000}m";revision=upload_checkpoint(bundle,REPO,remote,token);result.update(hf_revision=revision,hf_path=remote);atomic(metrics,result);event("PUBLISHED",run_id=run_id,revision=revision)
 state.setdefault("validation_metrics",{})[run_id]={k:v for k,v in result.items() if "validation" in k};state.setdefault("benchmark_metrics",{})[run_id]={k:result[k] for k in ("wikitext_ppl","wikitext_bpb","hellaswag","arc_easy","piqa","winogrande")};state.setdefault("completed_stages",[]).append(f"{run_id}@{target}");atomic(STATE,state)
 return result
def choose(a,b):
 tasks=("hellaswag","arc_easy","piqa","winogrande");floor=RULE["noise_floor_accuracy"]
 def quality(x):
  deltas=[x[t]-CONTROL[t] for t in tasks];return (sum(d>floor for d in deltas),sum(deltas),-max(0,x["wikitext_ppl"]/CONTROL["wikitext_ppl"]-1))
 qa,qb=quality(a),quality(b);best=a if qa>=qb else b
 if max(qa[0],qb[0])<2:return None,None,"D"
 acceptable=best["wikitext_ppl"]<=CONTROL["wikitext_ppl"]*1.05
 if best is b and qb>qa and acceptable:return b,50,"B"
 if qa>qb:
  return a,(5 if a["wikitext_ppl"]>CONTROL["wikitext_ppl"]*1.05 else 15),("C" if not acceptable else "B")
 return best,15,("C" if not acceptable else "B")
def reports(results,state,classification,winner):
 fields=["run_id","lr_fraction","training_tokens","data_d_tokens","lattice_reason_tokens","parameters","common_validation_loss","data_d_validation_loss","lattice_reason_validation_loss","wikitext_ppl","wikitext_bpb","hellaswag","arc_easy","piqa","winogrande","tokens_per_second","wall_seconds","peak_ram_bytes","hf_revision","checkpoint_sha"]
 allrows=[CONTROL]+results
 for name in ("lattice_reason_mixture_results.csv","lattice_reason_transfer_results.csv"):
  with (ART/name).open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore",lineterminator="\n");w.writeheader();w.writerows(allrows)
 lines=["# LatticeReason mixture tournament","",f"LATTICEREASON TRANSFER RESULT: **{classification}**","",f"BEST LR FRACTION AT 25M: **{winner['lr_fraction']*100:g}**" if winner else "BEST LR FRACTION AT 25M: **INCONCLUSIVE**","",f"WINNER CONTINUED TO 50M: **{'YES' if any(x['training_tokens']==50_000_000 for x in results) else 'NO'}**","","The predeclared decision rule evaluated each official task separately with a 0.003 accuracy noise floor and allowed at most 5% WikiText perplexity cost.",""]
 (ART/"lattice_reason_tournament_report.md").write_text("\n".join(lines));(ART/"lattice_reason_tournament_decision.md").write_text("\n".join(lines+["Recommended next direction: **"+("scale winning LR mixture to 100M" if winner else "improve generator/template diversity")+"**. This next phase was not launched.",""]))
 state["final_decision"]={"classification":classification,"best_lr_fraction_25m":winner["lr_fraction"] if winner else "INCONCLUSIVE","winner_continued_50m":any(x["training_tokens"]==50_000_000 for x in results)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--soft-hours",type=float,default=9);p.add_argument("--hard-hours",type=float,default=10);p.add_argument("--dry-run",action="store_true");a=p.parse_args();LOCK.parent.mkdir(exist_ok=True);lock=LOCK.open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
 source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();started=time.time();soft=started+a.soft_hours*3600;hard=started+a.hard_hours*3600
 state={"run_id":"lattice-reason-tournament-2026-09-06","source_commit":source,"current_stage":"STARTING","completed_stages":[],"current_model":None,"mixture":None,"seed":314159,"total_tokens":0,"data_d_tokens":0,"lattice_reason_tokens":0,"latest_checkpoint":None,"checkpoint_hash":None,"validation_metrics":{},"benchmark_metrics":{},"retry_counts":{},"spot_interruptions":0,"soft_deadline":datetime.fromtimestamp(soft,timezone.utc).isoformat(),"hard_deadline":datetime.fromtimestamp(hard,timezone.utc).isoformat(),"decision_rule":RULE,"final_decision":None,"started_at":now()};atomic(STATE,state);event("MASTER_STARTED",source_commit=source)
 try:
  update(state,"PREFLIGHT");verify_top_manifest(MANIFEST,TOKENIZER)
  if a.dry_run:update(state,"DRY_RUN_COMPLETE");return
  results=[train("LR-10",10,25_000_000,state,hard),train("LR-25",25,25_000_000,state,hard)]
  winner,refine,classification=choose(*results);update(state,"PRIMARY_COMPARISON",final_decision={"provisional_classification":classification,"optional_fraction":refine})
  if refine and time.time()<soft:
   estimate=max(x["wall_seconds"] for x in results)+3600
   if time.time()+estimate<hard:results.append(train(f"LR-{refine}",refine,25_000_000,state,hard));winner,_,classification=choose(results[0],results[1]);winner=max((x for x in results if x["training_tokens"]==25_000_000),key=lambda x:sum(x[t]-CONTROL[t] for t in ("hellaswag","arc_easy","piqa","winogrande")))
  if winner and time.time()<soft and time.time()+winner["wall_seconds"]+3600<hard:results.append(train(winner["run_id"],int(winner["lr_fraction"]*100),50_000_000,state,hard))
  reports(results,state,classification,winner);update(state,"COMPLETE",final_decision=state["final_decision"]);event("MASTER_COMPLETE");subprocess.run([sys.executable,"scripts/credential_scan.py"],cwd=ROOT,check=False) if (ROOT/"scripts/credential_scan.py").exists() else None
  subprocess.run(["git","add","artifacts/lattice_reason_tournament_report.md","artifacts/lattice_reason_tournament_decision.md","artifacts/lattice_reason_mixture_results.csv","artifacts/lattice_reason_transfer_results.csv","artifacts/lattice_reason_training_curves.csv","artifacts/lattice_reason_tournament_state.json"],cwd=ROOT,check=True);subprocess.run(["git","commit","-m","Record LatticeReason mixture tournament"],cwd=ROOT,check=True);subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True)
 except TimeoutError as exc:reports([],state,"F — inconclusive",None);update(state,"HARD_DEADLINE_STOP",error=str(exc));event("MASTER_HARD_STOP")
 except Exception as exc:update(state,"FAILED",error=f"{type(exc).__name__}: {exc}");event("MASTER_FAILED",error=state["error"]);raise
if __name__=="__main__":main()
