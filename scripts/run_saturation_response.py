"""Restart-safe controller for bounded WSD/systems/post-training branches.

Workers are ordinary deterministic programs; this controller never invokes an AI
agent or external judge.  Every child writes SHA-bound evidence before its stage
is marked complete.
"""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
try:from post900m_tournament import promote_cooldown,promote_system,promote_sft,promote_rl
except ModuleNotFoundError:from scripts.post900m_tournament import promote_cooldown,promote_system,promote_sft,promote_rl

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";BASE=ART/"checkpoints/co4-final-capacity32-data-d-v3/milestone-900000000.pt"
STATE=ART/"saturation_response_state.json";PREVIOUS=ART/"saturation_response_state.previous.json";LOCK=ART/"saturation_response.lock";EVENTS=ART/"logs/saturation_response_events.jsonl"
RESULT=ART/"saturation_response_results.json";STOP=False;CHILD=None
PHASES=("WSD_COOLDOWN","SYSTEMS_BENCHMARK","SFT","RLVR_GRPO","FINAL_SELECTION")
WSD_BASES=(("BASE_512M",ART/"checkpoints/co4-final-capacity32-data-d-v3/milestone-512000000.pt"),("BASE_750M",ART/"checkpoints/co4-final-capacity32-data-d-v3/milestone-750000000.pt"),("BASE_900M",BASE))
def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(obj,indent=2,sort_keys=True)+"\n").encode();tmp=path.with_name("."+path.name+".tmp")
    with tmp.open("wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    if path==STATE and path.exists():PREVIOUS.write_bytes(path.read_bytes())
    os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def event(kind,**kw):
    EVENTS.parent.mkdir(parents=True,exist_ok=True);fd=os.open(EVENTS,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o664)
    try:os.write(fd,(json.dumps({"at":time.time(),"event":kind,**kw},sort_keys=True)+"\n").encode());os.fsync(fd)
    finally:os.close(fd)
def load():
    for p in (STATE,PREVIOUS):
        try:
            x=json.loads(p.read_text());assert x.get("schema")=="saturation-response-v1";return x
        except Exception:pass
    return None
def save(s,stage=None,**kw):
    if stage:s["stage"]=stage
    s.update(kw);s["updated_at"]=time.time();atomic(STATE,s);event("STATE",stage=s["stage"])
def stop(*_):
    global STOP;STOP=True
    if CHILD and CHILD.poll() is None:CHILD.terminate()
def run(s,label,cmd):
    global CHILD
    if STOP:raise InterruptedError("stop requested")
    event("CHILD_START",label=label,command=[Path(cmd[0]).name,*cmd[1:]]);CHILD=subprocess.Popen(cmd,cwd=ROOT);s["child_pid"]=CHILD.pid;save(s)
    code=CHILD.wait();CHILD=None;s["child_pid"]=None;save(s);event("CHILD_END",label=label,exit_code=code)
    if code==75 and STOP:raise InterruptedError(label)
    if code:raise RuntimeError(f"{label} exited {code}")
def read(path):return json.loads(path.read_text())
def execute(s,dry=False,wsd_only=False):
    if dry:save(s,"DRY_RUN_COMPLETE",planned_phases=list(PHASES));return
    base=read(ART/"post900m_baseline_metrics.json")
    if "WSD_COOLDOWN" not in s["completed"]:
        save(s,"WSD_COOLDOWN");per_base={};promoted=[]
        for base_id,checkpoint in WSD_BASES:
            out=ART/f"saturation/wsd/{base_id.lower()}/results.json"
            if not out.exists():run(s,f"wsd-{base_id.lower()}",[sys.executable,"scripts/train_post900m_experiment.py","wsd","--base",str(checkpoint),"--base-id",base_id,"--output",str(out)])
            raw=read(out);winner,evidence=promote_cooldown(raw["base"],raw["candidates"]);per_base[base_id]={"checkpoint":str(checkpoint),"base_metrics":raw["base"],"winner":winner,"evidence":evidence};promoted.extend([winner] if winner else [])
        winner=min(promoted,key=lambda x:(x["wikitext_bpb"],x["validation_loss"],-sum(x[t] for t in ("hellaswag","arc_easy","piqa","winogrande")))) if promoted else None
        s["wsd"]={"bases":[x[0] for x in WSD_BASES],"per_base":per_base,"winner":winner,"selection_policy":"promote versus own parent, then lowest BPB/validation loss with official-score tie-break"};s["completed"].append("WSD_COOLDOWN");save(s)
    if wsd_only:
        atomic(RESULT,{"schema":"wsd-tournament-result-v1","status":"COMPLETE","wsd":s["wsd"]});save(s,"COMPLETE",result=str(RESULT));return
    if "SYSTEMS_BENCHMARK" not in s["completed"]:
        save(s,"SYSTEMS_BENCHMARK");out=ART/"saturation/systems/results.json"
        if not out.exists():run(s,"systems",[sys.executable,"scripts/benchmark_post900m_systems.py","--checkpoint",str(BASE),"--output",str(out)])
        raw=read(out);winner,evidence=promote_system(raw["baseline"],raw["candidates"]);s["systems"]={"winner":winner,"evidence":evidence,"raw":str(out)};s["completed"].append("SYSTEMS_BENCHMARK");save(s)
    pretrained=(s["wsd"]["winner"] or {"checkpoint":str(BASE)})["checkpoint"]
    if "SFT" not in s["completed"]:
        save(s,"SFT");out=ART/"saturation/sft/results.json"
        if not out.exists():run(s,"sft",[sys.executable,"scripts/train_post900m_experiment.py","sft","--base",pretrained,"--output",str(out)])
        winner,evidence=promote_sft(base,read(out)["candidates"]);s["sft"]={"winner":winner,"evidence":evidence};s["completed"].append("SFT");save(s)
    if s["sft"]["winner"] and "RLVR_GRPO" not in s["completed"]:
        save(s,"RLVR_GRPO");out=ART/"saturation/rlvr/results.json"
        if not out.exists():run(s,"rlvr",[sys.executable,"scripts/train_post900m_experiment.py","rlvr","--base",s["sft"]["winner"]["checkpoint"],"--output",str(out)])
        winner,evidence=promote_rl(s["sft"]["winner"],read(out)["candidates"]);s["rlvr"]={"winner":winner,"evidence":evidence};s["completed"].append("RLVR_GRPO");save(s)
    elif "RLVR_GRPO" not in s["completed"]:s["rlvr"]={"winner":None,"skipped":"no promoted SFT candidate"};s["completed"].append("RLVR_GRPO");save(s)
    finalists=[x for x in (s["wsd"].get("winner"),s["sft"].get("winner"),s["rlvr"].get("winner")) if x]
    best=max(finalists,key=lambda x:(sum(x[t] for t in ("hellaswag","arc_easy","piqa","winogrande"))/4,-x["wikitext_bpb"])) if finalists else None
    result={"schema":"saturation-response-result-v1","status":"COMPLETE","base_checkpoint":str(BASE),"best_candidate":best,"phases":{k:s.get(k) for k in ("wsd","systems","sft","rlvr")},"dpo":{"status":"SKIPPED","reason":"lower priority; no deterministic evidence requiring it"}}
    atomic(RESULT,result);save(s,"COMPLETE",result=str(RESULT))
def main():
    p=argparse.ArgumentParser();p.add_argument("--dry-run",action="store_true");p.add_argument("--wsd-only",action="store_true");p.add_argument("--state",type=Path);a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    global STATE,PREVIOUS,LOCK,EVENTS
    if a.state:STATE=a.state;PREVIOUS=a.state.with_suffix(".previous.json");LOCK=a.state.with_suffix(".lock");EVENTS=a.state.with_suffix(".events.jsonl")
    with LOCK.open("a+") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);s=load() or {"schema":"saturation-response-v1","stage":"START","completed":[],"child_pid":None,"base_checkpoint":str(BASE)}
        if s["stage"]=="COMPLETE":return 0
        try:execute(s,a.dry_run,a.wsd_only)
        except InterruptedError:save(s,"STOPPED_SAFE");return 75
        except Exception as e:save(s,"BLOCKED",blocker=f"{type(e).__name__}: {e}");return 2
    return 0
if __name__=="__main__":raise SystemExit(main())
