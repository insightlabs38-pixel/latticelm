"""Restart-safe autonomous successor to the 900M final-scale master."""
from __future__ import annotations
import argparse,csv,fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
from latticelm.config import LatticeConfig
from latticelm.data_d import sha256_file,verify_top_manifest
from latticelm.model import build_model
try:from post900m_scaling import analyze,classify
except ModuleNotFoundError:from scripts.post900m_scaling import analyze,classify

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts"
STATE=ART/"post900m_master_state.json";BACKUP=ART/"post900m_master_state.previous.json";LOCK=ART/"post900m_master.lock";EVENTS=ART/"logs/post900m_master_events.jsonl"
DECISION=ART/"post900m_decision.json";REPORT=ART/"post900m_report.md";SCALING=ART/"post900m_scaling_analysis.json";CERT=ART/"post900m_900m_certification.json";HANDOFF=ART/"post900m_handoff.json"
BASELINE=ART/"post900m_baseline_metrics.json";SATURATION_RESULT=ART/"saturation_response_results.json"
PRE_STATE=ART/"final_scale_master_state.json";CURVE=ART/"final_capacity_training_curve.csv";RUN="co4-final-capacity32-data-d-v3";CKPTS=ART/"checkpoints"/RUN
CONFIG=ROOT/"configs/final_capacity32_co4.json";TOK=ART/"tokenizers/babylm_2026_4k.json";DATA3=ART/"data/data_d_v3/canonical-1b";MANIFEST3=DATA3/"manifest.json"
PARAMS=32_678_640;TARGET=900_000_000;PRE_SERVICE="latticelm-final-scale-20260909.service";WAIT_SECONDS=300;STOP=False;CHILD=None
TERMINAL={"COMPLETE","BLOCKED","FAILED"}

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,obj,backup=None,immutable=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(obj,indent=2,sort_keys=True)+"\n").encode()
    if immutable and path.exists():
        if path.read_bytes()!=encoded:raise RuntimeError(f"immutable evidence disagreement: {path}")
        return
    tmp=path.with_name("."+path.name+".tmp")
    with tmp.open("wb") as f:f.write(encoded);f.flush();os.fsync(f.fileno())
    if backup and path.exists():shutil.copy2(path,backup)
    os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def event(kind,**kw):
    EVENTS.parent.mkdir(parents=True,exist_ok=True);fd=os.open(EVENTS,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o664)
    try:os.write(fd,(json.dumps({"at":now(),"event":kind,**kw},sort_keys=True)+"\n").encode());os.fsync(fd)
    finally:os.close(fd)
def load():
    errors=[]
    for p in (STATE,BACKUP):
        try:
            x=json.loads(p.read_text());
            if not isinstance(x,dict) or "current_stage" not in x:raise ValueError("invalid state schema")
            return x
        except (OSError,json.JSONDecodeError,ValueError) as e:errors.append(str(e))
    return None
def initial():return {"schema":"post900m-master-v1","master":"post900m","run_id":f"post900m-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}","current_stage":"WAIT_PREDECESSOR","created_at":now(),"updated_at":now(),"source_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"child_pid":None,"stop_requested":False,"completed_stages":[],"predecessor_service":PRE_SERVICE}
def save(s,stage=None,**kw):
    if stage:s["current_stage"]=stage
    s.update(kw);s["updated_at"]=now();atomic(STATE,s,BACKUP);event("STATE",stage=s["current_stage"])
def request_stop(*_):
    global STOP;STOP=True;s=load()
    # systemd may invoke ExecStop during normal teardown. Terminal evidence must
    # remain byte-stable across an idempotent start/stop cycle.
    if s and s.get("current_stage") not in TERMINAL:save(s,stop_requested=True)
    if CHILD and CHILD.poll() is None:CHILD.terminate()
class Lock:
    def __enter__(self):
        self.f=LOCK.open("a+");fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB);self.f.seek(0);self.f.truncate();self.f.write(str(os.getpid()));self.f.flush();return self
    def __exit__(self,*_):fcntl.flock(self.f,fcntl.LOCK_UN);self.f.close()
def run(s,label,cmd):
    global CHILD
    if STOP:raise InterruptedError("safe stop requested")
    event("CHILD_START",label=label,command=[Path(cmd[0]).name,*cmd[1:]]);CHILD=subprocess.Popen(cmd,cwd=ROOT);s["child_pid"]=CHILD.pid;save(s)
    code=CHILD.wait();CHILD=None;s["child_pid"]=None;save(s);event("CHILD_END",label=label,exit_code=code)
    if code==75 and STOP:raise InterruptedError(f"{label} stopped safely")
    if code:raise RuntimeError(f"{label} exited {code}")
def block(s,reason,kind="BLOCKED"):
    atomic(DECISION,{"classification":"BLOCKED","reason":reason,"at":now()});write_handoff(s,kind,"INTEGRITY_OR_SYSTEM_BLOCKER",reason);save(s,kind,blocker=reason);event(kind,reason=reason)
def service_active(name):
    return subprocess.run(["systemctl","is-active","--quiet",name],check=False).returncode==0
def predecessor_ready(s,once=False):
    while True:
        try:p=json.loads(PRE_STATE.read_text())
        except Exception as exc:return False,f"predecessor state unreadable: {exc}" if not service_active(PRE_SERVICE) else None
        stage=p.get("current_stage");child=p.get("child_pid")
        if stage=="COMPLETE" and p.get("final_tokens")==TARGET and child is None:return True,None
        if stage in {"BLOCKED","FAILED","STOPPED_SAFE"}:return False,f"predecessor terminal state {stage}: {p.get('last_error') or p.get('blocker','unspecified')}"
        if not service_active(PRE_SERVICE):return False,f"predecessor inactive in unexpected application state {stage}"
        if once:return False,None
        event("WAITING",predecessor_stage=stage,child_pid=child);time.sleep(WAIT_SECONDS)
        if STOP:raise InterruptedError("safe stop while waiting")
def _tuple_ok(x):return isinstance(x,(tuple,list)) and len(x)>0
def certify_900m():
    cp=CKPTS/f"milestone-{TARGET}.pt";side=cp.with_suffix(".sha256")
    if not cp.exists() or not side.exists():raise RuntimeError("missing immutable 900M checkpoint or sidecar")
    digest=sha256_file(cp)
    if side.read_text().strip()!=digest:raise RuntimeError("900M checkpoint SHA256 sidecar mismatch")
    state=torch.load(cp,map_location="cpu",weights_only=False);cfg=LatticeConfig.from_json(CONFIG);model=build_model(cfg)
    checks={"parameters":model.parameter_breakdown()["total"]==PARAMS,"config":state.get("config")==cfg.to_dict(),"strict_state_dict":False,"backend":state.get("backend")=="fp32_compile","lineage":state.get("lineage")==RUN and state.get("run_id")==RUN,"tokens_seen":state.get("tokens_seen")==TARGET}
    model.load_state_dict(state["model"],strict=True);checks["strict_state_dict"]=True
    required=("optimizer","scheduler","python_rng_state","numpy_rng_state","torch_rng_state","data_source_selector_state","next_batch_sha256")
    checks["exact_resume_fields"]=all(k in state for k in required);checks["rng_states"]=_tuple_ok(state.get("python_rng_state")) and _tuple_ok(state.get("numpy_rng_state")) and isinstance(state.get("torch_rng_state"),torch.Tensor)
    mh=sha256_file(MANIFEST3);top=verify_top_manifest(MANIFEST3,TOK);dc=json.loads((DATA3/"certification.json").read_text())
    checks.update({"manifest_hash":state.get("data_manifest_sha256")==mh==dc.get("manifest_sha256"),"tokenizer_hash":top.get("tokenizer_sha256")==sha256_file(TOK),"dataset_certification":all(dc.get(k)=="PASS" for k in ("acceptance","train_validation_disjointness","all_shards_mmap_and_hash","exact_resume","stable_unique_ids"))})
    rows=list(csv.DictReader(CURVE.open()));row=next((r for r in reversed(rows) if int(r["nominal_tokens"])==TARGET),None);checks["final_training_row"]=row is not None and row.get("checkpoint_sha256")==digest
    checks["numerical_health"]=bool(row) and all(math.isfinite(float(row[k])) for k in ("train_loss","data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss"))
    checks["no_catastrophic_degradation"]=bool(row) and float(row["data_d_validation_loss"])<=min(float(r["data_d_validation_loss"]) for r in rows[-5:])+.05
    prior=json.loads(CERT.read_text()) if CERT.exists() else {}
    result={"schema":"post900m-certification-v1","status":"PASS" if all(checks.values()) else "FAIL","checkpoint":str(cp.relative_to(ROOT)),"checkpoint_sha256":digest,"tokens":TARGET,"parameters":PARAMS,"manifest_sha256":mh,"tokenizer_sha256":sha256_file(TOK),"checks":checks,"certified_at":prior.get("certified_at",now())}
    atomic(CERT,result,immutable=True)
    if result["status"]!="PASS":raise RuntimeError("900M certification failed: "+",".join(k for k,v in checks.items() if not v))
    return result
def evaluate(s,cert):
    cp=ROOT/cert["checkpoint"];wiki=ART/"post900m_900m_wikitext.json";gibc=ART/"post900m_900m_gibc.json"
    if wiki.exists() and json.loads(wiki.read_text()).get("checkpoint_sha256")!=cert["checkpoint_sha256"]:raise RuntimeError("existing WikiText evidence belongs to another checkpoint")
    if not wiki.exists():run(s,"900m-wikitext",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(cp),"--tokenizer",str(TOK),"--output",str(wiki),"--threads","4"])
    if gibc.exists() and json.loads(gibc.read_text()).get("phase6_metadata",{}).get("checkpoint_sha256")!=cert["checkpoint_sha256"]:raise RuntimeError("existing GIBC evidence belongs to another checkpoint")
    if not gibc.exists():run(s,"900m-gibc",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(cp),"--tokenizer",str(TOK),"--output",str(gibc),"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","4"])
    return json.loads(wiki.read_text()),json.loads(gibc.read_text())
def freeze_continuation_milestone(tokens):
    root=ART/"checkpoints/co4-final-capacity32-data-d-v4";src=next((root/n for n in ("latest.pt","previous.pt","fallback.pt") if (root/n).exists() and (root/n).with_suffix(".sha256").exists() and sha256_file(root/n)==(root/n).with_suffix(".sha256").read_text().strip()),None)
    if src is None:raise RuntimeError("no valid continuation checkpoint")
    state=torch.load(src,map_location="cpu",weights_only=False)
    if state.get("tokens_seen")!=tokens:raise RuntimeError("continuation milestone token mismatch")
    dst=root/f"milestone-{tokens}.pt";side=dst.with_suffix(".sha256")
    if not dst.exists():shutil.copy2(src,dst);side.write_text(sha256_file(dst)+"\n")
    if not side.exists() or sha256_file(dst)!=side.read_text().strip():raise RuntimeError("continuation immutable milestone disagreement")
    return dst
def evaluate_continuation(s,tokens,full=False):
    cp=freeze_continuation_milestone(tokens);wiki=ART/f"post900m_{tokens}_wikitext.json";gibc=ART/f"post900m_{tokens}_gibc.json"
    if not wiki.exists():run(s,f"wikitext-{tokens}",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(cp),"--tokenizer",str(TOK),"--output",str(wiki),"--threads","4"])
    if full and not gibc.exists():run(s,f"gibc-{tokens}",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(cp),"--tokenizer",str(TOK),"--output",str(gibc),"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","4"])
    return json.loads(wiki.read_text()),json.loads(gibc.read_text()) if full else None
def wiki_history(final):
    out={900_000_000:final}
    for p in ART.glob("final_scale_*_wikitext.json"):
        try:out[int(p.stem.split("_")[2])]=json.loads(p.read_text())
        except Exception:pass
    p=ART/"final_capacity_205m_metrics.json"
    if p.exists():out[205_000_000]=json.loads(p.read_text())["wikitext"]
    return out
def transition_record(old,new,cp_sha):
    top3=json.loads(MANIFEST3.read_text());top4=json.loads(new.read_text());cert4=json.loads((new.parent/"certification.json").read_text())
    if cert4.get("acceptance")!="PASS" or cert4.get("manifest_sha256")!=sha256_file(new) or top4["total_unique_tokens"]<1_500_000_000:raise RuntimeError("DATA-D-v4 certification insufficient")
    record={"schema":"dataset-transition-v1","old_manifest_sha256":sha256_file(MANIFEST3),"new_manifest_sha256":sha256_file(new),"old_corpus_token_count":top3["total_unique_tokens"],"new_corpus_token_count":top4["total_unique_tokens"],"checkpoint_sha256":cp_sha,"token_count_at_transition":TARGET,"source_mixture_policy":top4["mixture_definition"],"data_stream_state_policy":"new deterministic stream initialized at 900M; model/optimizer/scheduler/RNG lineage retained; no claim of bit-exact uninterrupted data-stream resume","determinism_proof":cert4.get("deterministic_first_batch_sha256")}
    atomic(ART/"post900m_dataset_transition.json",record,immutable=True);return record
def evidence_name(path):
    try:return str(path.relative_to(ROOT))
    except ValueError:return str(path)
def write_handoff(s,status,terminal,reason=None,metrics=None,classification=None):
    cert=json.loads(CERT.read_text()) if CERT.exists() else {};classification=classification or s.get("classification");required={"1_5B_COMPLETE":"evaluate_1_5B_scaling","SATURATION_RESPONSE_COMPLETE":"strategic_review","INCONCLUSIVE":"resolve_900m_scaling_ambiguity","BLOCKED":"human_review","INTEGRITY_OR_SYSTEM_BLOCKER":"human_review"}.get(terminal,"human_review")
    obj={"master":"post900m","status":status,"terminal_stage":terminal,"decision_required":required,"base_checkpoint":{"tokens":s.get("final_tokens",TARGET),"sha256":s.get("final_checkpoint_sha256",cert.get("checkpoint_sha256"))},"predecessor_900m":{"sha256":cert.get("checkpoint_sha256")},"data":s.get("data",{"version":"DATA-D-v3","manifest_sha256":cert.get("manifest_sha256"),"certification":cert.get("status")}),"metrics":metrics or s.get("metrics",{}),"scaling":{"classification":classification,"projected_2b_gain":s.get("projected_2b_gain")},"reason":reason,"next_allowed_actions":["EVALUATE_2B_CONTINUATION","RUN_WSD_TOURNAMENT","RUN_POSTTRAINING_TOURNAMENT","RUN_SYSTEMS_OPTIMIZATION"],"evidence_files":[evidence_name(p) for p in (CERT,SCALING,DECISION,REPORT,SATURATION_RESULT) if p.exists()],"generated_at":now()};atomic(HANDOFF,obj)
def report(s,classification,cert,wiki,gibc):
    g=gibc["results"];metrics={"validation_loss":float(next(r for r in reversed(list(csv.DictReader(CURVE.open()))) if int(r["nominal_tokens"])==TARGET)["data_d_validation_loss"]),"wikitext_ppl":wiki["perplexity"],"wikitext_bpb":wiki["bits_per_byte"],**{k:g[k]["acc,none"] for k in ("hellaswag","arc_easy","piqa","winogrande")}}
    text=f"# Post-900M report\n\nClassification: **{classification}**\n\nThe immutable 900M checkpoint `{cert['checkpoint_sha256']}` passed model, checkpoint, dataset, exact-resume-field, and numerical-health certification. The complete 32M trajectory was analyzed with all-history and late-window asymptotic power-law fits.\n\nWikiText PPL: {metrics['wikitext_ppl']:.6f}\nWikiText BPB: {metrics['wikitext_bpb']:.6f}\nHellaSwag: {metrics['hellaswag']:.6f}\nARC-Easy: {metrics['arc_easy']:.6f}\nPIQA: {metrics['piqa']:.6f}\nWinoGrande: {metrics['winogrande']:.6f}\n\nPost-training is eligible only after the best pretrained base checkpoint is selected. No external pretrained teacher, judge, reward model, or evaluation data is permitted. 40M and 49M were screened out before training; they were not failed trained models.\n"
    REPORT.write_text(text);atomic(BASELINE,{"schema":"post900m-baseline-metrics-v1","candidate_id":"BASE_900M","checkpoint":cert["checkpoint"],"checkpoint_sha256":cert["checkpoint_sha256"],"reasoning_accuracy":0.0,**metrics});return metrics
def execute(s,once=False,dry_run=False):
    save(s,"WAIT_PREDECESSOR");ready,reason=predecessor_ready(s,once=once)
    if not ready:
        if reason:block(s,reason)
        return
    save(s,"CERTIFY_900M");cert=certify_900m();s["predecessor_900m"]=cert;save(s,"EVALUATE_900M")
    if dry_run:save(s,"DRY_RUN_COMPLETE",certification="PASS");return
    wiki,gibc=evaluate(s,cert);save(s,"ANALYZE_SCALING");a=analyze(CURVE,wiki_history(wiki));classification=classify(a);a["classification"]=classification;atomic(SCALING,a,immutable=True);prior_decision=json.loads(DECISION.read_text()) if DECISION.exists() else {};atomic(DECISION,{"classification":classification,"policy_version":"post900m-v1","checkpoint_sha256":cert["checkpoint_sha256"],"scaling_sha256":sha256_file(SCALING),"decided_at":prior_decision.get("decided_at",now())},immutable=True);s["classification"]=classification;s["projected_2b_gain"]=a["projected_900m_to_2_05b_gain"]
    metrics=report(s,classification,cert,wiki,gibc);s["metrics"]=metrics;save(s)
    if classification=="CONTINUE_TO_1_5B":
        save(s,"BUILD_DATA_D_V4");new=ART/"data/data_d_v4/canonical-2250m/manifest.json"
        if not new.exists():run(s,"build-data-d-v4",[sys.executable,"scripts/build_data_d_v4.py","--output",str(new.parent),"--total-tokens","2250000000"])
        if not (new.parent/"certification.json").exists():run(s,"certify-data-d-v4",[sys.executable,"scripts/certify_data_d_v3.py",str(new),"--output",str(new.parent/"certification.json"),"--minimum","1500000000"])
        transition=transition_record(MANIFEST3,new,cert["checkpoint_sha256"]);transition_path=ART/"post900m_dataset_transition.json";cont_curve=ART/"post900m_continuation_curve.csv";results=s.get("continuation_results",{})
        for target in (1_000_000_000,1_100_000_000,1_200_000_000,1_250_000_000,1_350_000_000,1_500_000_000):
            root=ART/"checkpoints/co4-final-capacity32-data-d-v4";existing=next((root/n for n in ("latest.pt","previous.pt","fallback.pt") if (root/n).exists() and (root/n).with_suffix(".sha256").exists() and sha256_file(root/n)==(root/n).with_suffix(".sha256").read_text().strip()),None);current=TARGET if existing is None else int(torch.load(existing,map_location="cpu",weights_only=False)["tokens_seen"])
            if current<target:
                save(s,f"TRAIN_{target}",current_tokens=current);run(s,f"train-{target}",[sys.executable,"scripts/train_post900m_continuation.py","--manifest",str(new),"--base-checkpoint",str(ROOT/cert["checkpoint"]),"--transition",str(transition_path),"--target",str(target),"--threads","16","--microbatch","8","--curve",str(cont_curve)])
            freeze_continuation_milestone(target)
            if target in (1_000_000_000,1_250_000_000,1_500_000_000):
                save(s,f"EVALUATE_{target}");w,g=evaluate_continuation(s,target,target==1_500_000_000);results[str(target)]={"wikitext":w,"gibc":g,"checkpoint_sha256":sha256_file(freeze_continuation_milestone(target))};s["continuation_results"]=results;save(s)
        final=results["1500000000"];g=final["gibc"]["results"];s["final_tokens"]=1_500_000_000;s["final_checkpoint_sha256"]=final["checkpoint_sha256"];s["data"]={"version":"DATA-D-v4","manifest_sha256":sha256_file(new),"certification":"PASS","unique_tokens":json.loads(new.read_text())["total_unique_tokens"]};s["metrics"]={"wikitext_ppl":final["wikitext"]["perplexity"],"wikitext_bpb":final["wikitext"]["bits_per_byte"],**{k:g[k]["acc,none"] for k in ("hellaswag","arc_easy","piqa","winogrande")}}
        write_handoff(s,"COMPLETE","1_5B_COMPLETE",metrics=s["metrics"],classification=classification);save(s,"COMPLETE");return
    if classification=="SATURATING_OR_LOW_VALUE":
        # Separate immutable lineages; never mutate BASE_900M or reinterpret the locked decision.
        atomic(ART/"post900m_saturation_response_plan.json",{"status":"READY","base_checkpoint_sha256":cert["checkpoint_sha256"],"ordered_phases":["WSD_COOLDOWN","SYSTEMS_BENCHMARK","SFT","RLVR_GRPO","FINAL_SELECTION"],"constraints":["immutable branches","deterministic supervision only","no external model judge","official evaluation required for promotion"]})
        save(s,"SATURATION_RESPONSE");run(s,"saturation-response",[sys.executable,"scripts/run_saturation_response.py"]);result=json.loads(SATURATION_RESULT.read_text());s["saturation_response"]=result
        write_handoff(s,"COMPLETE","SATURATION_RESPONSE_COMPLETE",metrics=metrics,classification=classification);save(s,"COMPLETE");return
    write_handoff(s,"COMPLETE",classification,metrics=metrics,classification=classification);save(s,"COMPLETE")
def main():
    p=argparse.ArgumentParser();p.add_argument("--status",action="store_true");p.add_argument("--stop",action="store_true");p.add_argument("--once",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--state",type=Path);a=p.parse_args()
    global STATE,BACKUP,LOCK,EVENTS
    if a.state:STATE=a.state;BACKUP=a.state.with_suffix(".previous.json");LOCK=a.state.with_suffix(".lock");EVENTS=a.state.with_suffix(".events.jsonl")
    if a.status:print(json.dumps(load(),indent=2));return 0
    if a.stop:request_stop();return 0
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    with Lock():
        s=load() or initial()
        if s["current_stage"] in TERMINAL:return 0 if s["current_stage"]=="COMPLETE" else 2
        try:execute(s,a.once,a.dry_run)
        except InterruptedError as exc:save(s,"STOPPED_SAFE",last_error=str(exc));return 75
        except Exception as exc:block(s,f"{type(exc).__name__}: {exc}");event("TRACEBACK",traceback=traceback.format_exc());return 2
    return 0
if __name__=="__main__":raise SystemExit(main())
