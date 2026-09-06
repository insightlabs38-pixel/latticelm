"""Unattended post-trainer evaluation, publication, and reasoning materialization.

This state machine never starts bulk generation unless the controlled native
checkpoint is verified at exactly 50M loss-bearing tokens.
"""
from __future__ import annotations
import csv,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import torch
import numpy as np
from huggingface_hub import get_token

from latticelm.hf_storage import export_checkpoint,sha256,upload_checkpoint
from latticelm.lattice_reason.dataset import (build_canonical,build_validation,
    compile_contamination_registry,sha256_file)
from latticelm.tokenizer import load_tokenizer

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";REC=ART/"checkpoints/co4-l-data-d-v2r1-25m/latest.pt"
SIDE=REC.with_suffix(".sha256");TOKENIZER=ART/"tokenizers/babylm_2026_4k.json";TOKENIZER_REPORT=ART/"tokenizers/babylm_2026_4k.report.json"
CURVE=ART/"data_d_v2r1_training_curve.csv";STATE=ART/"lattice_reason_overnight_state.json";EVENTS=ART/"logs/lattice_reason_overnight_events.jsonl"
POOL=ART/"data/lattice_reason/canonical";VALID=ART/"data/lattice_reason/validation";PUBLIC=ART/"lattice_reason"

def atomic(path:Path,value:dict):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n");os.replace(tmp,path)
def event(name:str,**fields):
    row={"at":datetime.now(timezone.utc).isoformat(),"event":name,**fields};EVENTS.parent.mkdir(parents=True,exist_ok=True)
    with EVENTS.open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
def run(label:str,command:list[str],retries:int=1):
    for attempt in range(retries+1):
        event("SUBPROCESS_START",label=label,attempt=attempt,command=command[1:3]);done=subprocess.run(command,cwd=ROOT)
        if done.returncode==0:return
        event("SUBPROCESS_FAILURE",label=label,attempt=attempt,code=done.returncode)
    raise RuntimeError(f"{label} failed after {retries+1} attempts")
def trainer_present()->bool:
    done=subprocess.run(["pgrep","-f","[r]un_phase7f_v2r1_training.py"],capture_output=True,text=True)
    return done.returncode==0
def latest_row()->dict[str,str]:
    with CURVE.open() as f:return list(csv.DictReader(f))[-1]
def verify_terminal()->tuple[dict,dict]:
    if not REC.exists() or sha256_file(REC)!=SIDE.read_text().strip():raise RuntimeError("native checkpoint hash mismatch")
    state=torch.load(REC,map_location="cpu",weights_only=False);row=latest_row()
    if state["tokens_seen"]!=50_000_000 or int(row["training_tokens"])!=50_000_000:raise RuntimeError("trainer did not reach exact 50M")
    if state["source"]!="DATA-D-BROAD-v2r1" or state["data_manifest_sha256"]!="f42df21984b27e77b6901a6335f751cee911efd1ff4579aba038a577255d601a":raise RuntimeError("terminal lineage mismatch")
    if state["source_tokens"]!={"fineweb_edu":25_000_000,"wikipedia":12_500_000,"fineweb":12_500_000}:raise RuntimeError("terminal source mixture mismatch")
    return state,row
def references()->dict[str,list[str]]:
    from prepare_phase7a_data import reference_documents
    refs=reference_documents();tok=load_tokenizer(TOKENIZER);raw=np.memmap(ART/"data/phase7a/common_validation.int32",mode="r",dtype="<i4")
    refs["common_validation"]=[tok.decode(raw[i:i+128].astype(int).tolist()) for i in range(0,len(raw),128)]
    return refs
def preserve_manifests(pool:dict,validation:dict):
    PUBLIC.mkdir(parents=True,exist_ok=True)
    for p in POOL.glob("*.manifest.json"):shutil.copy2(p,PUBLIC/p.name)
    shutil.copy2(POOL/"manifest.json",PUBLIC/"canonical.manifest.json");shutil.copy2(VALID/"validation.manifest.json",PUBLIC/"validation.manifest.json")
    summary={"generator_commit":pool["generator_commit"],"final_tokens":pool["token_count"],"example_count":pool["example_count"],"wall_seconds":pool["wall_seconds"],"tokens_per_second":pool["tokens_per_second"],"first_100m_wall_seconds":pool["first_100m_wall_seconds"],"selected_target":pool["selected_target"],"status":pool["status"],"family_counts":pool["family_counts"],"difficulty_counts":pool["difficulty_counts"],"peak_rss_bytes":pool["peak_rss_bytes"],"disk_bytes":pool["disk_bytes"],"free_disk_bytes":pool["free_disk_bytes"],"validation":validation,"views":pool["views"]}
    atomic(PUBLIC/"materialization_report.json",summary)
def main():
    state={"stage":"WAIT_TRAINER","started":datetime.now(timezone.utc).isoformat()};atomic(STATE,state);event("ORCHESTRATOR_STARTED")
    while trainer_present():time.sleep(60)
    try:
        terminal,row=verify_terminal();event("TRAINING_VERIFIED",tokens=terminal["tokens_seen"],sha256=sha256_file(REC));state["stage"]="EVALUATION";atomic(STATE,state)
        native=ART/"checkpoints/co4-l-data-d-v2r1-50m/native.pt";native.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REC,native);native.with_suffix(".sha256").write_text(sha256_file(native)+"\n")
        wiki=ART/"wikitext_data_d_v2r1_50m.json";gibc=ART/"gibc_data_d_v2r1_50m_raw.json"
        run("wikitext",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(native),"--tokenizer",str(TOKENIZER),"--output",str(wiki),"--threads","4"])
        run("gibc",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(native),"--tokenizer",str(TOKENIZER),"--output",str(gibc),"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","4"])
        w=json.loads(wiki.read_text());g=json.loads(gibc.read_text());metrics={"tokens_trained":50_000_000,"parameters":15_949_760,"val_loss":float(row["common_validation_loss"]),"val_ppl":float(row["common_validation_perplexity"]),"data_d_balanced_validation_loss":float(row["data_d_balanced_validation_loss"]),"fineweb_edu_validation_loss":float(row["fineweb_edu_validation_loss"]),"wikipedia_validation_loss":float(row["wikipedia_validation_loss"]),"fineweb_validation_loss":float(row["fineweb_validation_loss"]),"wall_seconds":float(row["cumulative_training_seconds"]),"tokens_per_second":float(row["tokens_per_second"]),"checkpoint_sha256":sha256_file(native),"data_manifest_sha256":terminal["data_manifest_sha256"],"wikitext_103_perplexity":w["perplexity"],"wikitext_bits_per_byte":w["bits_per_byte"],**{t:g["results"][t]["acc,none"] for t in ("hellaswag","arc_easy","piqa","winogrande")}}
        atomic(ART/"phase7f_v2r1_50m_metrics.json",metrics)
        bundle=ART/"cache/co4-l-data-d-v2r1-50m";manifest=export_checkpoint(native,bundle,"co4-l-data-d-v2r1-50m","final",TOKENIZER,TOKENIZER_REPORT,ART/"data_d_v2r1_manifest.json",metrics)
        token=get_token()
        if not token:raise RuntimeError("no cached Hugging Face credential")
        revision=upload_checkpoint(bundle,"insightlabs38-pixel/LatticeLM-research","experiments/co4-l-data-d-v2r1-50m",token)
        atomic(ART/"phase7f_v2r1_50m_huggingface.json",{"repo_id":"insightlabs38-pixel/LatticeLM-research","path":"experiments/co4-l-data-d-v2r1-50m","revision":revision,"verified":True,"native_checkpoint_sha256":sha256_file(native),"model_safetensors_sha256":sha256_file(bundle/"model.safetensors"),"tokenizer_sha256":sha256_file(TOKENIZER),"private_exact_resume_state_preserved":True,"safetensors_roundtrip_verified":manifest["safetensors_roundtrip_verified"]})
        event("MODEL_PUBLISHED",revision=revision);state["stage"]="CONTAMINATION_REGISTRY";atomic(STATE,state)
        guard=compile_contamination_registry(references());event("CONTAMINATION_REGISTRY_READY",ngrams=len(guard["grams"]))
        validation=build_validation(VALID,TOKENIZER,40,99173,guard);event("VALIDATION_COMPLETE",examples=validation["examples"])
        state["stage"]="MATERIALIZATION";atomic(STATE,state);commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
        pool=build_canonical(POOL,TOKENIZER,100_000_000,7319,1_000_000,commit,7.5*3600,8*3600,True,8<<30,False,guard)
        preserve_manifests(pool,validation);event("MATERIALIZATION_COMPLETE",tokens=pool["token_count"],status=pool["status"])
        state.update({"stage":"COMPLETE","hf_revision":revision,"generated_tokens":pool["token_count"]});atomic(STATE,state)
        subprocess.run(["git","add","artifacts/data_d_v2r1_training_curve.csv","artifacts/phase7f_v2r1_50m_metrics.json","artifacts/wikitext_data_d_v2r1_50m.json","artifacts/phase7f_v2r1_50m_huggingface.json","artifacts/lattice_reason","artifacts/lattice_reason_overnight_state.json"],cwd=ROOT,check=True)
        subprocess.run(["git","commit","-m","Finalize DATA-D 50M and LatticeReason materialization"],cwd=ROOT,check=True)
        subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True);event("FINAL_COMMIT_PUSHED")
    except Exception as exc:
        state.update({"stage":"FAILED","error":f"{type(exc).__name__}: {exc}"});atomic(STATE,state);event("ORCHESTRATOR_FAILED",error=state["error"]);raise
if __name__=="__main__":main()
