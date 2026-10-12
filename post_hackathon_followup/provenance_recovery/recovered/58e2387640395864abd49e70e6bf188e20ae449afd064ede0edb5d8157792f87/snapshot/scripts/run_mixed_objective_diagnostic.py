#!/usr/bin/env python3
"""Screen low-alpha blends from the isolated mixed-objective recovery pilot."""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latticelm.posttraining.state import atomic_json, sha256
import scripts.run_posttraining_salvage as salvage

OUT = ROOT / "artifacts/posttraining_salvage_followup/mixed_objective_diagnostic"
BASE = ROOT / "artifacts/posttraining/baseline/base.pt"
ENDPOINT = OUT / "candidates/joint-mix-500k/latest.pt"
TOKENIZER = ROOT / "artifacts/tokenizers/final_corpus_4k.json"
MANIFEST = ROOT / "artifacts/data/data_d_v4/canonical-2250m/manifest.json"
DEADLINE = datetime.fromisoformat("2026-10-01T05:40:19.945785+00:00").timestamp()
STATE = OUT / "evaluation_state.json"


def now(): return datetime.now(timezone.utc).isoformat()
def save(s): atomic_json(STATE, s)
def remaining(): return DEADLINE - time.time()
def ensure_time(seconds):
    if remaining() < seconds:
        raise TimeoutError(f"only {remaining():.1f}s remain; need {seconds}s reserve")
def run(label, argv, timeout):
    log = OUT / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        p = subprocess.run([str(x) for x in argv], cwd=ROOT, env={**os.environ, "PYTHONPATH":str(ROOT)},
                           stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode: raise RuntimeError(f"{label} exited {p.returncode}; see {log}")
    return log

def load(path): return json.loads(Path(path).read_text())
def write_report(s):
    base = s["baseline"]
    lines = ["# Mixed-objective recovery diagnostic", "", f"Updated: {now()}",
             f"Deadline: {datetime.fromtimestamp(DEADLINE, timezone.utc).isoformat()}", "",
             "Baseline gates: Data D ≤ 1.20× BASE; WikiText BPB ≤ 1.25× BASE. Reasoning proxy is screening only; GIBC is the official capability check.", "",
             "| Candidate | Alpha | Data D | D ratio | Wiki BPB | BPB ratio | Proxy acc | Proxy Δ | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for cid, c in s["candidates"].items():
        e = c.get("screen")
        if not e:
            lines.append(f"| {cid} | {c['alpha']:.2f} | pending | — | pending | — | pending | — | {c['status']} |")
            continue
        dr=e["data_d"]["loss"]/base["data_d_loss"]; wr=e["wikitext_bpb"]/base["wikitext_bpb"]
        lines.append(f"| {cid} | {c['alpha']:.2f} | {e['data_d']['loss']:.4f} | {dr:.3f}× | {e['wikitext_bpb']:.4f} | {wr:.3f}× | {e['proxy_accuracy']:.4f} | {e['proxy_accuracy']-base['proxy_accuracy']:+.4f} | D {'PASS' if dr<=1.2 else 'FAIL'}; Wiki {'PASS' if wr<=1.25 else 'FAIL'} |")
    if s.get("gibc"):
        vals=s["gibc"]["raw"].values();mean=sum(vals)/len(s["gibc"]["raw"])
        lines += ["", f"Full GIBC: {s['gibc']['candidate']} mean={mean:.4f}, BASE={base['gibc_mean']:.4f}, delta={mean-base['gibc_mean']:+.4f}."]
    lines += ["", f"Phase: {s['phase']}", ""]
    tmp=OUT/".mixed_objective_report.md.tmp";tmp.write_text("\n".join(lines));os.replace(tmp,OUT/"mixed_objective_report.md")

def main():
    if sha256(BASE)!="94b91a2d39c2889209cb47b2e04b2c0d307a25610c40c5a124b595020eb762b0": raise RuntimeError("BASE hash changed")
    result=load(OUT/"candidates/joint-mix-500k/result.json")
    if result["status"]!="COMPLETE" or result["training_tokens"]!=500000: raise RuntimeError("pilot did not complete exact budget")
    if sha256(ENDPOINT)!=result["checkpoint_sha256"] or result["parent_checkpoint_sha256"]!="f7d3b03d2d76986f1b74be67e7b07e80b91b5905111315248b11611e637a3cc9": raise RuntimeError("pilot provenance hash mismatch")
    import torch
    torch.set_num_threads(4)
    base_ck=torch.load(BASE,map_location="cpu",weights_only=False)
    end_ck=torch.load(ENDPOINT,map_location="cpu",weights_only=False)
    if base_ck["config"]!=end_ck["config"] or list(base_ck["model"])!=list(end_ck["model"]): raise RuntimeError("architecture/state key mismatch")
    for k,v in end_ck["model"].items():
        if not torch.isfinite(v).all(): raise RuntimeError(f"nonfinite endpoint tensor: {k}")
    old=load(STATE) if STATE.exists() else None
    if old:
        if old["baseline"]["sha256"]!=sha256(BASE) or old["endpoint"]["sha256"]!=sha256(ENDPOINT): raise RuntimeError("resume provenance mismatch")
        s=old
    else:
        base_eval=load(ROOT/"artifacts/posttraining_salvage_followup/candidates/BASE/final_eval/data_d.json") if (ROOT/"artifacts/posttraining_salvage_followup/candidates/BASE/final_eval/data_d.json").exists() else None
        parent_state=load(ROOT/"artifacts/posttraining_salvage_followup/state.json")
        be=parent_state["final_evaluations"]["BASE"]
        s={"schema":"mixed-objective-evaluation-v1","phase":"SCREENING","created_at":now(),"deadline_at":"2026-10-01T05:40:19.945785+00:00",
           "baseline":{"sha256":sha256(BASE),"data_d_loss":be["data_d"]["loss"],"wikitext_bpb":be["wikitext_bpb"],"gibc_mean":sum(be["gibc_raw"].values())/4,"proxy_accuracy":be["proxy_1024"]["accuracy"]},
           "endpoint":{"sha256":sha256(ENDPOINT),"parent_sha256":result["parent_checkpoint_sha256"],"training_tokens":result["training_tokens"],"wall_seconds":result["wall_seconds"],"processed_tokens":result["processed_tokens"],"lr":result["lr"],"replay_fraction":result["replay"],"updates":result["updates"]},"candidates":{},"gibc":None}
    for alpha in (0.05,0.10,0.15):
        aid=f"{alpha:.2f}"; cid=f"joint-mix-500k-a{aid}"; root=OUT/"candidates"/cid; root.mkdir(parents=True,exist_ok=True)
        if cid not in s["candidates"]:
            merged={}
            for k,x in base_ck["model"].items():
                y=end_ck["model"][k]
                if x.shape!=y.shape or x.dtype!=y.dtype: raise RuntimeError(f"tensor incompatibility: {k}")
                merged[k]=torch.lerp(x,y,alpha) if x.is_floating_point() else x.clone()
                if not torch.isfinite(merged[k]).all(): raise RuntimeError(f"nonfinite blend tensor: {k}")
            payload={"schema":"posttraining-salvage-interpolation-v1","config":base_ck["config"],"model":merged,
                     "merge":{"base_sha256":sha256(BASE),"specialist_sha256":sha256(ENDPOINT),"alpha":alpha}}
            dest=root/"checkpoint.pt";tmp=root/".checkpoint.pt.tmp";torch.save(payload,tmp);os.replace(tmp,dest)
            side_tmp=root/".checkpoint.pt.sha256.tmp";side_tmp.write_text(sha256(dest)+"\n");os.replace(side_tmp,dest.with_suffix(".pt.sha256"))
            s["candidates"][cid]={"alpha":alpha,"checkpoint":str(dest),"sha256":sha256(dest),"status":"CREATED","screen":None}
            save(s)
        c=s["candidates"][cid];ckpt=Path(c["checkpoint"])
        if sha256(ckpt)!=c["sha256"]: raise RuntimeError(f"blend SHA mismatch: {cid}")
        evalroot=root/"final_eval";evalroot.mkdir(exist_ok=True)
        proxy=evalroot/"proxy1024.json"
        if not proxy.exists():
            ensure_time(18*60)
            run(cid+"-proxy1024",[sys.executable,"scripts/evaluate_posttraining_proxy.py","--checkpoint",ckpt,"--parent",BASE,"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--output",proxy,"--examples","1024","--threads","16"],timeout=max(120,remaining()-12*60))
        p=load(proxy)
        if p.get("checkpoint_sha256")!=c["sha256"] or p.get("examples")!=1024: raise RuntimeError(f"invalid proxy result: {cid}")
        dpath=evalroot/"data_d.json"
        if not dpath.exists(): atomic_json(dpath,salvage.official_validation(ckpt))
        d=load(dpath)
        wpath=evalroot/"wikitext.json"
        if not wpath.exists():
            ensure_time(15*60)
            tmp=evalroot/".wikitext.json.tmp"
            run(cid+"-wiki",[sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",ckpt,"--tokenizer",TOKENIZER,"--output",tmp,"--threads","16","--batch-size","16"],timeout=max(120,remaining()-12*60))
            w=load(tmp)
            if w.get("checkpoint_sha256")!=c["sha256"]: raise RuntimeError(f"invalid WikiText result: {cid}")
            os.replace(tmp,wpath)
        w=load(wpath)
        c["screen"]={"data_d":d,"wikitext_bpb":w["bits_per_byte"],"proxy_accuracy":p["v2_ranking_accuracy"],"proxy_margin":p["v2_mean_margin"],"proxy_parent_delta":p.get("paired_vs_parent",{}).get("normalized",{}),"evaluated_at":now()};c["status"]="SCREENED";save(s);write_report(s)
    eligible=[]
    for cid,c in s["candidates"].items():
        e=c.get("screen")
        if not e: continue
        if e["data_d"]["loss"]<=1.2*s["baseline"]["data_d_loss"] and e["wikitext_bpb"]<=1.25*s["baseline"]["wikitext_bpb"] and e["proxy_accuracy"]>s["baseline"]["proxy_accuracy"]: eligible.append((e["proxy_accuracy"],cid))
    if eligible and not s.get("gibc") and remaining()>15*60:
        _,cid=max(eligible); c=s["candidates"][cid]; evalroot=Path(c["checkpoint"]).parent/"final_eval";dest=evalroot/"gibc.json"
        ensure_time(12*60)
        run(cid+"-gibc",[sys.executable,"scripts/evaluate_gibc.py","--checkpoint",c["checkpoint"],"--tokenizer",TOKENIZER,"--output",dest,"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","16","--batch-size","16"],timeout=max(120,remaining()-2*60))
        raw=load(dest)
        if raw.get("phase6_metadata",{}).get("checkpoint_sha256")!=c["sha256"]: raise RuntimeError("GIBC checkpoint identity mismatch")
        scores={t:raw["results"][t]["acc,none"] for t in ("hellaswag","arc_easy","piqa","winogrande")}
        s["gibc"]={"candidate":cid,"raw":scores,"evaluated_at":now()};s["phase"]="COMPLETE";save(s);write_report(s)
    else:
        s["phase"]="SCREENED_NO_GIBC" if not eligible else "SCREENED_GIBC_SKIPPED_DEADLINE";save(s);write_report(s)
    print(json.dumps({"phase":s["phase"],"screens":{k:v.get("screen") for k,v in s["candidates"].items()},"gibc":s.get("gibc")},indent=2))

if __name__=="__main__": main()
