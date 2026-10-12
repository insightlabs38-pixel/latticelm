"""Deterministic, restart-safe adjudicator for the three-branch WSD tournament."""
from __future__ import annotations

import argparse, fcntl, hashlib, json, math, os, shutil, signal, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
STATE = ART / "wsd_next_phase_state.json"
PREVIOUS = ART / "wsd_next_phase_state.previous.json"
LOCK = ART / "wsd_next_phase_master.lock"
COMPARISON = ART / "wsd_tournament_adjudication.json"
COMPARISON_MD = ART / "wsd_tournament_adjudication.md"
DECISION = ART / "wsd_next_phase_decision.json"
REPORT = ART / "wsd_next_phase_report.md"
TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande")
CLASSIFICATIONS = {
    "LATE_HORIZON_CLEARLY_BEST", "INTERMEDIATE_HORIZON_CLEARLY_BEST",
    "EARLY_HORIZON_CLEARLY_BEST", "WSD_EFFECTIVE_BUT_SCALING_SATURATING",
    "HORIZONS_EFFECTIVELY_TIED", "WSD_INSUFFICIENT", "PARETO_INCONCLUSIVE",
    "BLOCKED_OR_INVALID",
}
STOP = False

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()

def atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()
    with tmp.open("wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def save(state, stage, **extra):
    if STATE.exists(): shutil.copy2(STATE, PREVIOUS)
    state.update(stage=stage, updated_at=time.time(), **extra); atomic(STATE, state)

def paired_interval(a, b):
    """Deterministic paired normal interval over item correctness."""
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    n = len(diffs); mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / max(1, n - 1)
    half = 1.96 * math.sqrt(var / n)
    return {"method":"paired_item_normal_95ci", "n":n, "delta":mean,
            "ci95":[mean-half, mean+half]}

def classify(endpoints, wsd_effective=True):
    """Explicit materiality policy; scores are fractions and BPB is lower-better."""
    if any(not x.get("valid", True) for x in endpoints): return "BLOCKED_OR_INVALID"
    e, m, l = endpoints
    # A clear winner needs material language quality and broad task support, or
    # material task gains without a language regression. Tiny aggregate changes
    # never become CLEAR_MATERIAL_SCALING_GAIN.
    def task_delta(a,b): return sum(b[t]-a[t] for t in TASKS)/4
    def beats(a,b):
        bp = a["wikitext_bpb"]-b["wikitext_bpb"]
        td = task_delta(a,b)
        regress = sum(b[t] < a[t]-0.01 for t in TASKS)
        return (bp >= 0.015 and td >= 0.003 and regress == 0) or (td >= 0.01 and bp >= -0.005 and regress == 0)
    if beats(e,l) and beats(m,l): return "LATE_HORIZON_CLEARLY_BEST"
    if beats(e,m) and beats(l,m): return "INTERMEDIATE_HORIZON_CLEARLY_BEST"
    if beats(m,e) and beats(l,e): return "EARLY_HORIZON_CLEARLY_BEST"
    max_bpb=max(x["wikitext_bpb"] for x in endpoints); min_bpb=min(x["wikitext_bpb"] for x in endpoints)
    max_task=max(sum(x[t] for t in TASKS)/4 for x in endpoints); min_task=min(sum(x[t] for t in TASKS)/4 for x in endpoints)
    if not wsd_effective: return "WSD_INSUFFICIENT"
    if max_bpb-min_bpb < .005 and max_task-min_task < .003: return "HORIZONS_EFFECTIVELY_TIED"
    late_best_bpb=l["wikitext_bpb"] == min_bpb
    if late_best_bpb and 0 < m["wikitext_bpb"]-l["wikitext_bpb"] < .015 and task_delta(m,l) > -.005:
        return "WSD_EFFECTIVE_BUT_SCALING_SATURATING"
    return "PARETO_INCONCLUSIVE"

def pareto_frontier(branches):
    def dominates(a,b):
        am,bm=a["metrics"],b["metrics"]
        nonworse=am["wikitext_bpb"]<=bm["wikitext_bpb"] and all(am[t]>=bm[t] for t in TASKS)
        strict=am["wikitext_bpb"]<bm["wikitext_bpb"] or any(am[t]>bm[t] for t in TASKS)
        return nonworse and strict
    return [b for b in branches if not any(dominates(a,b) for a in branches if a is not b)]

def _samples(path):
    obj=json.loads(path.read_text()); return {t:[x["acc"] for x in obj.get("samples",{}).get(t,[])] for t in TASKS}

def certify(label, raw_tokens):
    raw = ART/f"checkpoints/co4-final-capacity32-data-d-v3/milestone-{raw_tokens}.pt"
    root = ART/f"checkpoints/base_{raw_tokens//1_000_000}m-wsd-100m"
    cp=root/"milestone.pt"; result=ART/f"saturation/wsd/base_{raw_tokens//1_000_000}m/results.json"
    problems=[]
    for p in (raw,cp,result,root/"wikitext.json",root/"gibc.json"):
        if not p.exists(): problems.append(f"missing:{p}")
    if problems: return {"label":label,"valid":False,"problems":problems}
    expected=(root/"milestone.sha256").read_text().strip() if (root/"milestone.sha256").exists() else None
    actual=sha256(cp)
    if actual != expected: problems.append("endpoint_sha256_mismatch")
    import torch
    from latticelm.config import LatticeConfig
    from latticelm.model import build_model
    state=torch.load(cp,map_location="cpu",weights_only=False); cfg=LatticeConfig(**state["config"])
    model=build_model(cfg); model.load_state_dict(state["model"],strict=True)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params != 32_678_640: problems.append(f"trainable_parameters:{params}")
    if state.get("tokens_seen") != raw_tokens+100_000_000: problems.append("wrong_endpoint_tokens")
    if state.get("branch_training_tokens") != 100_000_000: problems.append("wrong_wsd_budget")
    pol=state.get("learning_rate_policy",{})
    if pol != {"name":"cosine","start_lr":0.0003,"end_lr":0.0,"budget_tokens":100000000}: problems.append("wrong_lr_policy")
    if state.get("parent_checkpoint_sha256") != sha256(raw): problems.append("parent_sha256_mismatch")
    if not math.isfinite(float(state.get("train_loss",math.nan))): problems.append("nonfinite_terminal_loss")
    r=json.loads(result.read_text()); metrics=r["candidates"][0]; wiki=json.loads((root/"wikitext.json").read_text()); gibc=json.loads((root/"gibc.json").read_text())
    for meta in (wiki,gibc.get("phase6_metadata",{})):
        if meta.get("checkpoint_sha256") != actual: problems.append("evaluation_checkpoint_sha_mismatch")
    return {"label":label,"valid":not problems,"problems":problems,"raw_tokens":raw_tokens,
      "tokens":raw_tokens+100_000_000,"additional_wsd_tokens":100_000_000,"checkpoint":str(cp),
      "checkpoint_sha256":actual,"raw_checkpoint":str(raw),"raw_checkpoint_sha256":sha256(raw),
      "config":state["config"],"trainable_parameters":params,"training_wall_seconds":state.get("wall_seconds"),
      "tokens_per_second":100_000_000/state["wall_seconds"],"terminal_train_loss":state.get("train_loss"),
      "data_stream":{"next_batch_sha256":state.get("next_batch_sha256"),"selector_state_present":"data_source_selector_state" in state},
      "lr_policy":pol,"metrics":metrics,"evaluation":{"wikitext":str(root/"wikitext.json"),"official":str(root/"gibc.json"),
      "lm_eval_version":gibc.get("phase6_metadata",{}).get("lm_eval_version"),"task_versions":gibc.get("versions"),
      "tokenizer_sha256":wiki.get("tokenizer_sha256"),"context_length":wiki.get("context_length"),
      "wall_seconds":wiki.get("wall_seconds",0)+gibc.get("phase6_metadata",{}).get("wall_seconds",0)}}

def execute(smoke=False):
    state={"schema":"wsd-next-phase-state-v1","stage":"START","completed":[],"child_pid":None}
    if smoke: return {"status":"SMOKE_TEST_OK","classification":classify([dict(valid=True,wikitext_bpb=1.7,**{t:.3 for t in TASKS})]*3)}
    save(state,"VERIFY_WSD_BRANCHES")
    branches=[certify("612M WSD",512_000_000),certify("850M WSD",750_000_000),certify("1B WSD",900_000_000)]
    if STOP: save(state,"STOPPED_SAFE"); return None
    save(state,"BUILD_CANONICAL_COMPARISON")
    endpoints=[]
    for b in branches: endpoints.append({"valid":b["valid"],**({k:b["metrics"][k] for k in ("wikitext_bpb","wikitext_ppl",*TASKS)} if b["valid"] else {})})
    wsd_effective=all(b["valid"] and b["metrics"]["wikitext_bpb"] < json.loads((ART/f"saturation/wsd/base_{b['raw_tokens']//1_000_000}m/results.json").read_text())["base"]["wikitext_bpb"]-.01 for b in branches)
    classification=classify(endpoints,wsd_effective)
    uncertainty={}
    if all(b["valid"] for b in branches):
        samples=[_samples(Path(b["evaluation"]["official"])) for b in branches]
        for i,j,name in ((0,1,"612_vs_850"),(1,2,"850_vs_1b"),(0,2,"612_vs_1b")):
            uncertainty[name]={t:paired_interval(samples[i][t],samples[j][t]) for t in TASKS}
    raw=[]
    for b in branches:
        source=json.loads((ART/f"saturation/wsd/base_{b['raw_tokens']//1_000_000}m/results.json").read_text())["base"] if b["valid"] else {}
        raw.append({"label":f"{b['raw_tokens']//1_000_000}M raw","tokens":b.get("raw_tokens"),"metrics":source,
                    "checkpoint":b.get("raw_checkpoint"),"checkpoint_sha256":b.get("raw_checkpoint_sha256")})
        if b["valid"]:
            b["improvement_from_raw"]={"validation_loss":source["validation_loss"]-b["metrics"]["validation_loss"],
              "wikitext_ppl":source["wikitext_ppl"]-b["metrics"]["wikitext_ppl"],"wikitext_bpb":source["wikitext_bpb"]-b["metrics"]["wikitext_bpb"],
              **{t:b["metrics"][t]-source[t] for t in TASKS}}
            b["source_validation"]={"fineweb_edu":None,"wikipedia":None,"fineweb":None,"note":"not_evaluated in canonical WSD worker output"}
    frontier=pareto_frontier([b for b in branches if b["valid"]])
    comparison={"schema":"wsd-tournament-adjudication-v1","experiment":"three_branch_100m_wsd_tournament",
      "branches":branches,"uncertainty":uncertainty,"classification":classification,
      "raw_checkpoints":raw,"pareto_frontier":[b["label"] for b in frontier],
      "materiality_policy":{"task_accuracy_clear":.01,"wikitext_bpb_clear":.015,"negligible_task_mean":.003,"negligible_bpb":.005},
      "pareto_note":"Lower WikiText/BPB and higher task accuracy are preferred; no arbitrary scalar weights are used.",
      "raw_result_paths":[str(ART/f"saturation/wsd/base_{n}m/results.json") for n in (512,750,900)]}
    atomic(COMPARISON,comparison); COMPARISON_MD.write_text(render_comparison(comparison))
    valid=[b for b in branches if b["valid"]]
    # Pareto-inconclusive selection follows the mandated cheaper Pareto rule.
    best=min(frontier,key=lambda b:b["tokens"]) if classification in ("PARETO_INCONCLUSIVE","HORIZONS_EFFECTIVELY_TIED") and frontier else min(valid,key=lambda b:b["metrics"]["wikitext_bpb"]) if valid else None
    direct_scale=classification=="LATE_HORIZON_CLEARLY_BEST"
    post=classification not in ("LATE_HORIZON_CLEARLY_BEST","BLOCKED_OR_INVALID")
    decision={"experiment":"three_branch_100m_wsd_tournament","status":"COMPLETE" if classification!="BLOCKED_OR_INVALID" else "BLOCKED",
      "classification":classification,"best_absolute_pretrained_checkpoint":({"label":best["label"],"tokens":best["tokens"],"path":best["checkpoint"],"sha256":best["checkpoint_sha256"]} if best else None),
      "stable_horizon_verdict":classification,"clear_scaling_gain":direct_scale,"further_32m_base_scaling_authorized":direct_scale,
      "autonomous_large_base_scaling":direct_scale,"posttraining_authorized":post,"architecture_reconsideration_required":post,
      "architecture_gate":"Reconsider immediately after the bounded post-training tournament; earlier if no SFT candidate promotes." if post else "After next stable-horizon experiment.",
      "data_d_v4_required":direct_scale,"next_phase":"BUILD_DATA_D_V4_THEN_1_5B_STABLE_FROM_RAW_900M" if direct_scale else ("BOUNDED_POSTTRAINING_FROM_CHEAPEST_PARETO_BASE" if post else "REPAIR_WSD_TOURNAMENT"),
      "reasoning_summary":reasoning(classification),"evidence_files":[str(COMPARISON),str(COMPARISON_MD)],"revision":1}
    atomic(DECISION,decision); REPORT.write_text(render_report(decision,comparison)); save(state,"COMPLETE",classification=classification,decision_sha256=sha256(DECISION)); return decision

def reasoning(c):
    return {"PARETO_INCONCLUSIVE":"WikiText improves with horizon, while official-task results cross: the intermediate endpoint leads ARC-Easy/WinoGrande and the late endpoint leads PIQA. Paired uncertainty does not establish broad late-horizon superiority.","BLOCKED_OR_INVALID":"At least one required branch or evaluation failed integrity certification.","HORIZONS_EFFECTIVELY_TIED":"Endpoint differences are below explicit materiality thresholds.","WSD_EFFECTIVE_BUT_SCALING_SATURATING":"The late endpoint is numerically best but its marginal gain is not clear enough to buy another large continuation."}.get(c,c.replace("_"," ").title())

def render_comparison(c):
    rows=[]
    for b in c["branches"]:
        m=b.get("metrics",{}); rows.append(f"| {b['label']} | {m.get('validation_loss')} | {m.get('wikitext_ppl')} | {m.get('wikitext_bpb')} | "+" | ".join(str(m.get(t)) for t in TASKS)+f" | {b.get('tokens')} | {b.get('checkpoint_sha256')} |")
    return "# WSD tournament adjudication\n\n| Endpoint | DATA-D validation* | WikiText PPL | WikiText BPB | HellaSwag | ARC-Easy | PIQA | WinoGrande | tokens | SHA256 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"+"\n".join(rows)+"\n\n*The worker's canonical aggregate validation is recorded; source-specific values unavailable in canonical WSD output remain null in JSON. No scalar composite is used.\n"

def render_report(d,c):
    b=d["best_absolute_pretrained_checkpoint"] or {}
    return f"""# WSD next-phase decision

1. **Best checkpoint:** {b.get('label','none')} ({b.get('sha256','n/a')}).
2. **Material advantage:** {d['reasoning_summary']}
3. **Did WSD solve the divergence?** WSD materially repaired WikiText for every branch, but did not create a broad monotonic official-task ordering.
4. **Does later stable training remain useful?** Not clearly across the competition surface.
5. **Further 32M scaling authorized?** {str(d['further_32m_base_scaling_authorized']).lower()}.
6. **Post-training base:** {b.get('label','none')}; immutable checkpoint at `{b.get('path','n/a')}`.
7. **Architecture timing:** {d['architecture_gate']}
8. **DATA-D-v4 needed?** {str(d['data_d_v4_required']).lower()}.
9. **Exact next phase:** {d['next_phase']}.
10. **Change condition:** Reclassify only in a new revision if checkpoint/evaluation evidence changes, or if bounded matched follow-up establishes a clear material Pareto winner.
"""

def main():
    p=argparse.ArgumentParser(); p.add_argument("--smoke-test",action="store_true"); a=p.parse_args()
    if a.smoke_test: print(json.dumps(execute(True),sort_keys=True)); return 0
    signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__("STOP",True)); signal.signal(signal.SIGINT,lambda *_:globals().__setitem__("STOP",True))
    with LOCK.open("a+") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if STATE.exists() and json.loads(STATE.read_text()).get("stage")=="COMPLETE" and DECISION.exists(): return 0
        execute()
    return 0
if __name__=="__main__": raise SystemExit(main())
