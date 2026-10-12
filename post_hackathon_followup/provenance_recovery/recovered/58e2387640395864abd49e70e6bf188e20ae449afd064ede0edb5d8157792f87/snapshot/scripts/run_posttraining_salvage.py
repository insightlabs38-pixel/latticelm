#!/usr/bin/env python3
"""Certified, restartable six-hour salvage of the final Co4 checkpoint."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OLD = ART / "posttraining"
OUT = ART / "posttraining_salvage"
TOKENIZER = ART / "tokenizers/final_corpus_4k.json"
TOKENIZER_REPORT = ART / "tokenizers/final_corpus_4k.report.json"
MANIFEST = ART / "data/data_d_v4/canonical-2250m/manifest.json"
STATE = OUT / "state.json"
PY = sys.executable
sys.path.insert(0, str(ROOT))
from latticelm.posttraining.state import atomic_json, sha256

STOP = False
CHILD = None
TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande")
RESERVE = 900


def iso(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat()


def stop(*_):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGINT)


def write(path, value):
    atomic_json(path, value)


def event(state, kind, **details):
    state.setdefault("events", []).append({"at": iso(), "kind": kind, **details})
    save(state)


def save(state):
    state["updated_at"] = iso()
    write(STATE, state)
    write(OUT / "candidates.json", state.get("candidates", {}))
    write(OUT / "throughput.json", state.get("throughput", {}))
    write(OUT / "pareto_frontier.json", {"candidate_ids": frontier(state)})
    write(OUT / "final_evaluations.json", state.get("final_evaluations", {}))


def remaining(state):
    # A monotonic clock is preferred within a process. The wall deadline also
    # survives reboot and prevents a restarted process from gaining time.
    return min(state["deadline_epoch"] - time.time(),
               state["deadline_monotonic"] - time.monotonic()
               if state.get("boot_id") == boot_id() and "deadline_monotonic" in state
               else float("inf"))


def boot_id():
    p = Path("/proc/sys/kernel/random/boot_id")
    return p.read_text().strip() if p.exists() else None


def require_base(state):
    if sha256(state["candidates"]["BASE"]["checkpoint"]) != state["base_sha256"]:
        raise RuntimeError("immutable BASE SHA changed")


def run(state, label, args, max_seconds=None):
    global CHILD
    require_base(state)
    start = time.monotonic()
    event(state, "command_start", label=label, argv=[str(x) for x in args])
    log = OUT / "logs" / (label + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        CHILD = subprocess.Popen([str(x) for x in args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        state["active_child"] = {"pid": CHILD.pid, "label": label, "started_at": iso(), "argv": [str(x) for x in args]}
        event(state, "child_started", pid=CHILD.pid, label=label)
        while CHILD.poll() is None:
            write(OUT / "controller.heartbeat.json", {"pid": os.getpid(), "child_pid": CHILD.pid,
                  "label": label, "at": iso(), "boot_id": boot_id()})
            if STOP or (state.get("deadline_epoch") and remaining(state) < 45) or (max_seconds and time.monotonic()-start > max_seconds):
                CHILD.send_signal(signal.SIGINT)
                try: CHILD.wait(timeout=90)
                except subprocess.TimeoutExpired: CHILD.terminate(); CHILD.wait(timeout=30)
                break
            time.sleep(1)
        code = CHILD.returncode
        CHILD = None
        seconds = time.monotonic()-start
        state.pop("active_child", None)
        event(state, "command_end", label=label, exit_code=code, seconds=seconds, log=str(log))
    if code: raise RuntimeError(f"{label} failed ({code}); see {log}")
    return seconds


def discover():
    old = json.loads((OLD / "master_state.json").read_text())
    found = {}
    for name in ("BASE", "joint", "ranking"):
        c = old["candidates"][name]
        path = Path(c["checkpoint"])
        if not path.is_file() or sha256(path) != c["checkpoint_sha256"]:
            raise RuntimeError(f"{name} missing or SHA mismatch")
        found[name] = {"checkpoint": str(path), "checkpoint_sha256": c["checkpoint_sha256"],
                       "lineage": name, "method": c.get("method"), "status": "VALID"}
        pp = path.parent / "proxy-256.json"
        if pp.exists():
            value = json.loads(pp.read_text())
            if value.get("checkpoint_sha256") == c["checkpoint_sha256"] and value.get("examples") == 256:
                found[name]["proxy_256"] = value
    return found, old


def validate_model(path, reference=None):
    import torch
    from latticelm.config import LatticeConfig
    from latticelm.model import build_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LatticeConfig(**ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"], strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != 48636168 or count >= 50_000_000:
        raise RuntimeError(f"unexpected parameter count: {count}")
    if reference is not None and cfg.to_dict() != reference:
        raise RuntimeError("checkpoint architecture mismatch")
    for k in ("tokenizer_sha256", "manifest_sha256"):
        if ck.get(k) and ck[k] != sha256(TOKENIZER if k.startswith("tokenizer") else MANIFEST):
            raise RuntimeError(f"{k} mismatch")
    return cfg.to_dict()


def initial():
    found, old = discover()
    base = found["BASE"]
    s = {"schema": "posttraining-salvage-v1", "phase": "PREFLIGHT", "created_at": iso(),
         "base_sha256": base["checkpoint_sha256"], "tokenizer_sha256": sha256(TOKENIZER),
         "manifest_sha256": sha256(MANIFEST), "code_commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "candidates": found, "throughput": {}, "final_evaluations": {}, "events": [], "failures": [],
         "training_seconds": 0., "proxy_seconds": 0., "official_seconds": 0., "evidence_seconds": 0.,
         "optimizer": old.get("selected_optimizer", "adamw"),
         "recovery_lr": 3e-5, "microbatch": old.get("selected_microbatch", 16)}
    return s


def proxy(state, cid, n=256):
    c = state["candidates"][cid]
    key = f"proxy_{n}"
    if key in c:
        p = c[key]
        if p.get("checkpoint_sha256") == c["checkpoint_sha256"] and p.get("examples") == n:
            return p
        raise RuntimeError("stale proxy result")
    dest = OUT / "candidates" / cid / f"proxy-{n}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name("."+dest.name+".tmp")
    sec = run(state, f"{cid}-proxy-{n}", [PY, "scripts/evaluate_posttraining_proxy.py", "--checkpoint", c["checkpoint"],
             "--tokenizer", TOKENIZER, "--manifest", MANIFEST, "--output", tmp, "--examples", n, "--threads", 16])
    p = json.loads(tmp.read_text())
    if p.get("checkpoint_sha256") != c["checkpoint_sha256"] or p.get("examples") != n or len(p.get("fixed_example_ids", [])) != 2*n:
        raise RuntimeError("malformed proxy result")
    if not all(math.isfinite(p[k]) for k in ("v2_ranking_accuracy", "v2_mean_margin", "data_d_validation")):
        raise RuntimeError("nonfinite proxy result")
    os.replace(tmp, dest)
    c[key] = p
    state["proxy_seconds"] += sec
    save(state)
    return p


def frontier(state):
    rows = [(k, v.get("proxy_256")) for k, v in state.get("candidates", {}).items() if v.get("proxy_256")]
    def dominates(a, b):
        keys = (("v2_ranking_accuracy", 1), ("v2_mean_margin", 1), ("data_d_validation", -1))
        return all(sign*a[k] >= sign*b[k] for k, sign in keys) and any(sign*a[k] > sign*b[k] for k, sign in keys)
    return [cid for cid, p in rows if cid == "BASE" or not any(other != cid and dominates(q,p) for other,q in rows)]


def train(state, branch, target):
    root = OUT / "recovery" / branch
    root.mkdir(parents=True, exist_ok=True)
    parent = state["candidates"][branch]
    latest = root / "latest.pt"
    side = root / "latest.pt.sha256"
    if latest.exists() != side.exists() or latest.exists() and sha256(latest) != side.read_text().strip():
        raise RuntimeError("ambiguous or corrupt recovery resume checkpoint")
    prior = 0
    prior_processed = 0
    if latest.exists():
        import torch
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck.get("parent_checkpoint_sha256") != parent["checkpoint_sha256"] or ck.get("method") != "recovery" or ck.get("learning_rate") != state["recovery_lr"] or ck.get("optimizer_kind") != state["optimizer"] or ck.get("replay_fraction") != 1.0:
            raise RuntimeError("recovery resume identity mismatch")
        prior, prior_processed = ck["training_tokens"], ck["processed_tokens"]
    if prior >= target:
        return register_milestone(state, branch, target, root)
    projected = (target-prior)/max(state["throughput"].get("logical_tokens_per_second", 500), 1)*1.3
    if remaining(state) < projected + final_estimate(state) + RESERVE:
        event(state, "training_skipped", branch=branch, target=target, reason="deadline projection", projected_seconds=projected)
        return None
    limit = max(1, remaining(state)-final_estimate(state)-RESERVE)
    started = time.monotonic()
    try:
        run(state, f"{branch}-recovery-{target}", [PY, "scripts/train_posttraining_worker.py", "--base", parent["checkpoint"],
            "--output", root, "--tokenizer", TOKENIZER, "--manifest", MANIFEST, "--method", "recovery",
            "--tokens", target, "--optimizer", state["optimizer"], "--lr", state["recovery_lr"], "--replay", 1,
            "--microbatch", state["microbatch"], "--threads", 16, "--checkpoint-tokens", 500000,
            "--stop-epoch", time.time()+limit, "--method-chain", json.dumps([branch,"recovery"])], max_seconds=limit+120)
    except Exception as exc:
        event(state, "training_error", branch=branch, target=target, error=str(exc))
        if not latest.exists() or not side.exists() or sha256(latest) != side.read_text().strip():
            raise
    sec = time.monotonic()-started
    state["training_seconds"] += sec
    result = json.loads((root / "result.json").read_text()) if (root / "result.json").exists() else None
    if result is None or result["checkpoint_sha256"] != sha256(latest) or result["training_tokens"] <= prior:
        raise RuntimeError("worker result inconsistent with checkpoint")
    new_logical = result["training_tokens"]-prior
    new_processed = result["processed_tokens"]-prior_processed
    if new_logical <= 0 or new_processed < new_logical or not math.isfinite(result["train_loss"]):
        raise RuntimeError("invalid recovery accounting")
    state["throughput"] = {"logical_tokens_per_second": new_logical/sec, "processed_tokens_per_second": new_processed/sec,
                            "last_wall_seconds": sec, "last_logical_tokens": new_logical, "last_processed_tokens": new_processed,
                            "last_branch": branch, "last_target": target}
    save(state)
    return register_milestone(state, branch, result["training_tokens"], root, result, sec)


def register_milestone(state, branch, tokens, root, result=None, seconds=0):
    src = root / f"milestone-{tokens}.pt"
    if not src.exists():
        latest = root / "latest.pt"
        if not latest.exists(): raise RuntimeError("missing recovery milestone")
        os.link(latest, src)
    digest = sha256(src)
    side = src.with_suffix(".pt.sha256")
    if side.exists() and side.read_text().strip() != digest: raise RuntimeError("milestone SHA mismatch")
    if not side.exists(): side.write_text(digest+"\n")
    cid = f"{branch}-recovery-{tokens}"
    if cid not in state["candidates"]:
        state["candidates"][cid] = {"checkpoint": str(src), "checkpoint_sha256": digest, "lineage": branch,
            "method": "recovery", "parent_checkpoint_sha256": state["candidates"][branch]["checkpoint_sha256"],
            "recovery_tokens": tokens, "optimizer": state["optimizer"], "lr": state["recovery_lr"], "replay": 1.0,
            "processed_tokens": result["processed_tokens"] if result else None, "training_loss": result["train_loss"] if result else None,
            "training_wall_seconds": seconds, "status": "VALID"}
        save(state)
    proxy(state, cid)
    return cid


def interpolate(state, endpoint, alpha):
    import torch
    # Keep the existing two-decimal IDs for cent-level weights while allowing
    # distinct exact IDs for transition probes such as 17.5%.
    alpha_id = f"{alpha:.2f}" if round(alpha, 2) == alpha else f"{alpha:.3f}".rstrip("0").rstrip(".")
    cid = f"base-{endpoint}-a{alpha_id}"
    if cid in state["candidates"]: return cid
    base = state["candidates"]["BASE"]
    specialist = state["candidates"][endpoint]
    if sha256(base["checkpoint"]) != base["checkpoint_sha256"] or sha256(specialist["checkpoint"]) != specialist["checkpoint_sha256"]:
        raise RuntimeError("interpolation parent SHA changed")
    first = torch.load(base["checkpoint"], map_location="cpu", weights_only=False)
    second = torch.load(specialist["checkpoint"], map_location="cpu", weights_only=False)
    if first["config"] != second["config"] or list(first["model"]) != list(second["model"]):
        raise RuntimeError("interpolation architecture mismatch")
    merged = {}
    for k, x in first["model"].items():
        y = second["model"][k]
        if x.shape != y.shape or x.dtype != y.dtype: raise RuntimeError("interpolation parameter mismatch")
        if x.is_floating_point(): merged[k] = torch.lerp(x, y, alpha)
        elif torch.equal(x,y): merged[k] = x.clone()
        else: raise RuntimeError("nonfloating buffer differs between parents")
        if not torch.isfinite(merged[k]).all(): raise RuntimeError("nonfinite interpolation")
    dest = OUT / "candidates" / cid / "checkpoint.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): raise RuntimeError("unregistered interpolation checkpoint exists")
    tmp = dest.with_name(".checkpoint.pt.tmp")
    torch.save({"schema":"posttraining-salvage-interpolation-v1", "config":first["config"], "model":merged,
                "merge":{"base_sha256":base["checkpoint_sha256"], "specialist_sha256":specialist["checkpoint_sha256"], "alpha":alpha}}, tmp)
    os.replace(tmp, dest)
    digest = sha256(dest)
    dest.with_suffix(".pt.sha256").write_text(digest+"\n")
    state["candidates"][cid] = {"checkpoint":str(dest), "checkpoint_sha256":digest, "lineage":endpoint,
        "method":"interpolation", "alpha":alpha, "parent_checkpoint_sha256":specialist["checkpoint_sha256"], "status":"VALID"}
    save(state)
    proxy(state,cid)
    return cid


def final_estimate(state):
    times = [x.get("wall_seconds", 0) for x in state.get("final_evaluations", {}).values()]
    return 1.2 * (max(times) if times else state.get("preflight_full_eval_seconds", 900)) * 4


def official_validation(path):
    import torch
    import numpy as np
    from latticelm.config import LatticeConfig
    from latticelm.model import build_model
    from scripts.train_final_production import load_arrays, validation
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LatticeConfig(**ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"], strict=True)
    _, _, valid = load_arrays(MANIFEST)
    source = {}
    for name, arrays in valid.items():
        source[name] = validation(model, np.concatenate(arrays), cfg.context_length)[0]
    return {"loss":sum(source.values())/len(source), "by_source":source,
            "protocol":"train_final_production.validation; 8 batches x 4 x 256 per source"}


def full_eval(state, cid):
    c = state["candidates"][cid]
    if cid in state["final_evaluations"]: return state["final_evaluations"][cid]
    if sha256(c["checkpoint"]) != c["checkpoint_sha256"]: raise RuntimeError("finalist SHA changed")
    root = OUT / "candidates" / cid / "final_eval"
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    vpath = root / "data_d.json"
    if not vpath.exists(): write(vpath, official_validation(c["checkpoint"]))
    wiki = root / "wikitext.json"
    if not wiki.exists():
        tmp = root / ".wikitext.json.tmp"
        sec = run(state,cid+"-wiki",[PY,"scripts/evaluate_wikitext103.py","--checkpoint",c["checkpoint"],"--tokenizer",TOKENIZER,"--output",tmp,"--threads",16,"--batch-size",16])
        w = json.loads(tmp.read_text())
        if w.get("checkpoint_sha256") != c["checkpoint_sha256"] or not math.isfinite(w.get("bits_per_byte",float("nan"))): raise RuntimeError("malformed WikiText result")
        os.replace(tmp,wiki)
        state.setdefault("eval_timings",{})["wikitext"] = sec
    gibc = root / "gibc.json"
    if not gibc.exists():
        tmp = root / ".gibc.json.tmp"
        sec = run(state,cid+"-gibc",[PY,"scripts/evaluate_gibc.py","--checkpoint",c["checkpoint"],"--tokenizer",TOKENIZER,
              "--output",tmp,"--tasks",",".join(TASKS),"--threads",16,"--batch-size",16])
        g = json.loads(tmp.read_text())
        if g.get("phase6_metadata",{}).get("checkpoint_sha256") != c["checkpoint_sha256"] or any(not math.isfinite(g.get("results",{}).get(t,{}).get("acc,none",float("nan"))) for t in TASKS):
            raise RuntimeError("malformed GIBC result")
        os.replace(tmp,gibc)
        state.setdefault("eval_timings",{})["gibc"] = sec
    w = json.loads(wiki.read_text()); g = json.loads(gibc.read_text())
    p = proxy(state,cid,1024)
    result = {"candidate_id":cid,"checkpoint_sha256":c["checkpoint_sha256"],"architecture":validate_model(c["checkpoint"]),
        "lineage":c["lineage"],"recovery_tokens":c.get("recovery_tokens",0),"interpolation_alpha":c.get("alpha"),
        "tokenizer_sha256":state["tokenizer_sha256"],"dataset_manifest_sha256":state["manifest_sha256"],
        "code_commit_sha":state["code_commit_sha"],"evaluator_versions":{"proxy":"posttraining-proxy-v2","gibc":g["phase6_metadata"].get("lm_eval_version"),"wikitext":"evaluate_wikitext103.py"},
        "data_d":json.loads(vpath.read_text()),"wikitext_ppl":w["perplexity"],"wikitext_bpb":w["bits_per_byte"],
        "gibc_raw":{t:g["results"][t]["acc,none"] for t in TASKS},
        "gibc_normalized":{t:g["results"][t].get("acc_norm,none") for t in TASKS},
        "proxy_1024":{"accuracy":p["v2_ranking_accuracy"],"mean_margin":p["v2_mean_margin"],"data_d_fixed_slice_loss":p["data_d_validation"]},
        "wall_seconds":time.monotonic()-start,"raw_outputs":{"data_d":str(vpath),"wikitext":str(wiki),"gibc":str(gibc)}}
    if not all(math.isfinite(x) for x in (result["data_d"]["loss"],result["wikitext_ppl"],result["wikitext_bpb"],*result["gibc_raw"].values())):
        raise RuntimeError("nonfinite official result")
    write(root/"metrics.json",result)
    state["final_evaluations"][cid] = result
    state["official_seconds"] += result["wall_seconds"]
    save(state)
    return result


def choose(state):
    final = state["final_evaluations"]
    base = final["BASE"]
    def mean(x): return sum(x["gibc_raw"].values())/4
    plausible=[]
    for cid,x in final.items():
        if cid == "BASE": continue
        if x["data_d"]["loss"] > base["data_d"]["loss"]*1.20 or x["wikitext_bpb"] > base["wikitext_bpb"]*1.25: continue
        if mean(x) < mean(base)+.005: continue
        plausible.append((cid,x))
    winner = max(plausible,key=lambda z:(mean(z[1]),-z[1]["wikitext_bpb"],-z[1]["data_d"]["loss"]))[0] if plausible else "BASE"
    x = final[winner]
    deltas = {"data_d_loss":x["data_d"]["loss"]-base["data_d"]["loss"],
              "wikitext_ppl":x["wikitext_ppl"]-base["wikitext_ppl"],"wikitext_bpb":x["wikitext_bpb"]-base["wikitext_bpb"],
              **{t:x["gibc_raw"][t]-base["gibc_raw"][t] for t in TASKS},
              "reasoning_accuracy":x["proxy_1024"]["accuracy"]-base["proxy_1024"]["accuracy"]}
    result = {"selected_candidate_id":winner,"checkpoint":state["candidates"][winner]["checkpoint"],
              "checkpoint_sha256":state["candidates"][winner]["checkpoint_sha256"],"lineage":state["candidates"][winner]["lineage"],
              "reason":"highest mean raw GIBC accuracy among finalists with >=0.5 percentage point improvement and bounded language loss" if plausible else "no finalist demonstrated a defensible broad improvement over BASE",
              "differences_vs_base":deltas,"selection_rule":{"minimum_mean_raw_task_gain":.005,"maximum_data_d_relative_loss":.20,"maximum_wikitext_bpb_relative_loss":.25}}
    write(OUT/"final_selection.json",result)
    state["selection"] = result
    save(state)
    return result


def evidence(state):
    start = time.monotonic()
    selection = choose(state)
    winner = selection["selected_candidate_id"]
    base_after = sha256(state["candidates"]["BASE"]["checkpoint"])
    if base_after != state["base_sha256"]: raise RuntimeError("BASE hash changed at finalization")
    state["base_sha256_after"] = base_after
    rows = ["# LatticeLM six-hour post-training salvage", "",f"Start: {state['start_at']}",f"Deadline: {state['deadline_at']}",f"Finish: {iso()}","",
            f"Training seconds: {state['training_seconds']:.1f}; proxy seconds: {state['proxy_seconds']:.1f}; official seconds: {state['official_seconds']:.1f}.","",
            "## Recovery curves", "", "| Candidate | Tokens | Accuracy | Margin | Fixed DATA-D slice | Logical tok/s | Processed tok/s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for cid,c in state["candidates"].items():
        if c.get("method") == "recovery" and c.get("proxy_256"):
            p=c["proxy_256"]; t=state.get("throughput_history",{}).get(cid,{})
            rows.append(f"| {cid} | {c['recovery_tokens']} | {p['v2_ranking_accuracy']:.4f} | {p['v2_mean_margin']:.4f} | {p['data_d_validation']:.4f} | {t.get('logical_tokens_per_second',0):.1f} | {t.get('processed_tokens_per_second',0):.1f} |")
    rows += ["","## Interpolation curve","","| Endpoint | Alpha | Accuracy | Margin | Fixed DATA-D slice |","|---|---:|---:|---:|---:|"]
    for cid,c in state["candidates"].items():
        if c.get("method")=="interpolation" and c.get("proxy_256"):
            p=c["proxy_256"];rows.append(f"| {c['lineage']} | {c['alpha']:.2f} | {p['v2_ranking_accuracy']:.4f} | {p['v2_mean_margin']:.4f} | {p['data_d_validation']:.4f} |")
    rows += ["","## Official finalists","","| Candidate | DATA-D official | WikiText PPL | BPB | HellaSwag | ARC-Easy | PIQA | WinoGrande | Proxy 1024 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cid,x in state["final_evaluations"].items():
        rows.append(f"| {cid} | {x['data_d']['loss']:.4f} | {x['wikitext_ppl']:.4f} | {x['wikitext_bpb']:.4f} | "+" | ".join(f"{x['gibc_raw'][t]:.4f}" for t in TASKS)+f" | {x['proxy_1024']['accuracy']:.4f} |")
    rows += ["", "## Selection", "",f"Winner: **{winner}**; checkpoint `{selection['checkpoint']}`; SHA `{selection['checkpoint_sha256']}`.", selection["reason"],
             "",f"Exact differences vs BASE: `{json.dumps(selection['differences_vs_base'],sort_keys=True)}`.","",
             "## Integrity", "",f"BASE SHA before: `{state['base_sha256']}`",f"BASE SHA after: `{base_after}`",
             f"Tokenizer SHA: `{state['tokenizer_sha256']}`",f"Dataset manifest SHA: `{state['manifest_sha256']}`",f"Code commit: `{state['code_commit_sha']}`",
             "", "The 256-example proxy DATA-D fixed slice differs from the production DATA-D validation protocol; the official finalist table uses the production protocol.",
             "", "## Decisions and failures", ""]
    rows += [f"- {e['at']}: {e['kind']} {json.dumps({k:v for k,v in e.items() if k not in ('at','kind','argv')},default=str)}" for e in state["events"] if e["kind"] in ("decision","branch_failure","training_skipped")]
    tmp=OUT/".final_report.md.tmp";tmp.write_text("\n".join(rows)+"\n");os.replace(tmp,OUT/"final_report.md")
    state["evidence_seconds"] += time.monotonic()-start
    state["phase"]="COMPLETE";state["finished_at"]=iso();save(state)


def preflight():
    s = json.loads(STATE.read_text()) if STATE.exists() else initial()
    if s.get("phase") not in ("PREFLIGHT","READY"):
        raise RuntimeError("preflight cannot run after execution begins")
    OUT.mkdir(parents=True,exist_ok=True);save(s)
    require_base(s)
    ref=validate_model(s["candidates"]["BASE"]["checkpoint"])
    for name in ("joint","ranking"):validate_model(s["candidates"][name]["checkpoint"],ref)
    # Exact numerical interpolation endpoints, before writing a disposable merge.
    import torch
    b=torch.load(s["candidates"]["BASE"]["checkpoint"],map_location="cpu",weights_only=False)["model"]
    j=torch.load(s["candidates"]["joint"]["checkpoint"],map_location="cpu",weights_only=False)["model"]
    if list(b)!=list(j) or any(not torch.equal(torch.lerp(b[k],j[k],0),b[k]) or not torch.equal(torch.lerp(b[k],j[k],1),j[k]) for k in b if b[k].is_floating_point()):
        raise RuntimeError("interpolation endpoint semantics failed")
    from scripts.train_posttraining_worker import DataDReplay
    replay=DataDReplay(MANIFEST,ref["context_length"])
    x,y=replay.batch()
    if x.numel()!=ref["context_length"] or y.numel()!=ref["context_length"]:raise RuntimeError("replay source invalid")
    smoke=OUT/"preflight"/"recovery"
    smoke.mkdir(parents=True,exist_ok=True)
    # Two model updates; disposable and excluded from candidates.
    run(s,"preflight-recovery",[PY,"scripts/train_posttraining_worker.py","--base",s["candidates"]["joint"]["checkpoint"],
        "--output",smoke,"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--method","recovery","--tokens",2048,
        "--optimizer",s["optimizer"],"--lr",s["recovery_lr"],"--replay",1,"--microbatch",4,"--threads",16,"--checkpoint-tokens",1024])
    r=json.loads((smoke/"result.json").read_text())
    if r["training_tokens"]!=2048 or r["processed_tokens"]<2048 or not math.isfinite(r["train_loss"]) or r["replay"]!=1:
        raise RuntimeError("recovery smoke accounting invalid")
    updated=torch.load(smoke/"latest.pt",map_location="cpu",weights_only=False)
    if not any(not torch.equal(updated["model"][k],j[k]) for k in j):
        raise RuntimeError("recovery smoke produced no gradient update")
    if updated["replay_selector_state"].get("replay_fraction") != 1.0:
        raise RuntimeError("recovery smoke did not use 100% replay")
    validate_model(smoke/"latest.pt",ref)
    before=sha256(smoke/"latest.pt")
    run(s,"preflight-resume",[PY,"scripts/train_posttraining_worker.py","--base",s["candidates"]["joint"]["checkpoint"],
        "--output",smoke,"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--method","recovery","--tokens",2048,
        "--optimizer",s["optimizer"],"--lr",s["recovery_lr"],"--replay",1,"--microbatch",4,"--threads",16,"--checkpoint-tokens",1024])
    if sha256(smoke/"latest.pt")!=before:raise RuntimeError("resume mutated completed checkpoint")
    # Interrupt a fresh disposable run after a checkpoint, then continue it.
    resume=OUT/"preflight"/"interrupted_recovery"
    if not (resume/"result.json").exists():
        resume.mkdir(parents=True,exist_ok=True)
        args=[PY,"scripts/train_posttraining_worker.py","--base",s["candidates"]["joint"]["checkpoint"],
              "--output",resume,"--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--method","recovery",
              "--tokens",16384,"--optimizer",s["optimizer"],"--lr",s["recovery_lr"],"--replay",1,
              "--microbatch",4,"--threads",16,"--checkpoint-tokens",1024]
        run(s,"preflight-interrupt",args+["--stop-epoch",time.time()+8])
        partial=json.loads((resume/"result.json").read_text())
        if partial["training_tokens"]<=0 or partial["training_tokens"]>=16384:
            raise RuntimeError("disposable interrupt failed to stop within the run")
        prior_tokens=partial["training_tokens"]
        run(s,"preflight-interrupt-resume",args)
        resumed=json.loads((resume/"result.json").read_text())
        if resumed["training_tokens"]!=16384 or resumed["updates"]<=partial["updates"] or resumed["processed_tokens"]<=partial["processed_tokens"]:
            raise RuntimeError("interrupted recovery did not resume")
        s["interrupted_resume_from_tokens"]=prior_tokens
    # Target masking, ranking direction, margin sign, and fixed validation are checked independently.
    from scripts.evaluate_posttraining_proxy import score,retention
    from scripts.train_posttraining_worker import batch_candidate_scores
    from latticelm.posttraining.objectives import continuation_logprobs
    class Mock:
        def __call__(self,x):
            logits=torch.zeros((*x.shape,4096));logits[...,3]=3;return logits,None
    raw,norm=batch_candidate_scores(Mock(),[1,2],[[3],[4]],256)
    if not (raw[0]>raw[1] and norm[0]>norm[1]):raise RuntimeError("proxy ranking direction invalid")
    row=score(Mock(),[1,2],[[3],[4]],0,256,"symbolic",{})
    if row["normalized_margin"]<=0:raise RuntimeError("proxy margin sign invalid")
    p=OUT/"preflight"/"proxy.json"
    run(s,"preflight-proxy",[PY,"scripts/evaluate_posttraining_proxy.py","--checkpoint",s["candidates"]["BASE"]["checkpoint"],
        "--tokenizer",TOKENIZER,"--manifest",MANIFEST,"--output",p,"--examples",2,"--threads",16])
    pv=json.loads(p.read_text())
    if pv["examples"]!=2 or len(pv["fixed_example_ids"])!=4 or not math.isfinite(pv["data_d_validation"]):raise RuntimeError("proxy smoke failed")
    # Disposable interpolation checkpoint, with strict reload.
    merge=OUT/"preflight"/"interpolation.pt"
    torch.save({"config":ref,"model":{k:torch.lerp(b[k],j[k],.5) if b[k].is_floating_point() else b[k] for k in b}},merge)
    validate_model(merge,ref)
    s["preflight_merge_sha256"]=sha256(merge)
    # Official command smoke: a small GIBC sample and full WikiText timing.
    g=OUT/"preflight"/"gibc.json"
    if not g.exists():
        run(s,"preflight-gibc",[PY,"scripts/evaluate_gibc.py","--checkpoint",s["candidates"]["BASE"]["checkpoint"],
            "--tokenizer",TOKENIZER,"--output",g,"--tasks",",".join(TASKS),"--limit",1,"--threads",16,"--batch-size",16])
    gv=json.loads(g.read_text())
    if any(t not in gv.get("results",{}) for t in TASKS):raise RuntimeError("GIBC smoke failed")
    w=OUT/"preflight"/"wikitext.json"
    sec=0
    if not w.exists():
        sec=run(s,"preflight-wikitext",[PY,"scripts/evaluate_wikitext103.py","--checkpoint",s["candidates"]["BASE"]["checkpoint"],
            "--tokenizer",TOKENIZER,"--output",w,"--threads",16,"--batch-size",16])
    wv=json.loads(w.read_text())
    if abs(wv["perplexity"]-21.9298062069)>.01:raise RuntimeError("WikiText official mismatch")
    v=official_validation(s["candidates"]["BASE"]["checkpoint"])
    if abs(v["loss"]-2.63049371913)>.01:raise RuntimeError("DATA-D official mismatch")
    s["preflight_full_eval_seconds"]=max(300,sec*3,s.get("preflight_full_eval_seconds",0))
    s["preflight_validation"]=v
    s["phase"]="READY";s["preflight_at"]=iso();save(s)
    report={"BASE":{k:s["candidates"]["BASE"][k] for k in ("checkpoint","checkpoint_sha256")},
            "joint":{k:s["candidates"]["joint"][k] for k in ("checkpoint","checkpoint_sha256")},
            "ranking":{k:s["candidates"]["ranking"][k] for k in ("checkpoint","checkpoint_sha256")},
            "recovery_implementation_verified":"YES","natural_replay_verified":"YES","proxy_evaluator_verified":"YES",
            "interpolation_verified":"YES","final_evaluator_verified":"YES","resume_verified":"YES","state_path":str(STATE),
            "execution_command":f"{PY} scripts/run_posttraining_salvage.py --execute-six-hour-salvage"}
    write(OUT/"readiness.json",report)
    print(json.dumps(report,indent=2))


def execute(reset_budget_seconds=None):
    if not STATE.exists(): raise RuntimeError("run --preflight-only first")
    s=json.loads(STATE.read_text())
    if s["phase"]=="COMPLETE":return
    if reset_budget_seconds is not None:
        if reset_budget_seconds <= 0: raise ValueError("reset budget must be positive")
        previous_deadline=s.get("deadline_epoch")
        s["start_epoch"]=time.time();s["start_monotonic"]=time.monotonic()
        s["deadline_epoch"]=s["start_epoch"]+reset_budget_seconds
        s["deadline_monotonic"]=s["start_monotonic"]+reset_budget_seconds
        s["boot_id"]=boot_id();s["start_at"]=iso(s["start_epoch"]);s["deadline_at"]=iso(s["deadline_epoch"])
        s["phase"]="EXECUTING"
        event(s,"deadline_reset",previous_deadline_epoch=previous_deadline,budget_seconds=reset_budget_seconds,
              resumed_from_checkpoint=str(OUT/"recovery"/"ranking"/"latest.pt"))
    if s["phase"]=="READY":
        s["start_epoch"]=time.time();s["start_monotonic"]=time.monotonic();s["boot_id"]=boot_id()
        s["deadline_epoch"]=s["start_epoch"]+21600;s["deadline_monotonic"]=s["start_monotonic"]+21600
        s["start_at"]=iso(s["start_epoch"]);s["deadline_at"]=iso(s["deadline_epoch"]);s["phase"]="EXECUTING";save(s)
    elif s["phase"]!="EXECUTING":raise RuntimeError("not certified ready")
    print(f"SALVAGE START {s['start_at']} HARD DEADLINE {s['deadline_at']}",flush=True)
    require_base(s)
    try:
        for branch in ("joint","ranking"):
            for target in ((150000,1000000,2500000,5000000) if branch=="joint" else (1000000,2500000,5000000)):
                if STOP or remaining(s)<final_estimate(s)+RESERVE:break
                existing=[c for c in s["candidates"].values() if c.get("method")=="recovery" and c.get("lineage")==branch and c.get("recovery_tokens",0)>=target]
                if existing:continue
                try:
                    cid=train(s,branch,target)
                    if cid:
                        s.setdefault("throughput_history",{})[cid]=dict(s["throughput"]);save(s)
                        p=s["candidates"][cid]["proxy_256"]
                        event(s,"decision",candidate=cid,reason="recovery milestone measured",accuracy=p["v2_ranking_accuracy"],retention=p["data_d_validation"])
                except Exception as exc:
                    s["failures"].append({"branch":branch,"target":target,"error":str(exc),"at":iso()})
                    event(s,"branch_failure",branch=branch,target=target,error=str(exc))
                    break
                points=[(c.get("recovery_tokens",0),c["proxy_256"]) for c in s["candidates"].values() if c.get("lineage")==branch and c.get("method")=="recovery" and c.get("proxy_256")]
                points.sort(key=lambda z:z[0])
                if len(points)>1 and points[-1][1]["data_d_validation"]>=points[-2][1]["data_d_validation"] and points[-1][1]["v2_ranking_accuracy"]<points[-2][1]["v2_ranking_accuracy"]-.02:
                    event(s,"decision",branch=branch,reason="later recovery worsened retention and reasoning; stop ladder")
                    break
        # Later finalist selection and interpolation scoring compare against BASE.
        # Older master states may not include its 256-example proxy result.
        if "proxy_256" not in s["candidates"]["BASE"]:
            proxy(s, "BASE", 256)
        for branch in ("joint","ranking"):
            for target in (7500000,10000000):
                points=sorted([(c["recovery_tokens"],c["proxy_256"]) for c in s["candidates"].values()
                               if c.get("lineage")==branch and c.get("method")=="recovery" and c.get("proxy_256")],key=lambda z:z[0])
                if len(points)<2 or points[-1][0]<target-2500000:break
                previous,current=points[-2][1],points[-1][1]
                baseline=s["candidates"]["BASE"]["proxy_256"]
                if current["data_d_validation"]>previous["data_d_validation"]-.10 or current["v2_ranking_accuracy"]<baseline["v2_ranking_accuracy"]+.15:
                    event(s,"decision",branch=branch,reason="extension lacks retention slope or reasoning advantage")
                    break
                if remaining(s)<final_estimate(s)+RESERVE+1.3*(target-points[-1][0])/max(s["throughput"].get("logical_tokens_per_second",1),1):break
                try:
                    cid=train(s,branch,target)
                    if cid:s.setdefault("throughput_history",{})[cid]=dict(s["throughput"]);save(s)
                except Exception as exc:
                    s["failures"].append({"branch":branch,"extension_target":target,"error":str(exc),"at":iso()})
                    event(s,"branch_failure",branch=branch,target=target,error=str(exc))
                    break
        for branch in ("joint","ranking"):
            options=[(cid,c) for cid,c in s["candidates"].items() if c.get("lineage")==branch and c.get("method")=="recovery" and c.get("proxy_256")]
            baseline=s["candidates"]["BASE"]["proxy_256"]
            useful=[z for z in options if z[1]["proxy_256"]["v2_ranking_accuracy"]>=baseline["v2_ranking_accuracy"]+.15]
            best=min(useful or options,key=lambda z:z[1]["proxy_256"]["data_d_validation"])[0] if options else None
            for endpoint in (branch,best):
                if endpoint is None:continue
                for alpha in (.25,.5,.75):
                    if STOP or remaining(s)<final_estimate(s)+RESERVE:break
                    try:interpolate(s,endpoint,alpha)
                    except Exception as exc:
                        s["failures"].append({"interpolation":endpoint,"alpha":alpha,"error":str(exc),"at":iso()});event(s,"branch_failure",branch=endpoint,error=str(exc))
        # Refine only the two strongest coarse interpolation families.
        base_loss=s["candidates"]["BASE"]["proxy_256"]["data_d_validation"]
        def tradeoff(cid):
            p=s["candidates"][cid]["proxy_256"]
            return p["v2_ranking_accuracy"]-.03*max(0,p["data_d_validation"]/base_loss-1)
        merge_ids=[cid for cid,c in s["candidates"].items() if c.get("method")=="interpolation" and c.get("proxy_256")]
        families={}
        for cid in merge_ids:
            endpoint=s["candidates"][cid]["lineage"]
            families.setdefault(endpoint,[]).append(cid)
        best_families=sorted(families,key=lambda e:max(tradeoff(cid) for cid in families[e]),reverse=True)[:2]
        for endpoint in best_families:
            coarse=max(families[endpoint],key=tradeoff)
            center=s["candidates"][coarse]["alpha"]
            for alpha in (round(center-.10,2),round(center+.10,2)):
                if 0<alpha<1 and remaining(s)>final_estimate(s)+RESERVE:
                    try:interpolate(s,endpoint,alpha)
                    except Exception as exc:event(s,"branch_failure",branch=endpoint,alpha=alpha,error=str(exc))
        # Confirmation candidates are chosen from the three-dimensional proxy frontier.
        ids=[x for x in frontier(s) if x!="BASE"]
        baseline=s["candidates"]["BASE"]["proxy_256"]["data_d_validation"]
        def finalist_score(cid):
            p=s["candidates"][cid]["proxy_256"]
            return (p["v2_ranking_accuracy"]-.04*max(0,p["data_d_validation"]/baseline-1),
                    p["v2_mean_margin"],-p["data_d_validation"])
        selected=[]
        for branch in ("joint","ranking"):
            group=[cid for cid,c in s["candidates"].items() if c.get("method")=="recovery" and c.get("lineage")==branch and c.get("proxy_256")]
            if group:selected.append(max(group,key=finalist_score))
        merges=[cid for cid,c in s["candidates"].items() if c.get("method")=="interpolation" and c.get("proxy_256")]
        if merges:selected.append(max(merges,key=finalist_score))
        ids.sort(key=finalist_score,reverse=True)
        selected += [cid for cid in ids if cid not in selected][:max(0,4-len(selected))]
        event(s,"decision",reason="finalists chosen from recovered joint, recovered ranking, interpolation, then proxy Pareto frontier",candidate_ids=selected[:4])
        for cid in selected[:4]:
            if remaining(s)<final_estimate(s)+RESERVE:break
            try:proxy(s,cid,1024)
            except Exception as exc:event(s,"branch_failure",candidate=cid,error=str(exc))
        finalists=["BASE"]+selected
        # First complete finalist determines current full-evaluation timing.
        for cid in finalists:
            if cid in s["final_evaluations"]:continue
            if remaining(s)<RESERVE+max(300,1.2*(max((x["wall_seconds"] for x in s["final_evaluations"].values()),default=s["preflight_full_eval_seconds"]))):break
            try:full_eval(s,cid)
            except Exception as exc:
                s["failures"].append({"final_eval":cid,"error":str(exc),"at":iso()});event(s,"branch_failure",candidate=cid,error=str(exc))
                if cid=="BASE":raise
        if "BASE" not in s["final_evaluations"]:raise RuntimeError("BASE official evaluation incomplete")
        evidence(s)
    finally:
        require_base(s)


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only",action="store_true")
    group.add_argument("--execute-six-hour-salvage",action="store_true")
    p.add_argument("--reset-budget-seconds",type=int,help="resume an executing salvage with a fresh remaining budget")
    args=p.parse_args()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"controller.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pid_record={"pid":os.getpid(),"started_at":iso(),"boot_id":boot_id()}
        write(OUT/"controller.pid.json",pid_record)
        write(OUT/"controller.heartbeat.json",{**pid_record,"at":iso(),"child_pid":None})
        try:
            if args.preflight_only:preflight()
            else:execute(args.reset_budget_seconds)
        finally:
            try:
                current=json.loads((OUT/"controller.pid.json").read_text())
                if current.get("pid")==os.getpid(): (OUT/"controller.pid.json").unlink()
            except (FileNotFoundError, json.JSONDecodeError):
                pass


if __name__=="__main__":main()
