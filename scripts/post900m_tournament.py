"""Pure, deterministic promotion policy for the post-900M tournament."""
from __future__ import annotations

TASKS=("hellaswag","arc_easy","piqa","winogrande")

def official_score(m:dict)->float:
    return sum(float(m[t]) for t in TASKS)/len(TASKS)

def compare(base:dict,candidate:dict)->dict:
    deltas={k:float(candidate[k])-float(base[k]) for k in TASKS}
    ppl_change=float(candidate["wikitext_ppl"])/float(base["wikitext_ppl"])-1
    bpb_change=float(candidate["wikitext_bpb"])-float(base["wikitext_bpb"])
    validation_change=float(candidate["validation_loss"])-float(base["validation_loss"])
    return {"task_deltas":deltas,"mean_task_delta":sum(deltas.values())/len(deltas),
            "wikitext_ppl_relative_change":ppl_change,"wikitext_bpb_change":bpb_change,
            "validation_loss_change":validation_change}

def promote_cooldown(base:dict,candidates:list[dict])->tuple[dict|None,list[dict]]:
    evidence=[]
    for candidate in candidates:
        delta=compare(base,candidate); clear=(delta["validation_loss_change"]<=-.005 and
            delta["wikitext_bpb_change"]<=-.003 and delta["mean_task_delta"]>=-.003 and
            sum(v<-.015 for v in delta["task_deltas"].values())==0)
        evidence.append({"candidate":candidate["candidate_id"],"comparison":delta,"promotable":clear})
    valid=[c for c,e in zip(candidates,evidence) if e["promotable"]]
    return (min(valid,key=lambda x:(x["wikitext_bpb"],x["validation_loss"])) if valid else None),evidence

def promote_system(base:dict,candidates:list[dict])->tuple[dict|None,list[dict]]:
    evidence=[]
    for c in candidates:
        gain=float(c["tokens_per_second"])/float(base["tokens_per_second"])-1
        ok=gain>=.08 and bool(c.get("finite",False)) and bool(c.get("restart_verified",False)) and float(c.get("matched_loss_delta",9))<=.01
        evidence.append({"candidate":c["candidate_id"],"throughput_gain":gain,"promotable":ok})
    valid=[c for c,e in zip(candidates,evidence) if e["promotable"]]
    return (max(valid,key=lambda x:x["tokens_per_second"]) if valid else None),evidence

def promote_sft(base:dict,candidates:list[dict])->tuple[dict|None,list[dict]]:
    evidence=[]
    base_reason=float(base.get("reasoning_accuracy",0))
    for c in candidates:
        delta=compare(base,c); reason=float(c["reasoning_accuracy"])-base_reason
        ok=reason>=.03 and delta["wikitext_ppl_relative_change"]<=.05 and delta["wikitext_bpb_change"]<=.03 and delta["mean_task_delta"]>=-.005 and sum(v<-.02 for v in delta["task_deltas"].values())==0
        evidence.append({"candidate":c["candidate_id"],"reasoning_gain":reason,"comparison":delta,"promotable":ok})
    valid=[c for c,e in zip(candidates,evidence) if e["promotable"]]
    return (max(valid,key=lambda x:(x["reasoning_accuracy"],official_score(x),-x["wikitext_bpb"])) if valid else None),evidence

def promote_rl(base:dict,candidates:list[dict])->tuple[dict|None,list[dict]]:
    # RL candidates face the same retention gate; reward alone is never a promotion metric.
    return promote_sft(base,candidates)
