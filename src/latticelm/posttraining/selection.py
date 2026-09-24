"""Transparent Pareto selection with BASE as a permanent candidate."""
import math
TASKS=("hellaswag","arc_easy","piqa","winogrande")
def dominates(a,b):
 higher=TASKS;lower=("wikitext_bpb","data_d_validation")
 return all(a[k]>=b[k] for k in higher) and all(a[k]<=b[k] for k in lower) and (any(a[k]>b[k] for k in higher) or any(a[k]<b[k] for k in lower))
def proxy_shortlist(candidates,limit=8):
 """Keep BASE and the best available member of each method family."""
 valid={k:v for k,v in candidates.items() if k=="BASE" or v.get("status") in ("COMPLETE","SAFE_STOPPED_VALID","TRAINED")}
 def rank(item):
  p=item[1].get("proxy") or {};return (p.get("v2_ranking_accuracy",-1),p.get("v2_mean_margin",-1e9),-p.get("data_d_validation",1e9))
 groups={}
 for item in valid.items():
  cid,c=item
  if cid=="BASE":continue
  family=c.get("method","unknown")
  if family.startswith("rlvr"):family="rlvr"
  if family=="interpolation":family="merge"
  if family not in groups or rank(item)>rank(groups[family]):groups[family]=item
 chosen=["BASE"]+[cid for cid,_ in sorted(groups.values(),key=rank,reverse=True)[:max(0,limit-1)]]
 for cid,_ in sorted(valid.items(),key=rank,reverse=True):
  if len(chosen)>=limit:break
  if cid not in chosen:chosen.append(cid)
 return chosen
def select(candidates,wiki_limit=.25,data_limit=.20,min_mean_gain=.005):
 by={x["candidate_id"]:x for x in candidates};base=by["BASE"];eligible=[];rejected=[]
 for x in candidates:
  reasons=[]
  if not x.get("identity_pass",False):reasons.append("identity/config/hash failure")
  if x.get("parameter_count",50_000_000)>=50_000_000:reasons.append("parameter cap")
  if not all(math.isfinite(float(x.get(k,float("nan")))) for k in (*TASKS,"wikitext_bpb","data_d_validation")):reasons.append("nonfinite metric")
  elif x["wikitext_bpb"]>base["wikitext_bpb"]*(1+wiki_limit):reasons.append("severe WikiText degradation")
  elif x["data_d_validation"]>base["data_d_validation"]*(1+data_limit):reasons.append("severe DATA-D degradation")
  (rejected if reasons else eligible).append({"candidate":x,"reasons":reasons} if reasons else x)
 frontier=[x for x in eligible if not any(y is not x and dominates(y,x) for y in eligible)]
 def mean(x):return sum(x[t] for t in TASKS)/len(TASKS)
 improved=[x for x in frontier if x["candidate_id"]=="BASE" or dominates(x,base) or mean(x)>=mean(base)+min_mean_gain]
 winner=max(improved,key=lambda x:(mean(x),-x["wikitext_bpb"],-x["data_d_validation"])) if improved else base
 return {"selected":winner,"pareto_frontier":[x["candidate_id"] for x in frontier],"rejected":rejected,"rule":{"wiki_relative_limit":wiki_limit,"data_relative_limit":data_limit,"minimum_mean_task_gain":min_mean_gain,"primary":"mean raw accuracy","tie_breakers":["WikiText BPB","DATA-D validation"]}}
