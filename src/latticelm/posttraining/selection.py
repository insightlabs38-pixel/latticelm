"""Transparent Pareto selection with BASE as a permanent candidate."""
TASKS=("hellaswag","arc_easy","piqa","winogrande")
def dominates(a,b):
 higher=TASKS;lower=("wikitext_bpb","data_d_validation")
 return all(a[k]>=b[k] for k in higher) and all(a[k]<=b[k] for k in lower) and any(a[k]>b[k] for k in higher) or (all(a[k]>=b[k] for k in higher) and all(a[k]<=b[k] for k in lower) and any(a[k]<b[k] for k in lower))
def select(candidates,wiki_limit=.15,data_limit=.10,min_mean_gain=.005):
 by={x["candidate_id"]:x for x in candidates};base=by["BASE"];eligible=[];rejected=[]
 for x in candidates:
  reasons=[]
  if not x.get("identity_pass",False):reasons.append("identity/config/hash failure")
  if x.get("parameter_count",50_000_000)>=50_000_000:reasons.append("parameter cap")
  if x["wikitext_bpb"]>base["wikitext_bpb"]*(1+wiki_limit):reasons.append("severe WikiText degradation")
  if x["data_d_validation"]>base["data_d_validation"]*(1+data_limit):reasons.append("severe DATA-D degradation")
  (rejected if reasons else eligible).append({"candidate":x,"reasons":reasons} if reasons else x)
 frontier=[x for x in eligible if not any(y is not x and dominates(y,x) for y in eligible)]
 def mean(x):return sum(x[t] for t in TASKS)/len(TASKS)
 improved=[x for x in frontier if x["candidate_id"]=="BASE" or dominates(x,base) or mean(x)>=mean(base)+min_mean_gain]
 winner=max(improved,key=lambda x:(mean(x),-x["wikitext_bpb"],-x["data_d_validation"])) if improved else base
 return {"selected":winner,"pareto_frontier":[x["candidate_id"] for x in frontier],"rejected":rejected,"rule":{"wiki_relative_limit":wiki_limit,"data_relative_limit":data_limit,"minimum_mean_task_gain":min_mean_gain,"primary":"mean raw accuracy","tie_breakers":["WikiText BPB","DATA-D validation"]}}
