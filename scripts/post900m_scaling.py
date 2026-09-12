"""Deterministic, interpretable scaling analysis for the 32M final lineage."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np

PARAMETERS=32_678_640
LOSS_FIELDS=("data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss")

def _fit_power(rows,field,start=0):
    pts=[(float(r["nominal_tokens"]),float(r[field])) for r in rows if int(r["nominal_tokens"])>=start and r.get(field) not in (None,"")]
    if len(pts)<4:return None
    x=np.log(np.array([p[0] for p in pts]));y=np.array([p[1] for p in pts])
    # A three-parameter asymptotic power law, with a small deterministic grid for L_inf.
    lo=max(0.,min(y)-.5);hi=min(y)-1e-5;best=None
    for floor in np.linspace(lo,hi,300):
        z=y-floor
        if np.any(z<=0):continue
        slope,intercept=np.polyfit(x,np.log(z),1);pred=floor+np.exp(intercept+slope*x)
        sse=float(np.square(pred-y).sum())
        if best is None or sse<best[0]:best=(sse,float(floor),float(intercept),float(slope))
    if best is None:return None
    sse,floor,intercept,slope=best
    predict=lambda t:floor+math.exp(intercept+slope*math.log(t))
    return {"form":"loss=floor+a*tokens^b","start_tokens":start,"points":len(pts),"floor":floor,"a":math.exp(intercept),"b":slope,"rmse":math.sqrt(sse/len(pts)),"predicted_1_5b":predict(1_500_000_000),"predicted_2_05b":predict(2_050_000_000)}

def analyze(curve:Path,wiki_by_tokens:dict|None=None):
    rows=list(csv.DictReader(curve.open()));rows=sorted(rows,key=lambda r:int(r["nominal_tokens"]))
    rows=[r for r in rows if int(r["nominal_tokens"])<=900_000_000]
    if not rows or int(rows[-1]["nominal_tokens"])!=900_000_000:raise ValueError("complete 900M trajectory absent")
    models={}
    for field in LOSS_FIELDS:
        models[field]={"all":_fit_power(rows,field,50_000_000),"late":_fit_power(rows,field,300_000_000)}
    windows={}
    for n in (2,3,5):
        if len(rows)>=n:
            a,b=rows[-n],rows[-1];dt=int(b["nominal_tokens"])-int(a["nominal_tokens"])
            windows[str(n)]={f:{"absolute_improvement":float(a[f])-float(b[f]),"relative_improvement":(float(a[f])-float(b[f]))/float(a[f]),"per_100m":(float(a[f])-float(b[f]))*100_000_000/dt} for f in LOSS_FIELDS}
    source_regression=max(float(rows[-1][f])-min(float(x[f]) for x in rows[-5:]) for f in LOSS_FIELDS[1:])
    wiki=[]
    for token,value in sorted((wiki_by_tokens or {}).items()):
        if value:wiki.append({"tokens":int(token),"perplexity":float(value["perplexity"]),"bits_per_byte":float(value["bits_per_byte"])})
    result={"schema":"post900m-scaling-v1","parameters":PARAMETERS,"tokens_per_parameter":900_000_000/PARAMETERS,"trajectory_points":len(rows),"windows":windows,"fits":models,"source_specific_max_recent_regression":source_regression,"wikitext":wiki}
    vals=[x for x in (models["data_d_validation_loss"]["all"],models["data_d_validation_loss"]["late"]) if x]
    gains=[float(rows[-1]["data_d_validation_loss"])-x["predicted_1_5b"] for x in vals]
    result["projected_900m_to_1_5b_gain"]={"estimates":gains,"min":min(gains),"max":max(gains)} if gains else None
    result["projected_900m_to_2_05b_gain"]=[float(rows[-1]["data_d_validation_loss"])-x["predicted_2_05b"] for x in vals]
    return result

def classify(a,integrity=True,numerical=True):
    if not integrity or not numerical:return "BLOCKED"
    recent=a["windows"].get("3",{}).get("data_d_validation_loss",{})
    gain=a.get("projected_900m_to_1_5b_gain")
    wiki=a.get("wikitext",[])
    wiki_ok=len(wiki)<2 or wiki[-1]["bits_per_byte"]<=wiki[-2]["bits_per_byte"]+.005
    source_ok=a["source_specific_max_recent_regression"]<=.02
    if not gain or not gain["estimates"]:return "INCONCLUSIVE"
    disagreement=gain["max"]-gain["min"]
    if recent.get("absolute_improvement",0)>0.008 and gain["min"]>=0.012 and wiki_ok and source_ok:return "CONTINUE_TO_1_5B"
    if recent.get("absolute_improvement",0)<=0.004 and gain["max"]<0.012 and wiki_ok and source_ok:return "SATURATING_OR_LOW_VALUE"
    if disagreement>.02 or not wiki_ok or not source_ok:return "INCONCLUSIVE"
    return "INCONCLUSIVE"

def write(curve:Path,wiki:dict,out:Path):
    result=analyze(curve,wiki);result["classification"]=classify(result);out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result
