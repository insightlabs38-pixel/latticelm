#!/usr/bin/env python3
"""Deterministic held-out symbolic/natural proxy and paired comparison."""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.tokenizer import load_tokenizer
from latticelm.lattice_reason_v2 import generate_example
from latticelm.lattice_reason_v2.natural import generate_from_tokens,verify
from scripts.train_posttraining_worker import batch_candidate_scores

def summary(rows):
 if not rows:return {"examples":0}
 out={"examples":len(rows)}
 for mode in ("raw","normalized"):
  margins=np.asarray([r[mode+"_margin"] for r in rows]);lengths=np.asarray([r["answer_length"] for r in rows]);corr=float(np.corrcoef(lengths,margins)[0,1]) if np.std(lengths) and np.std(margins) else 0.
  out[mode]={"accuracy":float(np.mean(margins>0)),"mean_margin":float(np.mean(margins)),"median_margin":float(np.median(margins)),"margin_quantiles":{str(q):float(np.quantile(margins,q)) for q in (.05,.25,.5,.75,.95)},"answer_length_correlation":corr}
 return out
def load_model(path):
 ck=torch.load(path,map_location="cpu",weights_only=False);cfg=LatticeConfig(**ck["config"]);model=build_model(cfg);model.load_state_dict(ck["model"],strict=True);model.eval();return model,cfg
def score(model,prompt,candidates,correct,context,kind,extra):
 with torch.inference_mode():raw,norm=batch_candidate_scores(model,prompt,candidates,context)
 row={"kind":kind,"answer_length":len(candidates[correct]),**extra}
 for name,values in (("raw",raw.tolist()),("normalized",norm.tolist())):row[name+"_margin"]=values[correct]-max(v for j,v in enumerate(values) if j!=correct)
 return row
def evaluate(model,cfg,tok,arrays,n):
 rows=[]
 for i in range(n):
  ex=generate_example(i,split="validation");rows.append(score(model,tok.encode(ex.prompt),[tok.encode(" "+c) for c in ex.candidates],ex.candidate_position,cfg.context_length,"symbolic",{"example_id":ex.global_id,"skills":list(ex.skills),"difficulty":ex.composition_depth,"surface":ex.surface_grammar,"answer_format":ex.answer_format}))
  array=arrays[i%len(arrays)];natural=generate_from_tokens(array,i+10_000_000,source_id=f"data-d-v4-train-shard-{i%len(arrays)}")
  if not verify(natural,array):raise RuntimeError("natural provenance mismatch")
  rows.append(score(model,natural.prompt_ids,natural.candidates,natural.candidates.index(natural.answer_ids),cfg.context_length,"natural",{"example_id":natural.global_id,"transformation":natural.transformation}))
 return rows
def retention(model,cfg,arrays):
 losses=[]
 with torch.inference_mode():
  for array in arrays:
   for offset in (0,cfg.context_length+1):
    z=torch.tensor(np.asarray(array[offset:offset+cfg.context_length+1]).astype(np.int64));logits=model(z[:-1][None,:])[0];losses.append(float(F.cross_entropy(logits.reshape(-1,cfg.vocab_size),z[1:])))
 return float(np.mean(losses))
def report(rows,parent_rows=None):
 out={"symbolic":summary([r for r in rows if r["kind"]=="symbolic"]),"natural":summary([r for r in rows if r["kind"]=="natural"]),"subsets":{},"fixed_example_ids":[r["example_id"] for r in rows]};groups=defaultdict(list)
 for r in rows:
  if r["kind"]=="symbolic":
   for skill in r["skills"]:groups["skill:"+skill].append(r)
   for key in ("difficulty","surface","answer_format"):groups[key+":"+str(r[key])].append(r)
  else:groups["natural:"+r["transformation"]].append(r)
  groups["answer_length:"+str(min(r["answer_length"],8))].append(r)
 out["subsets"]={k:summary(v) for k,v in groups.items()}
 if parent_rows:
  out["paired_vs_parent"]={}
  for mode in ("raw","normalized"):
   deltas=np.asarray([r[mode+"_margin"]-p[mode+"_margin"] for r,p in zip(rows,parent_rows)]);se=float(np.std(deltas,ddof=1)/math.sqrt(len(deltas))) if len(deltas)>1 else 0.;mean=float(np.mean(deltas))
   out["paired_vs_parent"][mode]={"mean_margin_change":mean,"standard_error":se,"approximate_95pct_interval":[mean-1.96*se,mean+1.96*se],"fraction_toward_correct":float(np.mean(deltas>0)),"negative_to_positive":sum(p[mode+"_margin"]<=0<r[mode+"_margin"] for r,p in zip(rows,parent_rows)),"positive_to_negative":sum(r[mode+"_margin"]<=0<p[mode+"_margin"] for r,p in zip(rows,parent_rows))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--parent",type=Path);p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--examples",type=int,default=256);p.add_argument("--threads",type=int,default=16);a=p.parse_args();torch.set_num_threads(a.threads)
 model,cfg=load_model(a.checkpoint);tok=load_tokenizer(a.tokenizer);top=json.loads(a.manifest.read_text());base=a.manifest.parent;train=[];validation=[]
 for kind,dest in (("shards",train),("validation_shards",validation)):
  for item in top[kind]:
   meta=json.loads((base/item["manifest_path"]).read_text());dest.append(np.memmap(base/meta["path"],dtype="<i4",mode="r"))
 rows=evaluate(model,cfg,tok,train,a.examples);parent_rows=None
 if a.parent:
  parent,parent_cfg=load_model(a.parent)
  if parent_cfg.to_dict()!=cfg.to_dict():raise RuntimeError("paired proxy architecture mismatch")
  parent_rows=evaluate(parent,cfg,tok,train,a.examples)
 subsets=report(rows,parent_rows);result={"schema":"posttraining-proxy-v2","examples":a.examples,"data_d_validation":retention(model,cfg,validation),**subsets,"v2_ranking_accuracy":subsets["symbolic"]["normalized"]["accuracy"],"v2_mean_margin":subsets["symbolic"]["normalized"]["mean_margin"]}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:v for k,v in result.items() if k not in ("fixed_example_ids","subsets")}))
if __name__=="__main__":main()
