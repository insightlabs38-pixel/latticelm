#!/usr/bin/env python3
"""Cheap held-out v2 + certified DATA-D validation pruning metrics."""
import argparse,json,math
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.tokenizer import load_tokenizer
from latticelm.lattice_reason_v2 import generate_example
from scripts.train_posttraining_worker import candidate_score
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--examples",type=int,default=128);p.add_argument("--threads",type=int,default=16);a=p.parse_args();torch.set_num_threads(a.threads);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);cfg=LatticeConfig(**ck["config"]);model=build_model(cfg);model.load_state_dict(ck["model"]);model.eval();tok=load_tokenizer(a.tokenizer);top=json.loads(a.manifest.read_text());base=a.manifest.parent;losses=[]
 with torch.inference_mode():
  for item in top["validation_shards"]:
   meta=json.loads((base/item["manifest_path"]).read_text());array=np.memmap(base/meta["path"],dtype="<i4",mode="r")
   for offset in (0,cfg.context_length+1):
    z=torch.tensor(np.asarray(array[offset:offset+cfg.context_length+1]).astype(np.int64));logits=model(z[:-1][None,:])[0];losses.append(float(F.cross_entropy(logits.reshape(-1,cfg.vocab_size),z[1:])))
 correct=0;margins=[];lengths=[]
 for i in range(a.examples):
  ex=generate_example(i,split="validation");scores=[]
  for candidate in ex.candidates:scores.append(float(candidate_score(model,tok,ex.prompt,candidate,cfg.context_length)[1]))
  correct+=scores.index(max(scores))==ex.candidate_position;margins.append(scores[ex.candidate_position]-max(x for j,x in enumerate(scores) if j!=ex.candidate_position));lengths.append(len(tok.encode(ex.answer)))
 mean_x=sum(lengths)/len(lengths);mean_y=sum(margins)/len(margins);cov=sum((x-mean_x)*(y-mean_y) for x,y in zip(lengths,margins));den=math.sqrt(sum((x-mean_x)**2 for x in lengths)*sum((y-mean_y)**2 for y in margins));result={"schema":"posttraining-proxy-v1","data_d_validation":sum(losses)/len(losses),"v2_ranking_accuracy":correct/a.examples,"v2_mean_margin":sum(margins)/len(margins),"margin_answer_length_correlation":cov/den if den else 0.,"examples":a.examples};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
