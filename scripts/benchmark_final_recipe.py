"""Short synthetic end-to-end batch/compile benchmark for one candidate."""
from __future__ import annotations
import argparse,json,resource,time
from dataclasses import replace
from pathlib import Path
import torch
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from latticelm.final_recipe import make_optimizer

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--batch",type=int,required=True);p.add_argument("--backend",choices=("eager","compile"),required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--threads",type=int,default=16);p.add_argument("--steps",type=int,default=6);a=p.parse_args()
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);cfg=replace(LatticeConfig.from_json(a.config),batch_size=a.batch);torch.manual_seed(cfg.seed);model=build_model(cfg)
 if a.backend=="compile":model.compile(mode="max-autotune-no-cudagraphs",fullgraph=False)
 opt,_=make_optimizer(model,cfg.optimizer,cfg.learning_rate,cfg.weight_decay,(cfg.adam_beta1,cfg.adam_beta2));x=torch.randint(0,cfg.vocab_size,(a.batch,cfg.context_length));y=torch.randint(0,cfg.vocab_size,(a.batch,cfg.context_length));times=[];losses=[]
 total_start=time.perf_counter()
 for step in range(a.steps):
  start=time.perf_counter();opt.zero_grad(set_to_none=True);_,loss=model(x,y);loss.backward();opt.step();times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
 steady=sum(times[2:]);tokens=(a.steps-2)*a.batch*cfg.context_length
 out={"status":"PASS","batch":a.batch,"backend":a.backend,"optimizer":cfg.optimizer,"context":cfg.context_length,"steps":a.steps,"compile_warmup_seconds":sum(times[:2]),"steady_step_seconds":steady/(a.steps-2),"tokens_per_second":tokens/steady,"loss":losses[-1],"optimizer_step_included":True,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"total_seconds":time.perf_counter()-total_start}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out))
if __name__=="__main__":main()
