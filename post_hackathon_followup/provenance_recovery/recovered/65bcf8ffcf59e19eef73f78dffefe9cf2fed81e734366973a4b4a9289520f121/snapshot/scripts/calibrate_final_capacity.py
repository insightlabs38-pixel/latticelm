"""Bounded 32M full-step CPU calibration: two thread counts x two microbatches."""
from __future__ import annotations
import argparse,json,resource,time
from pathlib import Path
import torch
import torch.nn.functional as F
from latticelm.config import LatticeConfig
from latticelm.model import build_model
ROOT=Path(__file__).resolve().parents[1]
def one(threads,microbatch,backend,warmup=3,steps=10):
 torch.set_num_threads(threads);torch.manual_seed(773);cfg=LatticeConfig.from_json(ROOT/"configs/final_capacity32_co4.json");model=build_model(cfg)
 if model.parameter_breakdown()["total"]!=32_678_640:raise RuntimeError("parameter audit")
 if backend=="fp32_compile":model.compile(mode="max-autotune-no-cudagraphs",fullgraph=False)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay);g=torch.Generator().manual_seed(991);timings=[];losses=[];start=time.perf_counter()
 for step in range(warmup+steps):
  x=torch.randint(0,4096,(8,128),generator=g);y=torch.roll(x,-1,1);opt.zero_grad(set_to_none=True);t=time.perf_counter();loss=torch.zeros(())
  for off in range(0,8,microbatch):
   _,part=model(x[off:off+microbatch],y[off:off+microbatch]);(part*microbatch/8).backward();loss+=part.detach()*microbatch/8
  torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss));
  if step>=warmup:timings.append(time.perf_counter()-t)
 warm=time.perf_counter()-start-sum(timings);mean=sum(timings)/len(timings);return {"threads":threads,"microbatch":microbatch,"backend":backend,"tokens_per_second":1024/mean,"warmup_compile_seconds":warm,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"finite":all(map(torch.isfinite,[torch.tensor(x) for x in losses])),"losses":losses}
def main():
 p=argparse.ArgumentParser();p.add_argument("--backend",choices=("fp32_eager","fp32_compile"),required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=[]
 for threads in (12,16):
  for micro in (4,8):rows.append(one(threads,micro,a.backend))
 viable=[x for x in rows if x["finite"]];best=max(viable,key=lambda x:x["tokens_per_second"]);result={"rows":rows,"selected":best,"parameter_count":32_678_640,"status":"PASS"};a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
