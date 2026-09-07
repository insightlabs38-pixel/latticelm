"""Bounded end-to-end Co4-L CPU backend benchmark."""
from __future__ import annotations
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from latticelm.config import LatticeConfig
from latticelm.model import build_model

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/seed2026_co4_l.json"

def batch(step:int,device="cpu"):
 g=torch.Generator().manual_seed(84001+step)
 x=torch.randint(0,4096,(8,128),generator=g,device=device);return x,torch.roll(x,-1,1)

def run(backend:str,warmup:int,steps:int)->dict:
 torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.manual_seed(7719)
 cfg=LatticeConfig.from_json(CONFIG);model=build_model(cfg);opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,betas=(cfg.adam_beta1,cfg.adam_beta2),weight_decay=cfg.weight_decay)
 use_bf16="bf16" in backend;use_compile="compile" in backend;compile_seconds=0.0
 if use_compile:
  started=time.perf_counter();model=torch.compile(model,mode="max-autotune-no-cudagraphs",fullgraph=False);compile_seconds=time.perf_counter()-started
 losses=[];grad_finite=True;timings=[]
 def step(i):
  nonlocal grad_finite
  x,y=batch(i);opt.zero_grad(set_to_none=True);started=time.perf_counter()
  with torch.autocast("cpu",dtype=torch.bfloat16,enabled=use_bf16):loss=model(x,y)[1]
  loss.backward();grad_finite=grad_finite and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
  torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();return float(loss),time.perf_counter()-started

 started=time.perf_counter()
 try:
  for i in range(warmup):loss,elapsed=step(i);losses.append(loss)
  warmup_seconds=time.perf_counter()-started
  for i in range(warmup,warmup+steps):loss,elapsed=step(i);losses.append(loss);timings.append(elapsed)
  counters={}
  if use_compile:
   from torch._dynamo.utils import counters as dc
   counters={str(k):dict(v) for k,v in dc.items() if v}
  mean=sum(timings)/len(timings)
  return {"backend":backend,"status":"PASS","tokens_per_second":8*128/mean,"steady_seconds":sum(timings),"warmup_seconds":warmup_seconds,"compile_wrapper_seconds":compile_seconds,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"losses":losses,"finite_gradients":grad_finite,"loss_sanity":all(np.isfinite(losses)) and losses[-1]<losses[0]*1.25,"compile_counters":counters,"parameter_count":sum(p.numel() for p in model.parameters()),"dtype_policy":"CPU autocast BF16; parameters and AdamW state FP32" if use_bf16 else "FP32"}
 except Exception as exc:
  return {"backend":backend,"status":"FAIL","error":f"{type(exc).__name__}: {exc}","tokens_per_second":0,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}

def main():
 p=argparse.ArgumentParser();p.add_argument("--backend",choices=("fp32_eager","fp32_compile","bf16_eager","bf16_compile"),required=True);p.add_argument("--warmup",type=int,default=5);p.add_argument("--steps",type=int,default=20);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 result=run(a.backend,a.warmup,a.steps);a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2)+"\n");tmp.replace(a.output);print(json.dumps(result));return 0 if result["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
