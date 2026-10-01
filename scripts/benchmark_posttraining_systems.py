#!/usr/bin/env python3
"""Measure completion-batch throughput on the frozen BASE after handoff."""
import argparse,json,resource,signal,time
from pathlib import Path
import torch
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from scripts.train_posttraining_worker import batch_completion

COMPILE_MODE="max-autotune-no-cudagraphs"
COMPILE_WARMUP_TIMEOUT_SECONDS=120
MIN_COMPILE_SPEEDUP=1.10
MAX_COMPILE_BREAK_EVEN_TOKENS=1_000_000

def benchmark(model,context,batch,repeats=3):
 rows=[]
 for i in range(batch):
  ids=torch.randint(0,model.config.vocab_size,(context+1,)).tolist();rows.append((ids[:-1],ids[1:],[False]*(context//2)+[True]*(context-context//2)))
 times=[];losses=[]
 for _ in range(repeats):
  model.zero_grad(set_to_none=True);start=time.perf_counter();loss,logical,full=batch_completion(model,rows);loss.backward();times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
 seconds=min(times[1:] or times);return {"microbatch":batch,"logical_completion_tokens_per_second":logical/seconds,"full_processed_tokens_per_second":full/seconds,"updates_per_second":1/seconds,"rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"stable":all(torch.isfinite(torch.tensor(losses)).tolist()),"seconds":seconds}
def benchmark_items(model,items,repeats=4):
 times=[];losses=[];logical=full=0
 for _ in range(repeats):
  model.zero_grad(set_to_none=True);start=time.perf_counter();loss,logical,full=batch_completion(model,items);loss.backward();times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
 seconds=min(times[1:] or times)
 return {"logical_completion_tokens_per_second":logical/seconds,"full_processed_tokens_per_second":full/seconds,"seconds_per_update":seconds,"stable":all(torch.isfinite(torch.tensor(losses)).tolist())}
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--threads",type=int,default=16);p.add_argument("--skip-compile",action="store_true");a=p.parse_args();torch.set_num_threads(a.threads);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);model=build_model(LatticeConfig(**ck["config"]));model.load_state_dict(ck["model"],strict=True);model.train();rows=[]
 for batch in (1,4,8,16):
  try:rows.append(benchmark(model,min(model.config.context_length,128),batch))
  except (RuntimeError,MemoryError) as e:rows.append({"microbatch":batch,"stable":False,"error":str(e)[:200]})
 good=[r for r in rows if r["stable"]];winner=max(good,key=lambda r:r["logical_completion_tokens_per_second"]) if good else None
 compiled={"attempted":False,"selected":False,"compile_mode":COMPILE_MODE,"reason":"compile intentionally skipped by --skip-compile" if a.skip_compile else "no stable eager microbatch"}
 if winner and not a.skip_compile:
  compiled["attempted"]=True
  def timeout(*_):raise TimeoutError(f"compile warmup exceeded {COMPILE_WARMUP_TIMEOUT_SECONDS} seconds")
  old=signal.signal(signal.SIGALRM,timeout)
  try:
   batch=winner["microbatch"];context=min(model.config.context_length,128);items=[]
   for _ in range(batch):
    ids=torch.randint(0,model.config.vocab_size,(context+1,)).tolist();items.append((ids[:-1],ids[1:],[False]*(context//2)+[True]*(context-context//2)))
   eager_result=benchmark_items(model,items)
   model.zero_grad(set_to_none=True);eager_loss,_,_=batch_completion(model,items);eager_loss.backward();grads={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None};model.zero_grad(set_to_none=True)
   start=time.perf_counter();signal.alarm(COMPILE_WARMUP_TIMEOUT_SECONDS);compiled_model=torch.compile(model,mode=COMPILE_MODE,fullgraph=False);compiled_loss,_,_=batch_completion(compiled_model,items);compiled_loss.backward();signal.alarm(0);warmup=time.perf_counter()-start;parity=bool(torch.allclose(eager_loss,compiled_loss,atol=1e-4,rtol=1e-4)) and all(torch.allclose(grads[k],p.grad,atol=1e-4,rtol=1e-3) for k,p in model.named_parameters() if k in grads)
   model.zero_grad(set_to_none=True);compiled_result=benchmark_items(compiled_model,items);gain=compiled_result["logical_completion_tokens_per_second"]/eager_result["logical_completion_tokens_per_second"]
   eager_tps=eager_result["logical_completion_tokens_per_second"];compiled_tps=compiled_result["logical_completion_tokens_per_second"]
   break_even_tokens=(warmup/(1/eager_tps-1/compiled_tps)) if compiled_tps>eager_tps else None
   selected=bool(parity and gain>=MIN_COMPILE_SPEEDUP and break_even_tokens is not None and break_even_tokens<=MAX_COMPILE_BREAK_EVEN_TOKENS)
   compiled={"attempted":True,"selected":selected,"compile_mode":COMPILE_MODE,"fullgraph":False,"parity":parity,"compile_warmup_seconds":warmup,"eager_logical_tokens_per_second":eager_tps,"compiled_logical_tokens_per_second":compiled_tps,"speedup":gain,"break_even_logical_tokens":break_even_tokens,"selection_gates":{"minimum_speedup":MIN_COMPILE_SPEEDUP,"maximum_break_even_logical_tokens":MAX_COMPILE_BREAK_EVEN_TOKENS,"warmup_timeout_seconds":COMPILE_WARMUP_TIMEOUT_SECONDS},"reason":"selected" if selected else "parity, speedup or amortization gate"}
  except Exception as e:compiled={**compiled,"attempted":True,"selected":False,"reason":f"{type(e).__name__}: {str(e)[:300]}"}
  finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
 result={"schema":"posttraining-systems-benchmark-v1","batch_candidates":rows,"selected_microbatch":winner["microbatch"] if winner else 1,"compile":compiled}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
