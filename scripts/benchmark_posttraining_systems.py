#!/usr/bin/env python3
"""Measure completion-batch throughput on the frozen BASE after handoff."""
import argparse,json,resource,signal,time
from pathlib import Path
import torch
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from scripts.train_posttraining_worker import batch_completion

def benchmark(model,context,batch,repeats=3):
 rows=[]
 for i in range(batch):
  ids=torch.randint(0,model.config.vocab_size,(context+1,)).tolist();rows.append((ids[:-1],ids[1:],[False]*(context//2)+[True]*(context-context//2)))
 times=[];losses=[]
 for _ in range(repeats):
  model.zero_grad(set_to_none=True);start=time.perf_counter();loss,logical,full=batch_completion(model,rows);loss.backward();times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
 seconds=min(times[1:] or times);return {"microbatch":batch,"logical_completion_tokens_per_second":logical/seconds,"full_processed_tokens_per_second":full/seconds,"updates_per_second":1/seconds,"rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"stable":all(torch.isfinite(torch.tensor(losses)).tolist()),"seconds":seconds}
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--threads",type=int,default=16);p.add_argument("--skip-compile",action="store_true");a=p.parse_args();torch.set_num_threads(a.threads);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);model=build_model(LatticeConfig(**ck["config"]));model.load_state_dict(ck["model"],strict=True);model.train();rows=[]
 for batch in (1,4,8,16):
  try:rows.append(benchmark(model,min(model.config.context_length,128),batch))
  except (RuntimeError,MemoryError) as e:rows.append({"microbatch":batch,"stable":False,"error":str(e)[:200]})
 good=[r for r in rows if r["stable"]];winner=max(good,key=lambda r:r["logical_completion_tokens_per_second"]) if good else None
 compiled={"selected":False,"reason":"compile omitted in pre-handoff smoke" if a.skip_compile else "compile benchmark did not pass"}
 if winner and not a.skip_compile:
  def timeout(*_):raise TimeoutError("compile warmup exceeded 120 seconds")
  old=signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
  try:
   batch=winner["microbatch"];context=min(model.config.context_length,128);items=[]
   for _ in range(batch):
    ids=torch.randint(0,model.config.vocab_size,(context+1,)).tolist();items.append((ids[:-1],ids[1:],[False]*(context//2)+[True]*(context-context//2)))
   model.zero_grad(set_to_none=True);eager,_,_=batch_completion(model,items);eager.backward();grads={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None};model.zero_grad(set_to_none=True);start=time.perf_counter();compiled_model=torch.compile(model);compiled_loss,logical,_=batch_completion(compiled_model,items);compiled_loss.backward();warmup=time.perf_counter()-start;parity=bool(torch.allclose(eager,compiled_loss,atol=1e-4,rtol=1e-4)) and all(torch.allclose(grads[k],p.grad,atol=1e-4,rtol=1e-3) for k,p in model.named_parameters() if k in grads)
   model.zero_grad(set_to_none=True);start=time.perf_counter();loss,_,_=batch_completion(compiled_model,items);loss.backward();steady=time.perf_counter()-start;gain=(logical/steady)/winner["logical_completion_tokens_per_second"]
   compiled={"selected":bool(parity and gain>1.1 and warmup<3600),"parity":parity,"warmup_seconds":warmup,"steady_logical_tokens_per_second":logical/steady,"speedup":gain,"reason":"selected" if parity and gain>1.1 and warmup<3600 else "parity, speed or warmup gate"}
  except Exception as e:compiled={"selected":False,"reason":f"{type(e).__name__}: {str(e)[:180]}"}
  finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)
 result={"schema":"posttraining-systems-benchmark-v1","batch_candidates":rows,"selected_microbatch":winner["microbatch"] if winner else 1,"compile":compiled}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
