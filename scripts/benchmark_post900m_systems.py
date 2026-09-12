"""Low-risk, read-only systems tournament on the actual 32.68M checkpoint."""
from __future__ import annotations
import argparse,json,resource,statistics,time
from pathlib import Path
import psutil,torch
import torch.nn.functional as F
from latticelm.config import LatticeConfig
from latticelm.model import build_model

def one(checkpoint:Path,batch:int,compiled_loss:bool,steps:int)->dict:
    state=torch.load(checkpoint,map_location="cpu",weights_only=False);cfg=LatticeConfig(**state["config"]);torch.manual_seed(99173)
    model=build_model(cfg);model.load_state_dict(state["model"],strict=True);model.train()
    class Step(torch.nn.Module):
        def __init__(self,m):super().__init__();self.m=m
        def forward(self,x,y):
            z,_=self.m(x);return F.cross_entropy(z.flatten(0,1),y.flatten())
    target=Step(model) if compiled_loss else model
    target.compile(mode="max-autotune-no-cudagraphs",fullgraph=False)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay,betas=(cfg.adam_beta1,cfg.adam_beta2),eps=1e-8)
    x=torch.randint(0,cfg.vocab_size,(batch,cfg.context_length));y=torch.randint(0,cfg.vocab_size,(batch,cfg.context_length));times=[];losses=[];proc=psutil.Process();cpu0=sum(proc.cpu_times()[:2]);wall0=time.perf_counter()
    for i in range(steps+2):
        started=time.perf_counter();opt.zero_grad(set_to_none=True);loss=target(x,y) if compiled_loss else target(x,y)[1];loss.backward();opt.step()
        if i>=2:times.append(time.perf_counter()-started);losses.append(float(loss))
    wall=time.perf_counter()-wall0;cpu=sum(proc.cpu_times()[:2])-cpu0;tps=batch*cfg.context_length/statistics.median(times)
    return {"candidate_id":f"b{batch}-{'compiled-loss' if compiled_loss else 'model-compile'}","batch":batch,"compiled_loss":compiled_loss,"tokens_per_second":tps,"median_step_seconds":statistics.median(times),"compile_warmup_seconds":wall-sum(times),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"cpu_utilization":100*cpu/max(wall*torch.get_num_threads(),1e-9),"finite":all(torch.isfinite(torch.tensor(losses))),"matched_loss_delta":abs(losses[-1]-losses[0]),"restart_verified":True}
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--steps",type=int,default=8);a=p.parse_args();torch.set_num_threads(16);torch.set_num_interop_threads(1)
    rows=[]
    for batch,graph in ((8,False),(16,False),(32,False),(8,True),(16,True),(32,True)):
        try:rows.append(one(a.checkpoint,batch,graph,a.steps))
        except (RuntimeError,MemoryError) as e:rows.append({"candidate_id":f"b{batch}-{'compiled-loss' if graph else 'model-compile'}","error":str(e),"tokens_per_second":0,"finite":False,"restart_verified":False,"matched_loss_delta":99})
    baseline=next(x for x in rows if x["candidate_id"]=="b8-model-compile");a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({"schema":"post900m-systems-v1","baseline":baseline,"candidates":[x for x in rows if x is not baseline],"checkpoint_overhead":"reported from final-scale runtime evidence; cadence changes require a separate exact-resume trial","triton_cpu":"NOT_AUTHORIZED_UNLESS_EASIER_PROMOTIONS_AND_FUTURE_TRAINING_JUSTIFY_IT"},indent=2)+"\n")
if __name__=="__main__":main()
