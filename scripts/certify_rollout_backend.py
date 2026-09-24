#!/usr/bin/env python3
"""Post-production real-policy parity and CPU topology benchmark."""
import argparse,json,resource,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
def run_worker(args):
 checkpoint,tokenizer,threads,seed,batched=args
 import torch
 from latticelm.config import LatticeConfig
 from latticelm.model import build_model
 from latticelm.tokenizer import load_tokenizer
 from latticelm.lattice_reason_v2 import generate_example,verify_candidate
 from latticelm.posttraining.rollout import sample,sample_many,certify_trajectory
 torch.set_num_threads(threads);torch.set_num_interop_threads(1);ck=torch.load(checkpoint,map_location="cpu",weights_only=False);cfg=LatticeConfig(**ck["config"]);model=build_model(cfg);model.load_state_dict(ck["model"]);model.eval();tok=load_tokenizer(tokenizer);tokens=0;started=time.perf_counter()
 for i in range(2):
  ex=generate_example(seed+i,split="validation");prompt=tok.encode(ex.prompt)[-240:];verify=lambda ids:verify_candidate(ex,tok.decode(list(ids)).strip())
  trajectories=sample_many(model,prompt,ex.global_id,verify,4,max_new_tokens=8,seed=seed+i*4,policy_sha256="topology-only") if batched else [sample(model,prompt,ex.global_id,verify,max_new_tokens=8,seed=seed+i*4+j,policy_sha256="topology-only") for j in range(4)]
  if any(certify_trajectory(model,t,current_policy_sha256="topology-only")["status"]!="PASS" for t in trajectories):raise RuntimeError("topology rollout parity failed")
  tokens+=sum(len(t.generated_ids) for t in trajectories)
 return {"tokens":tokens,"seconds":time.perf_counter()-started,"rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--tokenizer",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 import torch
 from latticelm.config import LatticeConfig
 from latticelm.model import build_model
 from latticelm.tokenizer import load_tokenizer
 from latticelm.lattice_reason_v2 import generate_example,verify_candidate
 from latticelm.posttraining.rollout import sample,policy_identity,certify_trajectory
 torch.set_num_threads(1);torch.set_num_interop_threads(1);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);cfg=LatticeConfig(**ck["config"]);model=build_model(cfg);model.load_state_dict(ck["model"]);model.eval();tok=load_tokenizer(a.tokenizer);ex=generate_example(0,split="validation");identity=policy_identity(model);trajectory=sample(model,tok.encode(ex.prompt)[-240:],ex.global_id,lambda ids:verify_candidate(ex,tok.decode(list(ids)).strip()),8,seed=77,policy_sha256=identity);cert=certify_trajectory(model,trajectory,current_policy_sha256=identity)
 if cert["status"]!="PASS":raise RuntimeError("real-policy rollout parity failed")
 rows=[]
 for processes,threads,batched in ((1,16,False),(1,16,True),(2,8,False),(4,4,False),(8,2,False)):
  started=time.perf_counter()
  with ProcessPoolExecutor(max_workers=processes) as pool:values=list(pool.map(run_worker,[(a.checkpoint,a.tokenizer,threads,1000+i*10,batched) for i in range(processes)]))
  wall=time.perf_counter()-started;rows.append({"processes":processes,"threads":threads,"batched":batched,"verified_rollout_tokens":sum(x["tokens"] for x in values),"wall_seconds":wall,"aggregate_tokens_per_second":sum(x["tokens"] for x in values)/wall,"aggregate_peak_rss_bytes":sum(x["rss_bytes"] for x in values),"stable":True,"online_worker_supported":processes==1})
 winner=max((x for x in rows if x["online_worker_supported"]),key=lambda x:x["aggregate_tokens_per_second"]);result={"schema":"rollout-backend-certification-v1","status":"PASS","trajectory_parity":cert,"topologies":rows,"selected":winner,"topology_reason":"online policy updates use the certified single-process variant"};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
