#!/usr/bin/env python3
"""Exact-resume post-training worker for SFT, ranking, SimPO, RFT and RLVR."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,signal,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from latticelm.config import LatticeConfig
from latticelm.final_recipe import make_optimizer
from latticelm.model import build_model
from latticelm.tokenizer import load_tokenizer
from latticelm.lattice_reason_v2 import generate_example,verify_candidate
from latticelm.lattice_reason_v2.natural import generate_from_tokens
from latticelm.posttraining.objectives import (ReplayMixer,completion_mask,continuation_logprobs,
 masked_causal_loss,ranking_cross_entropy,simpo_loss,grpo_advantages,rloo_advantages,policy_gradient_loss)
from latticelm.posttraining.rollout import sample,certify_trajectory,policy_identity
from latticelm.posttraining.state import sha256,write_sha_sidecar
STOP=False
def request_stop(*_):
 global STOP;STOP=True

class DataDReplay:
 def __init__(self,manifest,context):
  top=json.loads(Path(manifest).read_text());base=Path(manifest).parent;self.arrays=[];self.context=context;self.cursor=0
  for item in top["shards"]:
   meta=json.loads((base/item["manifest_path"]).read_text());self.arrays.append(np.memmap(base/meta["path"],dtype="<i4",mode="r"))
 def batch(self):
  array=self.arrays[self.cursor%len(self.arrays)];span=self.context+1;start=(self.cursor*span)%(len(array)-span);self.cursor+=1;z=torch.tensor(np.asarray(array[start:start+span]).astype(np.int64));return z[:-1][None,:],z[1:][None,:]
 def state_dict(self):return {"cursor":self.cursor}
 def load_state_dict(self,state):self.cursor=int(state["cursor"])
 def natural(self,index):
  array=self.arrays[index%len(self.arrays)];return generate_from_tokens(array,index,source_id=f"data-d-v4-train-shard-{index%len(self.arrays)}")

def save_checkpoint(root,payload,milestone=False):
 root.mkdir(parents=True,exist_ok=True);tmp=root/".latest.pt.tmp";torch.save(payload,tmp);digest=sha256(tmp);latest=root/"latest.pt";os.replace(tmp,latest);write_sha_sidecar(latest)
 if milestone:
  path=root/f"milestone-{payload['training_tokens']}.pt"
  if not path.exists():os.link(latest,path);write_sha_sidecar(path)
 return digest
def freeze_record(root,index,value):
 path=root/"frozen"/f"{index:012d}.json";path.parent.mkdir(parents=True,exist_ok=True);encoded=(json.dumps(value,sort_keys=True,default=str)+"\n").encode()
 if path.exists():
  if path.read_bytes()!=encoded:raise RuntimeError("immutable pair/rollout conflict")
 else:
  tmp=path.with_suffix(".tmp")
  with tmp.open("wb") as f:f.write(encoded);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path);write_sha_sidecar(path)
 return str(path)
def encode_example(tok,ex,context):
 p=tok.encode(ex.prompt);a=tok.encode(" "+ex.answer);p=p[-max(1,context-len(a)):];ids=([1]+p+a)[-context:];prompt_len=len(ids)-len(a);x=ids[:-1];y=ids[1:];mask=[i>=prompt_len-1 for i in range(len(x))];return x,y,mask
def candidate_score(model,tok,prompt,candidate,context):
 p=tok.encode(prompt);a=tok.encode(" "+candidate);ids=([1]+p+a)[-(context+1):];x=torch.tensor(ids[:-1])[None,:];y=torch.tensor(ids[1:])[None,:];mask=torch.zeros_like(y,dtype=torch.bool);mask[:,-min(len(a),len(y[0])):]=True;logits=model(x)[0];return continuation_logprobs(logits,y,mask)
def token_candidate_score(model,prompt,candidate,context):
 ids=([1]+list(prompt)+list(candidate))[-(context+1):];x=torch.tensor(ids[:-1])[None,:];y=torch.tensor(ids[1:])[None,:];mask=torch.zeros_like(y,dtype=torch.bool);mask[:,-min(len(candidate),len(y[0])):]=True;return continuation_logprobs(model(x)[0],y,mask)
def payload(model,opt,args,cursor,tokens,updates,mixer,replay,parent_sha,last_loss):
 return {"schema":"posttraining-checkpoint-v1","model":model.state_dict(),"optimizers":opt.state_dict(),"config":model.config.to_dict(),"method":args.method,"parent_checkpoint_sha256":parent_sha,"tokenizer_sha256":sha256(args.tokenizer),"manifest_sha256":sha256(args.manifest),"training_tokens":tokens,"updates":updates,"data_cursor":cursor,"curriculum_state":{"composition_depth":1+cursor%4},"replay_selector_state":mixer.state_dict(),"data_replay_state":replay.state_dict(),"rollout_generation_cursor":cursor,"reward_statistics":{},"frozen_data_root":str(args.output/"frozen"),"next_batch_identity":hashlib.sha256(f"{args.seed}|{cursor}|{replay.cursor}".encode()).hexdigest(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"optimizer_kind":args.optimizer,"learning_rate":args.lr,"replay_fraction":args.replay,"train_loss":last_loss}
def main():
 p=argparse.ArgumentParser();p.add_argument("--base",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--method",choices=("sft","ranking","rft","simpo","rlvr-grpo","rlvr-rloo","recovery"),required=True);p.add_argument("--tokens",type=int,required=True);p.add_argument("--optimizer",choices=("muon_hybrid","adamw"),default="muon_hybrid");p.add_argument("--lr",type=float,default=3e-5);p.add_argument("--replay",type=float,default=.2);p.add_argument("--seed",type=int,default=260921);p.add_argument("--threads",type=int,default=16);p.add_argument("--checkpoint-tokens",type=int,default=1_000_000);p.add_argument("--stop-epoch",type=float,default=float("inf"));a=p.parse_args();signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);parent_sha=sha256(a.base);base=torch.load(a.base,map_location="cpu",weights_only=False);cfg=LatticeConfig(**base["config"]);model=build_model(cfg);model.load_state_dict(base["model"],strict=True);tok=load_tokenizer(a.tokenizer);opt,_=make_optimizer(model,a.optimizer,a.lr,cfg.weight_decay,(cfg.adam_beta1,cfg.adam_beta2));mixer=ReplayMixer(1.0 if a.method=="recovery" else a.replay,a.seed);replay=DataDReplay(a.manifest,cfg.context_length);cursor=tokens=updates=0;last=float("nan")
 latest=a.output/"latest.pt"
 if latest.exists() and latest.with_suffix(".pt.sha256").exists() and sha256(latest)==latest.with_suffix(".pt.sha256").read_text().strip():
  ck=torch.load(latest,map_location="cpu",weights_only=False)
  if ck["parent_checkpoint_sha256"]!=parent_sha or ck["method"]!=a.method or ck["tokenizer_sha256"]!=sha256(a.tokenizer):raise RuntimeError("resume identity mismatch")
  model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizers"]);mixer.load_state_dict(ck["replay_selector_state"]);replay.load_state_dict(ck["data_replay_state"]);cursor=ck["data_cursor"];tokens=ck["training_tokens"];updates=ck["updates"];last=ck["train_loss"]
  if ck["next_batch_identity"]!=hashlib.sha256(f"{a.seed}|{cursor}|{replay.cursor}".encode()).hexdigest():raise RuntimeError("resume next-batch identity mismatch")
  random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"]);torch.set_rng_state(ck["torch_rng"])
 model.train();started=time.perf_counter()
 while tokens<a.tokens:
  ex=generate_example(cursor,a.seed,"train",1+cursor%4);opt.zero_grad();increment=0;use_replay=mixer.next_is_replay()
  if use_replay:
   xt,yt=replay.batch();loss=F.cross_entropy(model(xt)[0].reshape(-1,cfg.vocab_size),yt.reshape(-1));increment=yt.numel()
  elif a.method=="rft":
   prompt=tok.encode(ex.prompt)[-max(1,cfg.context_length-16):];identity=policy_identity(model);trajectories=[sample(model,prompt,ex.global_id,lambda ids:verify_candidate(ex,tok.decode(list(ids)).strip()),max_new_tokens=8,eos_id=getattr(tok,"eos_id",2),seed=a.seed+cursor*8+i,policy_sha256=identity) for i in range(8)];successful=next((t for t in trajectories if t.verifier_result),None)
   if successful is None:
    cursor+=1
    if cursor>10000 and tokens==0:raise RuntimeError("RFT eligibility failed: no verified trajectories")
    continue
   freeze_record(a.output,cursor,{"schema":"rft-trajectory-v1","trajectory":successful.to_dict()})
   ids=list(successful.prompt_ids)+list(successful.generated_ids);xt=torch.tensor(ids[:-1])[None,:];yt=torch.tensor(ids[1:])[None,:];mt=torch.zeros_like(yt,dtype=torch.bool);mt[:,-len(successful.generated_ids):]=True;loss=masked_causal_loss(model(xt)[0],yt,mt);increment=int(mt.sum())
  elif a.method in {"sft","recovery"}:
   x,y,mask=encode_example(tok,ex,cfg.context_length);xt=torch.tensor(x)[None,:];yt=torch.tensor(y)[None,:];mt=torch.tensor(mask)[None,:];loss=masked_causal_loss(model(xt)[0],yt,mt);increment=int(mt.sum())
  elif a.method=="ranking":
   if cursor%5==0:
    natural=replay.natural(cursor);correct=natural.candidates.index(natural.answer_ids);freeze_record(a.output,cursor,{"schema":"natural-ranking-pair-v1",**natural.to_dict(),"correct_index":correct,"error_type":"nearby_same_source_span"});scores=[token_candidate_score(model,natural.prompt_ids,c,cfg.context_length)[1][0] for c in natural.candidates];increment=sum(map(len,natural.candidates))
   else:
    correct=ex.candidate_position;freeze_record(a.output,cursor,{"schema":"ranking-pair-v1","example_id":ex.global_id,"world_state_hash":ex.world_state_hash,"candidates":ex.candidates,"correct_index":correct,"error_type":ex.hard_negative_error_type});scores=[candidate_score(model,tok,ex.prompt,c,cfg.context_length)[1][0] for c in ex.candidates];increment=sum(len(tok.encode(c)) for c in ex.candidates)
   score=torch.stack(scores)[None,:];loss=ranking_cross_entropy(score,torch.tensor([correct]))
  elif a.method=="simpo":
   freeze_record(a.output,cursor,{"schema":"simpo-pair-v1","example_id":ex.global_id,"chosen":ex.answer,"rejected":next(x for x in ex.candidates if x!=ex.answer),"error_type":ex.hard_negative_error_type})
   chosen=candidate_score(model,tok,ex.prompt,ex.answer,cfg.context_length)[1];rejected=candidate_score(model,tok,ex.prompt,next(x for x in ex.candidates if x!=ex.answer),cfg.context_length)[1];loss=simpo_loss(chosen,rejected);increment=len(tok.encode(ex.answer))
  else:
   prompt=tok.encode(ex.prompt)[-max(1,cfg.context_length-16):];identity=policy_identity(model);trajectories=[sample(model,prompt,ex.global_id,lambda ids:verify_candidate(ex,tok.decode(list(ids)).strip()),max_new_tokens=8,eos_id=getattr(tok,"eos_id",2),seed=a.seed+cursor*4+i,policy_sha256=identity) for i in range(4)]
   if any(certify_trajectory(model,t,current_policy_sha256=identity)["status"]!="PASS" for t in trajectories):raise RuntimeError("rollout parity gate failed")
   freeze_record(a.output,cursor,{"schema":"rlvr-rollout-group-v1","example_id":ex.global_id,"trajectories":[t.to_dict() for t in trajectories]})
   rewards=torch.tensor([t.reward for t in trajectories]);groups=torch.zeros(4,dtype=torch.long);advantages=grpo_advantages(rewards,groups) if a.method=="rlvr-grpo" else rloo_advantages(rewards,groups);seqs=[];masks=[];targets=[]
   for t in trajectories:
    ids=list(t.prompt_ids)+list(t.generated_ids);seqs.append(ids[:-1]);targets.append(ids[1:]);masks.append([False]*(len(t.prompt_ids)-1)+list(t.completion_mask))
   width=max(map(len,seqs));pad=lambda z,v:z+[v]*(width-len(z));xt=torch.tensor([pad(x,0) for x in seqs]);yt=torch.tensor([pad(x,0) for x in targets]);mt=torch.tensor([pad(x,False) for x in masks]);logits=model(xt)[0];sampled=F.log_softmax(logits,-1).gather(-1,yt[...,None]).squeeze(-1);loss=policy_gradient_loss(sampled,mt,advantages);increment=int(mt.sum())
  if not torch.isfinite(loss):raise FloatingPointError("nonfinite post-training loss")
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();last=float(loss.detach());tokens+=max(1,min(increment,a.tokens-tokens));updates+=1;cursor+=1
  if tokens==a.tokens or tokens//a.checkpoint_tokens!=(tokens-increment)//a.checkpoint_tokens:save_checkpoint(a.output,payload(model,opt,a,cursor,tokens,updates,mixer,replay,parent_sha,last),True)
  if STOP or time.time()>=a.stop_epoch:
   save_checkpoint(a.output,payload(model,opt,a,cursor,tokens,updates,mixer,replay,parent_sha,last),True);(a.output/"safe_stop.json").write_text(json.dumps({"status":"SAFE_STOP","training_tokens":tokens,"at":time.time()})+"\n");return 75
 result={"schema":"posttraining-worker-result-v1","status":"COMPLETE","method":a.method,"checkpoint":str(a.output/"latest.pt"),"checkpoint_sha256":sha256(a.output/"latest.pt"),"parent_checkpoint_sha256":parent_sha,"training_tokens":tokens,"updates":updates,"train_loss":last,"wall_seconds":time.perf_counter()-started};(a.output/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":raise SystemExit(main() or 0)
