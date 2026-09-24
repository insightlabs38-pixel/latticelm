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
 masked_causal_loss,ranking_objective,joint_objective,simpo_loss,grpo_advantages,rloo_advantages,policy_gradient_loss)
from latticelm.posttraining.rollout import sample,sample_many,certify_trajectory
from latticelm.posttraining.provenance import append_record
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
 return append_record(root/"frozen",index,value)
def encode_example(tok,ex,context):
 p=tok.encode(ex.prompt);a=tok.encode(" "+ex.answer);p=p[-max(1,context-len(a)):];ids=([1]+p+a)[-context:];prompt_len=len(ids)-len(a);x=ids[:-1];y=ids[1:];mask=[i>=prompt_len-1 for i in range(len(x))];return x,y,mask
def candidate_score(model,tok,prompt,candidate,context):
 p=tok.encode(prompt);a=tok.encode(" "+candidate);ids=([1]+p+a)[-(context+1):];x=torch.tensor(ids[:-1])[None,:];y=torch.tensor(ids[1:])[None,:];mask=torch.zeros_like(y,dtype=torch.bool);mask[:,-min(len(a),len(y[0])):]=True;logits=model(x)[0];return continuation_logprobs(logits,y,mask)
def token_candidate_score(model,prompt,candidate,context):
 ids=([1]+list(prompt)+list(candidate))[-(context+1):];x=torch.tensor(ids[:-1])[None,:];y=torch.tensor(ids[1:])[None,:];mask=torch.zeros_like(y,dtype=torch.bool);mask[:,-min(len(candidate),len(y[0])):]=True;return continuation_logprobs(model(x)[0],y,mask)
def batch_candidate_scores(model,prompt,candidates,context):
 rows=[]
 for candidate in candidates:
  ids=([1]+list(prompt)+list(candidate))[-(context+1):]
  rows.append((ids[:-1],ids[1:],min(len(candidate),len(ids)-1)))
 width=max(len(x) for x,_,_ in rows);x=torch.zeros((len(rows),width),dtype=torch.long);y=x.clone();mask=torch.zeros_like(x,dtype=torch.bool)
 for i,(xi,yi,n) in enumerate(rows):
  x[i,:len(xi)]=torch.tensor(xi);y[i,:len(yi)]=torch.tensor(yi);mask[i,len(yi)-n:len(yi)]=True
 return continuation_logprobs(model(x)[0],y,mask)
def batch_completion(model,items):
 width=max(len(x) for x,_,_ in items);x=torch.zeros((len(items),width),dtype=torch.long);y=x.clone();m=torch.zeros_like(x,dtype=torch.bool)
 for i,(xi,yi,mi) in enumerate(items):x[i,:len(xi)]=torch.tensor(xi);y[i,:len(yi)]=torch.tensor(yi);m[i,:len(mi)]=torch.tensor(mi)
 return masked_causal_loss(model(x)[0],y,m),int(m.sum()),sum(len(xi) for xi,_,_ in items)
def rollout_group(model,prompt,example_id,verify,k,seed,eos_id,identity,batched):
 if batched:
  return [t for start in range(0,k,4) for t in sample_many(model,prompt,example_id,verify,min(4,k-start),max_new_tokens=8,eos_id=eos_id,seed=seed+start,policy_sha256=identity)]
 return [sample(model,prompt,example_id,verify,max_new_tokens=8,eos_id=eos_id,seed=seed+i,policy_sha256=identity) for i in range(k)]
def payload(model,opt,args,cursor,tokens,updates,mixer,replay,parent_sha,last_loss):
 return {"schema":"posttraining-checkpoint-v1","model":model.state_dict(),"optimizers":opt.state_dict(),"config":model.config.to_dict(),"method":args.method,"method_chain":args.method_chain,"parent_checkpoint_sha256":parent_sha,"tokenizer_sha256":sha256(args.tokenizer),"manifest_sha256":sha256(args.manifest),"training_tokens":tokens,"processed_tokens":args.processed_tokens,"updates":updates,"data_cursor":cursor,"curriculum_state":{"composition_depth":1+cursor%4},"replay_selector_state":mixer.state_dict(),"data_replay_state":replay.state_dict(),"rollout_generation_cursor":cursor,"reward_statistics":{},"frozen_data_root":str(args.output/"frozen"),"next_batch_identity":hashlib.sha256(f"{args.seed}|{cursor}|{replay.cursor}".encode()).hexdigest(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"optimizer_kind":args.optimizer,"learning_rate":args.lr,"replay_fraction":args.replay,"train_loss":last_loss,"ranking_mode":args.ranking_mode,"natural_fraction":args.natural_fraction,"rank_weight":args.rank_weight,"margin_weight":args.margin_weight,"microbatch":args.microbatch,"compile_mode":args.compile_mode}
def main():
 p=argparse.ArgumentParser();p.add_argument("--base",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--method",choices=("sft","ranking","joint","rft","simpo","rlvr-grpo","rlvr-rloo","recovery"),required=True);p.add_argument("--tokens",type=int,required=True);p.add_argument("--optimizer",choices=("muon_hybrid","adamw"),default="muon_hybrid");p.add_argument("--lr",type=float,default=3e-5);p.add_argument("--replay",type=float,default=.2);p.add_argument("--seed",type=int,default=260921);p.add_argument("--threads",type=int,default=16);p.add_argument("--checkpoint-tokens",type=int,default=1_000_000);p.add_argument("--stop-epoch",type=float,default=float("inf"));p.add_argument("--microbatch",type=int,default=1);p.add_argument("--compile-mode",type=int,choices=(0,1),default=0);p.add_argument("--ranking-mode",choices=("ranking_raw","ranking_norm","ranking_dual"),default="ranking_dual");p.add_argument("--natural-fraction",type=float,default=.2);p.add_argument("--rank-weight",type=float,default=.5);p.add_argument("--margin-weight",type=float,default=0.);p.add_argument("--method-chain",default="");p.add_argument("--rollout-topology",type=Path);p.add_argument("--frontier-map",type=Path);p.add_argument("--rollout-k",type=int,default=8);a=p.parse_args();signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);parent_sha=sha256(a.base);base=torch.load(a.base,map_location="cpu",weights_only=False);cfg=LatticeConfig(**base["config"]);model=build_model(cfg);model.load_state_dict(base["model"],strict=True);tok=load_tokenizer(a.tokenizer);opt,_=make_optimizer(model,a.optimizer,a.lr,cfg.weight_decay,(cfg.adam_beta1,cfg.adam_beta2));mixer=ReplayMixer(1.0 if a.method=="recovery" else 0. if a.method=="joint" else a.replay,a.seed);replay=DataDReplay(a.manifest,cfg.context_length);cursor=tokens=updates=0;last=float("nan")
 latest=a.output/"latest.pt"
 if latest.exists() and latest.with_suffix(".pt.sha256").exists() and sha256(latest)==latest.with_suffix(".pt.sha256").read_text().strip():
  ck=torch.load(latest,map_location="cpu",weights_only=False)
  if ck["parent_checkpoint_sha256"]!=parent_sha or ck["method"]!=a.method or ck["tokenizer_sha256"]!=sha256(a.tokenizer) or ck["manifest_sha256"]!=sha256(a.manifest) or ck.get("microbatch",1)!=a.microbatch or ck.get("ranking_mode","ranking_dual")!=a.ranking_mode or ck.get("compile_mode",0)!=a.compile_mode:raise RuntimeError("resume identity mismatch")
  model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizers"]);mixer.load_state_dict(ck["replay_selector_state"]);replay.load_state_dict(ck["data_replay_state"]);cursor=ck["data_cursor"];tokens=ck["training_tokens"];updates=ck["updates"];last=ck["train_loss"]
  if ck["next_batch_identity"]!=hashlib.sha256(f"{a.seed}|{cursor}|{replay.cursor}".encode()).hexdigest():raise RuntimeError("resume next-batch identity mismatch")
  random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"]);torch.set_rng_state(ck["torch_rng"])
 if a.microbatch<1 or not 0<=a.natural_fraction<=1 or a.rollout_k<1:raise ValueError("invalid worker batch or mixture")
 if a.rollout_topology:
  topology=json.loads(a.rollout_topology.read_text()).get("selected",{})
  if topology.get("threads"):torch.set_num_threads(int(topology["threads"]))
 else:topology=None
 frontier=json.loads(a.frontier_map.read_text()).get("regions",{}) if a.frontier_map and a.frontier_map.exists() else {}
 model.train();train_model=torch.compile(model) if a.compile_mode and a.method in ("sft","recovery") else model;started=time.perf_counter();processed_tokens=ck.get("processed_tokens",0) if latest.exists() and "ck" in locals() else 0;a.processed_tokens=processed_tokens;rollout_audit_every=32
 while tokens<a.tokens:
  if STOP or time.time()>=a.stop_epoch:
   if tokens==0:
    a.output.mkdir(parents=True,exist_ok=True);(a.output/"ineligible.json").write_text(json.dumps({"status":"NO_VALID_UPDATE_BEFORE_STOP","cursor":cursor})+"\n");return 0
   save_checkpoint(a.output,payload(model,opt,a,cursor,tokens,updates,mixer,replay,parent_sha,last),True);status="SAFE_STOPPED_VALID";break
  ex=generate_example(cursor,a.seed,"train",1+cursor%4);opt.zero_grad();increment=0;use_replay=mixer.next_is_replay()
  if a.method in {"sft","recovery"}:
   items=[]
   for j in range(a.microbatch):
    take_replay=use_replay if j==0 else mixer.next_is_replay()
    if take_replay:
     xt,yt=replay.batch();items.append((xt[0].tolist(),yt[0].tolist(),[True]*yt.numel()))
    else:
     item=generate_example(cursor+j,a.seed,"train",1+(cursor+j)%4);items.append(encode_example(tok,item,cfg.context_length))
   loss,increment,processed=batch_completion(train_model,items);processed_tokens+=processed;cursor+=len(items)-1
  elif use_replay:
   xt,yt=replay.batch();loss=F.cross_entropy(model(xt)[0].reshape(-1,cfg.vocab_size),yt.reshape(-1));increment=yt.numel();processed_tokens+=xt.numel()
  elif a.method=="rft":
   region=frontier.get(str(("+".join(ex.skills),ex.composition_depth)),{});rate=region.get("pass_at_8");stride=8 if rate==0 else 4 if rate is not None and rate>=.95 else 1
   if cursor%stride:
    cursor+=1;continue
   k=min(32,max(4,16 if rate is not None and 0<rate<.25 else 4 if rate is not None and rate>.75 else a.rollout_k));prompt=tok.encode(ex.prompt)[-max(1,cfg.context_length-16):];identity=f"{parent_sha}:{updates}";trajectories=rollout_group(model,prompt,ex.global_id,lambda ids:verify_candidate(ex,tok.decode(list(ids)).strip()),k,a.seed+cursor*32,getattr(tok,"eos_id",2),identity,bool(topology and topology.get("batched")));successful=list({t.generated_ids:t for t in trajectories if t.verifier_result}.values())
   if not successful:
    cursor+=1
    if cursor>1024 and tokens==0:
     a.output.mkdir(parents=True,exist_ok=True);(a.output/"ineligible.json").write_text(json.dumps({"status":"NO_LEARNABLE_TRAJECTORY","attempted_examples":cursor})+"\n");return 0
    continue
   audit=trajectories if updates<2 or updates%rollout_audit_every==0 else trajectories[:1]
   if any(certify_trajectory(model,t,current_policy_sha256=identity)["status"]!="PASS" for t in audit):
    if any(certify_trajectory(model,t,current_policy_sha256=identity)["status"]!="PASS" for t in trajectories):raise RuntimeError("RFT rollout parity gate failed")
   freeze_record(a.output,cursor,{"schema":"rft-trajectory-group-v2","example_id":ex.global_id,"policy_identity":identity,"difficulty":{"skills":ex.skills,"composition_depth":ex.composition_depth,"reasoning_depth":ex.reasoning_depth,"distractors":ex.distractor_count,"prior_pass_at_8":rate},"sampled_k":k,"successful_trajectories":[t.to_dict() for t in successful]})
   chosen=successful[0];ids=list(chosen.prompt_ids)+list(chosen.generated_ids);xt=torch.tensor(ids[:-1])[None,:];yt=torch.tensor(ids[1:])[None,:];mt=torch.zeros_like(yt,dtype=torch.bool);mt[:,-len(chosen.generated_ids):]=True;loss=masked_causal_loss(model(xt)[0],yt,mt);increment=int(mt.sum());processed_tokens+=xt.numel()
  elif a.method in {"ranking","joint"}:
   if cursor%10<int(a.natural_fraction*10):
    natural=replay.natural(cursor);correct=natural.candidates.index(natural.answer_ids);freeze_record(a.output,cursor,{"schema":"natural-ranking-pair-v1",**natural.to_dict(),"correct_index":correct,"error_type":"nearby_same_source_span"});prompt=natural.prompt_ids;candidates=natural.candidates;increment=sum(map(len,candidates))
   else:
    correct=ex.candidate_position;freeze_record(a.output,cursor,{"schema":"ranking-pair-v1","example_id":ex.global_id,"world_state_hash":ex.world_state_hash,"candidates":ex.candidates,"correct_index":correct,"error_type":ex.hard_negative_error_type});prompt=tok.encode(ex.prompt);candidates=[tok.encode(" "+c) for c in ex.candidates];increment=sum(map(len,candidates))
   raw,norm=batch_candidate_scores(model,prompt,candidates,cfg.context_length);rank=ranking_objective(raw[None,:],norm[None,:],torch.tensor([correct]),a.ranking_mode,margin_weight=a.margin_weight);processed_tokens+=sum(min(cfg.context_length,len(prompt)+len(c)) for c in candidates)
   if a.method=="joint":
    item=encode_example(tok,ex,cfg.context_length);completion,extra,full=batch_completion(model,[item]);processed_tokens+=full;increment+=extra;replay_loss=None
    if a.replay>0:
     xt,yt=replay.batch();replay_loss=F.cross_entropy(model(xt)[0].reshape(-1,cfg.vocab_size),yt.reshape(-1));processed_tokens+=xt.numel();increment+=yt.numel()
    loss=joint_objective(completion,rank,replay_loss,rank_weight=a.rank_weight,replay_weight=a.replay)
   else:loss=rank
  elif a.method=="simpo":
   candidates=[tok.encode(" "+c) for c in ex.candidates];_,scores=batch_candidate_scores(model,tok.encode(ex.prompt),candidates,cfg.context_length);wrong=scores.detach().clone();wrong[ex.candidate_position]=float("-inf");hard=int(wrong.argmax());freeze_record(a.output,cursor,{"schema":"simpo-pair-v2","example_id":ex.global_id,"chosen":ex.answer,"rejected":ex.candidates[hard],"error_type":ex.hard_negative_error_type,"candidate_scores":scores.detach().tolist()});loss=simpo_loss(scores[ex.candidate_position:ex.candidate_position+1],scores[hard:hard+1]);increment=len(candidates[ex.candidate_position]);processed_tokens+=sum(min(cfg.context_length,len(tok.encode(ex.prompt))+len(c)) for c in candidates)
  else:
   prompt=tok.encode(ex.prompt)[-max(1,cfg.context_length-16):];identity=f"{parent_sha}:{updates}";trajectories=rollout_group(model,prompt,ex.global_id,lambda ids:verify_candidate(ex,tok.decode(list(ids)).strip()),a.rollout_k,a.seed+cursor*a.rollout_k,getattr(tok,"eos_id",2),identity,bool(topology and topology.get("batched")))
   audit=trajectories if updates<2 or updates%rollout_audit_every==0 else trajectories[:1]
   if any(certify_trajectory(model,t,current_policy_sha256=identity)["status"]!="PASS" for t in audit):
    if any(certify_trajectory(model,t,current_policy_sha256=identity)["status"]!="PASS" for t in trajectories):raise RuntimeError("rollout parity gate failed")
   freeze_record(a.output,cursor,{"schema":"rlvr-rollout-group-v1","example_id":ex.global_id,"trajectories":[t.to_dict() for t in trajectories]})
   rewards=torch.tensor([t.reward for t in trajectories]);groups=torch.zeros(len(trajectories),dtype=torch.long);advantages=grpo_advantages(rewards,groups) if a.method=="rlvr-grpo" else rloo_advantages(rewards,groups);seqs=[];masks=[];targets=[]
   for t in trajectories:
    ids=list(t.prompt_ids)+list(t.generated_ids);seqs.append(ids[:-1]);targets.append(ids[1:]);masks.append([False]*(len(t.prompt_ids)-1)+list(t.completion_mask))
   width=max(map(len,seqs));pad=lambda z,v:z+[v]*(width-len(z));xt=torch.tensor([pad(x,0) for x in seqs]);yt=torch.tensor([pad(x,0) for x in targets]);mt=torch.tensor([pad(x,False) for x in masks]);logits=model(xt)[0];sampled=F.log_softmax(logits,-1).gather(-1,yt[...,None]).squeeze(-1);loss=policy_gradient_loss(sampled,mt,advantages);increment=int(mt.sum());processed_tokens+=sum(map(len,seqs))
  if not torch.isfinite(loss):raise FloatingPointError("nonfinite post-training loss")
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();last=float(loss.detach());tokens+=max(1,min(increment,a.tokens-tokens));updates+=1;cursor+=1
  a.processed_tokens=processed_tokens
  if tokens==a.tokens or tokens//a.checkpoint_tokens!=(tokens-increment)//a.checkpoint_tokens:save_checkpoint(a.output,payload(model,opt,a,cursor,tokens,updates,mixer,replay,parent_sha,last),True)
  if STOP or time.time()>=a.stop_epoch:
   save_checkpoint(a.output,payload(model,opt,a,cursor,tokens,updates,mixer,replay,parent_sha,last),True);status="SAFE_STOPPED_VALID";break
 else:status="COMPLETE"
 result={"schema":"posttraining-worker-result-v2","status":status,"method":a.method,"method_chain":a.method_chain,"checkpoint":str(a.output/"latest.pt"),"checkpoint_sha256":sha256(a.output/"latest.pt"),"parent_checkpoint_sha256":parent_sha,"training_tokens":tokens,"processed_tokens":processed_tokens,"updates":updates,"train_loss":last,"wall_seconds":time.perf_counter()-started,"optimizer":a.optimizer,"lr":a.lr,"replay":a.replay,"microbatch":a.microbatch,"compile_mode":a.compile_mode,"ranking_mode":a.ranking_mode,"natural_fraction":a.natural_fraction,"rank_weight":a.rank_weight,"margin_weight":a.margin_weight,"topology":topology};a.output.mkdir(parents=True,exist_ok=True);tmp=a.output/".result.json.tmp";tmp.write_text(json.dumps(result,indent=2)+"\n");os.replace(tmp,a.output/"result.json");print(json.dumps(result))
if __name__=="__main__":raise SystemExit(main() or 0)
