"""Exact-resume Co4-L DATA-D/LatticeReason base-pretraining tournament worker."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,resource,signal,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.data_d import Int32Shard,SourceStream,sha256_file,verify_top_manifest
from latticelm.lattice_reason.sampling import TokenBalancedReasonStream,TournamentMixture
from latticelm.model import build_model
from latticelm.tokenizer import load_tokenizer
from run_phase7c_final import evaluate,roll_checkpoint

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";TOKENIZER=ART/"tokenizers/babylm_2026_4k.json"
CONFIG=ROOT/"configs/phase7d_co4_l.json";COMMON=ART/"data/phase7a/common_validation.int32";FAMILIES=("entity_reference","state_tracking","spatial_relational","temporal_causal","boolean_logical","arithmetic_quantity","procedural_planning","physical_closed_world")
STOP_REQUESTED=False

def request_stop(*_):
 global STOP_REQUESTED;STOP_REQUESTED=True

def arrays(base,children):
 out={s:[] for s in ("fineweb_edu","wikipedia","fineweb")}
 for child in children:
  manifest=json.loads((base/child["manifest_path"]).read_text());out[manifest["source"]].append(Int32Shard(base/manifest["path"],manifest).tokens)
 return out

def tensor(values,limit=None):
 a=np.concatenate([np.asarray(x) for x in values]);a=a[:limit] if limit else a
 return torch.from_numpy(a.astype(np.int64))

def lr_validation():
 tok=load_tokenizer(TOKENIZER);all_rows=[];groups={}
 for line in (ART/"data/lattice_reason/validation/validation.jsonl").read_text().splitlines():
  row=json.loads(line);tokens=tok.encode(row["rendered_text"]);all_rows.extend(tokens)
  for key in (f"family_{row['family']}",f"difficulty_{row['difficulty']}"):groups.setdefault(key,[]).extend(tokens)
 return torch.tensor(all_rows,dtype=torch.long),{k:torch.tensor(v,dtype=torch.long) for k,v in groups.items()}

def masked_loss(model,x,y,increment):
 logits=model(x)[0];losses=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),reduction="none").view(8,128)
 mask=torch.zeros_like(losses);mask.view(-1)[:increment]=1
 return (losses*mask).sum()/increment

def next_digest(mixer):
 state=mixer.state_dict();x,y,labels,families=mixer.batch();digest=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest();mixer.load_state_dict(state)
 return digest

def payload(model,opt,sched,config,mixer,run_id,percent,step,tokens,source_tokens,family_tokens,elapsed,manifest_hash,preemptions):
 return {"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"config":config.to_dict(),"run_id":run_id,
  "source":"DATA-D-BROAD-v2r1+LatticeReason-v1","mixture":{"data_d_percent":100-percent,"lattice_reason_percent":percent},"step":step,
  "tokens_seen":tokens,"source_tokens":dict(source_tokens),"lattice_reason_family_tokens":dict(family_tokens),"mixture_scheduler_state":mixer.state_dict(),
  "python_rng_state":random.getstate(),"torch_rng_state":torch.get_rng_state(),"cumulative_training_seconds":elapsed,
  "data_manifest_sha256":manifest_hash,"tokenizer_sha256":sha256_file(TOKENIZER),"next_batch_sha256":next_digest(mixer),"preemptions":preemptions}

def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--run-id",required=True);p.add_argument("--lr-percent",type=int,required=True,choices=(5,10,15,25,50));p.add_argument("--target",type=int,required=True);p.add_argument("--config",default=str(CONFIG));p.add_argument("--expected-parameters",type=int,default=15_949_760);p.add_argument("--hard-deadline-epoch",type=float);p.add_argument("--fresh",action="store_true");p.add_argument("--resume",action="store_true");a=p.parse_args()
 if a.fresh==a.resume:p.error("choose exactly one of --fresh/--resume")
 manifest_path=Path(a.manifest).resolve();manifest=verify_top_manifest(manifest_path,TOKENIZER);base=manifest_path.parent
 cert=json.loads((base/"certification.json").read_text());manifest_hash=sha256_file(manifest_path)
 if cert["acceptance_gates"]!="PASS" or cert["manifest_sha256"]!=manifest_hash:raise RuntimeError("uncertified DATA-D")
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 config=LatticeConfig.from_json(a.config);torch.set_num_threads(config.num_threads);torch.set_num_interop_threads(1);random.seed(config.seed);torch.manual_seed(config.seed)
 train=arrays(base,manifest["shards"]);valid=arrays(base,manifest["validation_shards"])
 streams={s:SourceStream(train[s],128,config.seed+o) for s,o in zip(train,(11,23,37))};reason=TokenBalancedReasonStream(load_tokenizer(TOKENIZER),128,7319);mixer=TournamentMixture(streams,reason,a.lr_percent)
 vals={s:tensor(valid[s]) for s in valid};balanced=torch.cat([vals[s][:250_000] for s in vals]);common=tensor([np.memmap(COMMON,mode="r",dtype="<i4")]);lr_val,lr_groups=lr_validation()
 model=build_model(config)
 if model.parameter_breakdown()["total"]!=a.expected_parameters:raise RuntimeError("parameter identity mismatch")
 opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,betas=(config.adam_beta1,config.adam_beta2),eps=1e-8);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
 recovery=ART/"checkpoints"/a.run_id;latest=recovery/"latest.pt";curve=ART/"lattice_reason_training_curves.csv"
 step=tokens=0;source_tokens=Counter();family_tokens=Counter();prior=0.;preemptions=0
 if a.fresh and latest.exists():raise RuntimeError("refusing to overwrite lineage")
 if a.resume:
  if not latest.exists() or sha256_file(latest)!=latest.with_suffix(".sha256").read_text().strip():raise RuntimeError("invalid recovery checkpoint")
  state=torch.load(latest,map_location="cpu",weights_only=False)
  if state["run_id"]!=a.run_id or state["mixture"]["lattice_reason_percent"]!=a.lr_percent or state["data_manifest_sha256"]!=manifest_hash:raise RuntimeError("resume identity mismatch")
  model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);sched.load_state_dict(state["scheduler"]);mixer.load_state_dict(state["mixture_scheduler_state"])
  random.setstate(state["python_rng_state"]);torch.set_rng_state(state["torch_rng_state"])
  if next_digest(mixer)!=state["next_batch_sha256"]:raise RuntimeError("next-batch resume identity failure")
  step=int(state["step"]);tokens=int(state["tokens_seen"]);source_tokens.update(state["source_tokens"]);family_tokens.update(state["lattice_reason_family_tokens"]);prior=float(state["cumulative_training_seconds"]);preemptions=int(state.get("preemptions",0))+1
 if a.target<=tokens:raise RuntimeError("target must exceed current tokens")
 milestones={5_000_000,10_000_000,15_000_000,20_000_000,25_000_000,50_000_000,75_000_000,100_000_000,125_000_000,150_000_000,175_000_000,200_000_000};seen={m for m in milestones if m<=tokens};started=time.perf_counter();session_tokens=tokens;loss_value=float("nan")
 while tokens<a.target:
  x,y,labels,families=mixer.batch();increment=min(1024,a.target-tokens)
  # np.stack may preserve int32 when every selected DATA-D row is int32.
  # Cross entropy requires int64 targets regardless of mixture composition.
  x=torch.as_tensor(x,dtype=torch.long);y=torch.as_tensor(y,dtype=torch.long)
  loss=model(x,y)[1] if increment==1024 else masked_loss(model,x,y,increment)
  if not torch.isfinite(loss):raise FloatingPointError("non-finite loss")
  opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip);opt.step();sched.step()
  step+=1;tokens+=increment;loss_value=float(loss.detach());sc,fc=mixer.realized(labels,families,increment);source_tokens.update(sc);family_tokens.update(fc)
  crossed=[m for m in milestones-seen if tokens>=m];deadline_stop=bool(a.hard_deadline_epoch and time.time()>=a.hard_deadline_epoch);evaluation=tokens==a.target or bool(crossed) or STOP_REQUESTED or deadline_stop;periodic=step%196==0 or evaluation
  if not(periodic or evaluation):continue
  elapsed=prior+time.perf_counter()-started;state=payload(model,opt,sched,config,mixer,a.run_id,a.lr_percent,step,tokens,source_tokens,family_tokens,elapsed,manifest_hash,preemptions);path,checksum=roll_checkpoint(recovery,state,evaluation)
  if not evaluation:continue
  if STOP_REQUESTED or deadline_stop:print(json.dumps({"safe_stop":True,"tokens":tokens}),flush=True);return 75
  metrics={"common_validation_loss":evaluate(model,common,8,128),"data_d_validation_loss":evaluate(model,balanced,8,128),"lattice_reason_validation_loss":evaluate(model,lr_val,8,128),**{f"lattice_reason_{k}_validation_loss":evaluate(model,v,8,128) for k,v in lr_groups.items()},**{f"{s}_validation_loss":evaluate(model,v,8,128) for s,v in vals.items()}}
  row={"run_id":a.run_id,"lr_fraction":a.lr_percent/100,"training_tokens":tokens,"data_d_tokens":source_tokens["DATA-D"],"lattice_reason_tokens":source_tokens["LatticeReason"],"step":step,"train_loss":loss_value,**metrics,"tokens_per_second":(tokens-session_tokens)/max(elapsed-prior,1e-9),"wall_seconds":elapsed,"peak_ram_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"checkpoint_sha":checksum,**{f"lr_{f}_tokens":family_tokens[f] for f in FAMILIES}}
  curve.parent.mkdir(exist_ok=True);exists=curve.exists()
  with curve.open("a",newline="") as h:
   w=csv.DictWriter(h,fieldnames=list(row),lineterminator="\n");w.writeheader() if not exists else None;w.writerow(row)
  print(json.dumps(row),flush=True);seen.update(crossed)
 if tokens!=a.target or sum(source_tokens.values())!=tokens or sum(family_tokens.values())!=source_tokens["LatticeReason"]:raise RuntimeError("terminal token accounting failure")

if __name__=="__main__":main()
