"""Train or exactly resume the controlled Co4-L DATA-D-BROAD-v2r1 lineage."""
from __future__ import annotations

import argparse, csv, json, math, random, resource, signal, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.data_d import ExactMixtureV2, Int32Shard, SourceStream, sha256_file, verify_top_manifest
from latticelm.model import build_model
from run_phase7c_final import evaluate, roll_checkpoint

ROOT=Path(__file__).resolve().parents[1];TOKENIZER=ROOT/"artifacts/tokenizers/babylm_2026_4k.json"
COMMON=ROOT/"artifacts/data/phase7a/common_validation.int32";CONFIG=ROOT/"configs/phase7d_co4_l.json"
SOURCES=("fineweb_edu","wikipedia","fineweb");TARGET=50_000_000;STOP_REQUESTED=False
FIELDS=["checkpoint","nominal_tokens","training_tokens","step","train_loss","common_validation_loss","common_validation_perplexity",
 "data_d_balanced_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss","cumulative_training_seconds",
 "interval_training_seconds","tokens_per_second","peak_rss_bytes","learning_rate","fineweb_edu_tokens","wikipedia_tokens","fineweb_tokens",
 "fineweb_edu_sequences_drawn","wikipedia_sequences_drawn","fineweb_sequences_drawn","checkpoint_sha256","preemptions"]


def load_arrays(base:Path,children:list[dict])->dict[str,list[np.ndarray]]:
 output={source:[] for source in SOURCES}
 for child in children:
  manifest=json.loads((base/child["manifest_path"]).read_text());shard=Int32Shard(base/manifest["path"],manifest)
  output[manifest["source"]].append(shard.tokens)
 return output


def validation_tensor(arrays:list[np.ndarray],limit:int|None=None)->torch.Tensor:
 values=np.concatenate([np.asarray(value) for value in arrays]);values=values[:limit] if limit else values
 return torch.from_numpy(values.astype(np.int64))


def masked_loss(model,x,y,increment):
 logits=model(x)[0];losses=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1),reduction="none").view(8,128)
 mask=torch.zeros_like(losses);per_row=increment//8
 if per_row*8!=increment:raise RuntimeError("terminal token count must preserve exact source mixture")
 mask[:,:per_row]=1
 return (losses*mask).sum()/increment


def state_payload(model,optimizer,scheduler,config,mixer,step,tokens,source_tokens,elapsed,last_eval,best,manifest_hash,preemptions):
 return {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"config":config.to_dict(),
  "source":"DATA-D-BROAD-v2r1","lineage":"co4-l-data-d-v2r1-25m","step":step,"tokens_seen":tokens,"source_tokens":source_tokens,
  "data_source_selector_state":mixer.state_dict(),"python_rng_state":random.getstate(),"torch_rng_state":torch.get_rng_state(),
  "cumulative_training_seconds":elapsed,"last_evaluation_seconds":last_eval,"best_validation_loss":best,"preemptions":preemptions,
  "data_manifest_sha256":manifest_hash}


def main():
 global STOP_REQUESTED
 def request_stop(*_):
  global STOP_REQUESTED;STOP_REQUESTED=True
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--fresh",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--stop-tokens",type=int);p.add_argument("--hard-deadline-epoch",type=float);a=p.parse_args()
 if a.fresh==a.resume:p.error("choose exactly one of --fresh or --resume")
 target=a.stop_tokens or TARGET
 if target>100_000_000:raise RuntimeError("controlled DATA-D-v2r1 lineage must not continue past its unique 100M corpus")
 manifest_path=Path(a.manifest).resolve();manifest=verify_top_manifest(manifest_path,TOKENIZER);base=manifest_path.parent
 certification=json.loads((base/"certification.json").read_text())
 if certification["acceptance_gates"]!="PASS" or certification["manifest_sha256"]!=sha256_file(manifest_path):raise RuntimeError("uncertified DATA-D corpus")
 config=LatticeConfig.from_json(CONFIG);torch.set_num_threads(config.num_threads);torch.set_num_interop_threads(1);random.seed(config.seed);torch.manual_seed(config.seed)
 train=load_arrays(base,manifest["shards"]);valid=load_arrays(base,manifest["validation_shards"])
 streams={source:SourceStream(train[source],128,config.seed+offset) for source,offset in zip(SOURCES,(11,23,37))};mixer=ExactMixtureV2(streams)
 vals={source:validation_tensor(valid[source]) for source in SOURCES};balanced=torch.cat([vals[source][:250_000] for source in SOURCES]);common=validation_tensor([np.memmap(COMMON,mode="r",dtype="<i4")])
 model=build_model(config)
 if model.parameter_breakdown()["total"]!=15_949_760:raise RuntimeError("Co4-L parameter identity mismatch")
 optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay,betas=(config.adam_beta1,config.adam_beta2),eps=1e-8)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
 recovery=ROOT/"artifacts/checkpoints/co4-l-data-d-v2r1-25m";latest=recovery/"latest.pt";curve=ROOT/"artifacts/data_d_v2r1_training_curve.csv"
 step=tokens=0;source_tokens={source:0 for source in SOURCES};prior=last_eval=0.;best=float("inf");preemptions=0
 if a.fresh and latest.exists():raise RuntimeError("refusing to overwrite existing DATA-D lineage")
 if a.resume:
  if not latest.exists() or sha256_file(latest)!=latest.with_suffix(".sha256").read_text().strip():raise RuntimeError("invalid recovery checkpoint")
  state=torch.load(latest,map_location="cpu",weights_only=False)
  if state["source"]!="DATA-D-BROAD-v2r1" or state["data_manifest_sha256"]!=sha256_file(manifest_path):raise RuntimeError("wrong resume lineage/manifest")
  model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
  random.setstate(state["python_rng_state"]);torch.set_rng_state(state["torch_rng_state"]);mixer.load_state_dict(state["data_source_selector_state"])
  step=int(state["step"]);tokens=int(state["tokens_seen"]);source_tokens={k:int(v) for k,v in state["source_tokens"].items()};prior=float(state["cumulative_training_seconds"]);last_eval=float(state["last_evaluation_seconds"]);best=float(state["best_validation_loss"]);preemptions=int(state.get("preemptions",0))+1
 if target<=tokens:raise RuntimeError("target must exceed recovered tokens")
 milestones={1_000_000,3_000_000,5_000_000,7_500_000,10_000_000,15_000_000,20_000_000,25_000_000,35_000_000,50_000_000,75_000_000,100_000_000};evaluated={x for x in milestones if x<=tokens}
 session_start=time.perf_counter();session_start_tokens=tokens;train_loss=float("nan")
 while tokens<target:
  x,y,_=mixer.batch();x=torch.from_numpy(x.astype(np.int64));y=torch.from_numpy(y.astype(np.int64));increment=min(1024,target-tokens)
  loss=model(x,y)[1] if increment==1024 else masked_loss(model,x,y,increment)
  if not torch.isfinite(loss):raise FloatingPointError("non-finite loss")
  optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip);optimizer.step();scheduler.step()
  step+=1;tokens+=increment;per=increment//8;source_tokens["fineweb_edu"]+=per*4;source_tokens["wikipedia"]+=per*2;source_tokens["fineweb"]+=per*2;train_loss=float(loss.detach())
  crossed=[m for m in milestones-evaluated if abs(tokens-m)<=512];deadline_stop=bool(a.hard_deadline_epoch and time.time()>=a.hard_deadline_epoch);evaluation=tokens==target or bool(crossed) or STOP_REQUESTED or deadline_stop;periodic=step%config.checkpoint_interval==0 or evaluation
  if not(periodic or evaluation):continue
  elapsed=prior+time.perf_counter()-session_start;payload=state_payload(model,optimizer,scheduler,config,mixer,step,tokens,source_tokens,elapsed,last_eval,best,sha256_file(manifest_path),preemptions)
  path,checksum=roll_checkpoint(recovery,payload,evaluation)
  if not evaluation:continue
  if STOP_REQUESTED or deadline_stop:print(json.dumps({"safe_stop":True,"tokens":tokens}),flush=True);return 75
  losses={source:evaluate(model,vals[source],8,128) for source in SOURCES};balanced_loss=evaluate(model,balanced,8,128);common_loss=evaluate(model,common,8,128);best=min(best,common_loss)
  elapsed=prior+time.perf_counter()-session_start;payload=state_payload(model,optimizer,scheduler,config,mixer,step,tokens,source_tokens,elapsed,elapsed,best,sha256_file(manifest_path),preemptions);path,checksum=roll_checkpoint(recovery,payload,True)
  nominal=target if tokens==target else min(crossed,key=lambda m:abs(tokens-m));evaluated.add(nominal)
  row={"checkpoint":f"co4-l-data-d-v2r1-{nominal}","nominal_tokens":nominal,"training_tokens":tokens,"step":step,"train_loss":train_loss,
   "common_validation_loss":common_loss,"common_validation_perplexity":math.exp(common_loss),"data_d_balanced_validation_loss":balanced_loss,
   **{f"{source}_validation_loss":losses[source] for source in SOURCES},"cumulative_training_seconds":elapsed,"interval_training_seconds":elapsed-last_eval,
   "tokens_per_second":(tokens-session_start_tokens)/max(elapsed-prior,1e-9),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
   "learning_rate":optimizer.param_groups[0]["lr"],**{f"{source}_tokens":source_tokens[source] for source in SOURCES},
   **{f"{source}_sequences_drawn":streams[source].draws for source in SOURCES},"checkpoint_sha256":checksum,"preemptions":preemptions}
  exists=curve.exists();curve.parent.mkdir(parents=True,exist_ok=True)
  with curve.open("a",newline="") as handle:
   writer=csv.DictWriter(handle,fieldnames=FIELDS,lineterminator="\n");writer.writeheader() if not exists else None;writer.writerow(row)
  print(json.dumps(row),flush=True);last_eval=elapsed
 expected={"fineweb_edu":target//2,"wikipedia":target//4,"fineweb":target//4}
 if tokens!=target or source_tokens!=expected:raise RuntimeError("exact terminal mixture failure")


if __name__=="__main__":main()
