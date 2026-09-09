"""Exact-resume 32.68M Co4 capacity/final-lineage training on DATA-D-v3."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,resource,signal,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from latticelm.config import LatticeConfig
from latticelm.data_d import ExactMixtureV2,Int32Shard,SourceStream,sha256_file,verify_top_manifest
from latticelm.model import build_model
from run_phase7c_final import evaluate,roll_checkpoint
ROOT=Path(__file__).resolve().parents[1];TOK=ROOT/"artifacts/tokenizers/babylm_2026_4k.json";CONFIG=ROOT/"configs/final_capacity32_co4.json";SOURCES=("fineweb_edu","wikipedia","fineweb");EXPECTED=32_678_640;STOP=False
FIELDS=["nominal_tokens","training_tokens","step","train_loss","data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss","cumulative_training_seconds","tokens_per_second","peak_rss_bytes","learning_rate","checkpoint_sha256","backend","threads","microbatch","preemptions"]
def arrays(base,children):
 out={s:[] for s in SOURCES}
 for child in children:
  m=json.loads((base/child["manifest_path"]).read_text());out[m["source"]].append(Int32Shard(base/m["path"],m).tokens)
 return out
def tensor(values,limit=None):
 x=np.concatenate([np.asarray(v) for v in values]);return torch.from_numpy(x[:limit].astype(np.int64))
def next_hash(mix):
 state=mix.state_dict();x,y,_=mix.batch();mix.load_state_dict(state);return hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
def payload(model,opt,sched,cfg,mix,run_id,step,tokens,elapsed,manifest,backend,microbatch,preemptions):
 return {"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"config":cfg.to_dict(),"source":"DATA-D-BROAD-v3","lineage":run_id,"run_id":run_id,"step":step,"tokens_seen":tokens,"data_source_selector_state":mix.state_dict(),"python_rng_state":random.getstate(),"torch_rng_state":torch.get_rng_state(),"cumulative_training_seconds":elapsed,"data_manifest_sha256":manifest,"next_batch_sha256":next_hash(mix),"backend":backend,"microbatch":microbatch,"preemptions":preemptions}
def main():
 global STOP
 signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__("STOP",True));signal.signal(signal.SIGINT,lambda *_:globals().__setitem__("STOP",True))
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--target",type=int,required=True);p.add_argument("--run-id",default="co4-final-capacity32-data-d-v3");p.add_argument("--config",type=Path,default=CONFIG);p.add_argument("--expected-parameters",type=int,default=EXPECTED);p.add_argument("--backend",choices=("fp32_eager","fp32_compile"),required=True);p.add_argument("--threads",type=int,required=True);p.add_argument("--microbatch",type=int,choices=(4,8),default=8);p.add_argument("--fresh",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--hard-deadline-epoch",type=float);p.add_argument("--curve",type=Path,default=ROOT/"artifacts/final_capacity_training_curve.csv");a=p.parse_args()
 if a.fresh==a.resume:p.error("choose exactly one of --fresh/--resume")
 top=verify_top_manifest(a.manifest,TOK);cert=json.loads((a.manifest.parent/"certification.json").read_text());mh=sha256_file(a.manifest)
 if a.target<=0 or a.target%8:raise RuntimeError("target must be a positive multiple of eight")
 if cert.get("acceptance")!="PASS" or cert.get("manifest_sha256")!=mh or top["total_unique_tokens"]<a.target:raise RuntimeError("uncertified or insufficient DATA-D-v3")
 cfg=LatticeConfig.from_json(a.config);torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);random.seed(cfg.seed);torch.manual_seed(cfg.seed)
 train=arrays(a.manifest.parent,top["shards"]);valid=arrays(a.manifest.parent,top["validation_shards"]);streams={s:SourceStream(train[s],128,cfg.seed+n) for s,n in zip(SOURCES,(11,23,37))};mix=ExactMixtureV2(streams);vals={s:tensor(valid[s],250_000) for s in SOURCES};balanced=torch.cat(list(vals.values()))
 model=build_model(cfg)
 if model.parameter_breakdown()["total"]!=a.expected_parameters:raise RuntimeError("final capacity parameter identity mismatch")
 if a.backend=="fp32_compile":model.compile(mode="max-autotune-no-cudagraphs",fullgraph=False)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay,betas=(cfg.adam_beta1,cfg.adam_beta2),eps=1e-8);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0);root=ROOT/"artifacts/checkpoints"/a.run_id;latest=root/"latest.pt";step=tokens=0;prior=0.;preemptions=0
 if a.fresh and latest.exists():raise RuntimeError("refusing existing lineage")
 if a.resume:
  choices=[root/x for x in ("latest.pt","previous.pt","fallback.pt")];found=next((x for x in choices if x.exists() and x.with_suffix(".sha256").exists() and sha256_file(x)==x.with_suffix(".sha256").read_text().strip()),None)
  if not found:raise RuntimeError("no valid checkpoint fallback")
  state=torch.load(found,map_location="cpu",weights_only=False)
  if state["lineage"]!=a.run_id or state["data_manifest_sha256"]!=mh or state["backend"]!=a.backend:raise RuntimeError("resume lineage mismatch")
  model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"]);sched.load_state_dict(state["scheduler"]);random.setstate(state["python_rng_state"]);torch.set_rng_state(state["torch_rng_state"]);mix.load_state_dict(state["data_source_selector_state"])
  if next_hash(mix)!=state["next_batch_sha256"]:raise RuntimeError("next-batch mismatch")
  step=int(state["step"]);tokens=int(state["tokens_seen"]);prior=float(state["cumulative_training_seconds"]);preemptions=int(state.get("preemptions",0))+1
 if a.target<=tokens:raise RuntimeError("target already reached")
 started=time.perf_counter();start_tokens=tokens;loss_value=float("nan")
 while tokens<a.target:
  x,y,_=mix.batch();x=torch.from_numpy(x.astype(np.int64));y=torch.from_numpy(y.astype(np.int64));increment=min(1024,a.target-tokens);opt.zero_grad(set_to_none=True)
  per_row=increment//8
  if increment%8:raise RuntimeError("terminal target must preserve exact source mixture")
  loss=torch.zeros(())
  for off in range(0,8,a.microbatch):
   logits,_=model(x[off:off+a.microbatch]);parts=F.cross_entropy(logits.reshape(-1,4096),y[off:off+a.microbatch].reshape(-1),reduction="none").view(a.microbatch,128);part=parts[:,:per_row].sum()/increment;part.backward();loss=loss+part.detach()
  if not torch.isfinite(loss) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError("non-finite training state")
  torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();sched.step();step+=1;tokens+=increment;loss_value=float(loss.detach());elapsed=prior+time.perf_counter()-started
  terminal=tokens==a.target;deadline=bool(a.hard_deadline_epoch and time.time()>=a.hard_deadline_epoch-900);periodic=step%cfg.checkpoint_interval==0 or terminal or STOP or deadline
  if periodic:_,checksum=roll_checkpoint(root,payload(model,opt,sched,cfg,mix,a.run_id,step,tokens,elapsed,mh,a.backend,a.microbatch,preemptions),terminal)
  if STOP or deadline:print(json.dumps({"safe_stop":True,"tokens":tokens}));return 75
  if not terminal:continue
  losses={s:evaluate(model,vals[s],8,128) for s in SOURCES};balanced_loss=evaluate(model,balanced,8,128);elapsed=prior+time.perf_counter()-started;_,checksum=roll_checkpoint(root,payload(model,opt,sched,cfg,mix,a.run_id,step,tokens,elapsed,mh,a.backend,a.microbatch,preemptions),True)
  row={"nominal_tokens":a.target,"training_tokens":tokens,"step":step,"train_loss":loss_value,"data_d_validation_loss":balanced_loss,**{f"{s}_validation_loss":losses[s] for s in SOURCES},"cumulative_training_seconds":elapsed,"tokens_per_second":(tokens-start_tokens)/max(elapsed-prior,1e-9),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"learning_rate":opt.param_groups[0]["lr"],"checkpoint_sha256":checksum,"backend":a.backend,"threads":a.threads,"microbatch":a.microbatch,"preemptions":preemptions};exists=a.curve.exists();a.curve.parent.mkdir(parents=True,exist_ok=True)
  with a.curve.open("a",newline="") as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader() if not exists else None;w.writerow(row)
  print(json.dumps(row),flush=True)
if __name__=="__main__":raise SystemExit(main())
