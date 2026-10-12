"""Exact-restart continuation after a declared DATA-D-v3 -> v4 transition."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,resource,signal,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from latticelm.config import LatticeConfig
from latticelm.data_d import ExactMixtureV2,Int32Shard,SourceStream,sha256_file,verify_top_manifest
from latticelm.model import build_model
try:from run_phase7c_final import evaluate,roll_checkpoint
except ModuleNotFoundError:from scripts.run_phase7c_final import evaluate,roll_checkpoint

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";TOK=ART/"tokenizers/babylm_2026_4k.json";CONFIG=ROOT/"configs/final_capacity32_co4.json"
SOURCES=("fineweb_edu","wikipedia","fineweb");EXPECTED=32_678_640;BASE_TOKENS=900_000_000;STOP=False
FIELDS=["nominal_tokens","training_tokens","step","train_loss","data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss","cumulative_training_seconds","session_training_seconds","tokens_per_second","peak_rss_bytes","learning_rate","checkpoint_sha256","backend","threads","microbatch","preemptions","data_manifest_sha256","dataset_transition"]
def arrays(base,children):
 out={s:[] for s in SOURCES}
 for child in children:
  m=json.loads((base/child["manifest_path"]).read_text());out[m["source"]].append(Int32Shard(base/m["path"],m).tokens)
 return out
def tensor(values,limit=None):
 x=np.concatenate([np.asarray(v) for v in values]);return torch.from_numpy(x[:limit].astype(np.int64))
def next_hash(mix):
 state=mix.state_dict();x,y,_=mix.batch();mix.load_state_dict(state);return hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
def payload(model,opt,sched,cfg,mix,run_id,step,tokens,elapsed,manifest,microbatch,preemptions,transition):
 return {"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"config":cfg.to_dict(),"source":"DATA-D-BROAD-v4","lineage":run_id,"run_id":run_id,"parent_lineage":"co4-final-capacity32-data-d-v3","step":step,"tokens_seen":tokens,"data_source_selector_state":mix.state_dict(),"python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state(),"cumulative_training_seconds":elapsed,"data_manifest_sha256":manifest,"next_batch_sha256":next_hash(mix),"backend":"fp32_compile","microbatch":microbatch,"preemptions":preemptions,"dataset_transition":transition}
def valid_checkpoint(root):
 for name in ("latest.pt","previous.pt","fallback.pt"):
  p=root/name
  if p.exists() and p.with_suffix(".sha256").exists() and sha256_file(p)==p.with_suffix(".sha256").read_text().strip():return p
 return None
def main():
 global STOP
 signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__("STOP",True));signal.signal(signal.SIGINT,lambda *_:globals().__setitem__("STOP",True))
 p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--base-checkpoint",type=Path,required=True);p.add_argument("--transition",type=Path,required=True);p.add_argument("--target",type=int,required=True);p.add_argument("--run-id",default="co4-final-capacity32-data-d-v4");p.add_argument("--threads",type=int,default=16);p.add_argument("--microbatch",type=int,choices=(4,8),default=8);p.add_argument("--curve",type=Path,default=ART/"post900m_continuation_curve.csv");a=p.parse_args()
 if a.target<=BASE_TOKENS or a.target%8:raise RuntimeError("target must exceed 900M and preserve batch divisibility")
 top=verify_top_manifest(a.manifest,TOK);cert=json.loads((a.manifest.parent/"certification.json").read_text());mh=sha256_file(a.manifest);transition=json.loads(a.transition.read_text())
 if cert.get("acceptance")!="PASS" or cert.get("manifest_sha256")!=mh or top["total_unique_tokens"]<a.target-BASE_TOKENS:raise RuntimeError("uncertified or insufficient DATA-D-v4")
 if transition.get("new_manifest_sha256")!=mh or transition.get("token_count_at_transition")!=BASE_TOKENS:raise RuntimeError("dataset transition mismatch")
 cfg=LatticeConfig.from_json(CONFIG);torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);train=arrays(a.manifest.parent,top["shards"]);valid=arrays(a.manifest.parent,top["validation_shards"]);streams={s:SourceStream(train[s],128,cfg.seed+n+4000) for s,n in zip(SOURCES,(11,23,37))};mix=ExactMixtureV2(streams);vals={s:tensor(valid[s],250_000) for s in SOURCES};balanced=torch.cat(list(vals.values()));model=build_model(cfg)
 if model.parameter_breakdown()["total"]!=EXPECTED:raise RuntimeError("parameter identity mismatch")
 random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);model.compile(mode="max-autotune-no-cudagraphs",fullgraph=False);opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay,betas=(cfg.adam_beta1,cfg.adam_beta2),eps=1e-8);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
 root=ART/"checkpoints"/a.run_id;resume=valid_checkpoint(root);step=0;tokens=BASE_TOKENS;prior=0.;preemptions=0
 if resume:
  state=torch.load(resume,map_location="cpu",weights_only=False)
  if state.get("lineage")!=a.run_id or state.get("data_manifest_sha256")!=mh or state.get("config")!=cfg.to_dict() or state.get("dataset_transition")!=transition:raise RuntimeError("continuation resume identity mismatch")
  model.load_state_dict(state["model"],strict=True);opt.load_state_dict(state["optimizer"]);sched.load_state_dict(state["scheduler"]);random.setstate(state["python_rng_state"]);np.random.set_state(state["numpy_rng_state"]);torch.set_rng_state(state["torch_rng_state"]);mix.load_state_dict(state["data_source_selector_state"])
  if next_hash(mix)!=state["next_batch_sha256"]:raise RuntimeError("next batch mismatch")
  step=int(state["step"]);tokens=int(state["tokens_seen"]);prior=float(state["cumulative_training_seconds"]);preemptions=int(state.get("preemptions",0))+1
 else:
  base=torch.load(a.base_checkpoint,map_location="cpu",weights_only=False)
  if base.get("tokens_seen")!=BASE_TOKENS or sha256_file(a.base_checkpoint)!=transition.get("checkpoint_sha256") or base.get("config")!=cfg.to_dict():raise RuntimeError("900M transition base mismatch")
  model.load_state_dict(base["model"],strict=True);opt.load_state_dict(base["optimizer"]);sched.load_state_dict(base["scheduler"]);random.setstate(base["python_rng_state"]);np.random.set_state(base["numpy_rng_state"]);torch.set_rng_state(base["torch_rng_state"]);step=int(base["step"]);prior=float(base["cumulative_training_seconds"])
 if a.target<=tokens:raise RuntimeError("target already reached")
 started=time.perf_counter();session_start=tokens;loss_value=float("nan")
 while tokens<a.target:
  x,y,_=mix.batch();x=torch.from_numpy(x.astype(np.int64));y=torch.from_numpy(y.astype(np.int64));increment=min(1024,a.target-tokens);opt.zero_grad(set_to_none=True);per_row=increment//8
  if increment%8:raise RuntimeError("terminal target violates mixture")
  loss=torch.zeros(())
  for off in range(0,8,a.microbatch):
   logits,_=model(x[off:off+a.microbatch]);parts=F.cross_entropy(logits.reshape(-1,4096),y[off:off+a.microbatch].reshape(-1),reduction="none").view(a.microbatch,128);part=parts[:,:per_row].sum()/increment;part.backward();loss+=part.detach()
  if not torch.isfinite(loss) or not all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters()):raise FloatingPointError("non-finite continuation")
  torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();sched.step();step+=1;tokens+=increment;loss_value=float(loss);elapsed=prior+time.perf_counter()-started;terminal=tokens==a.target
  if step%cfg.checkpoint_interval==0 or terminal or STOP:roll_checkpoint(root,payload(model,opt,sched,cfg,mix,a.run_id,step,tokens,elapsed,mh,a.microbatch,preemptions,transition),terminal)
  if STOP:return 75
 if terminal:
  losses={s:evaluate(model,vals[s],8,128) for s in SOURCES};balanced_loss=evaluate(model,balanced,8,128);elapsed=prior+time.perf_counter()-started;_,digest=roll_checkpoint(root,payload(model,opt,sched,cfg,mix,a.run_id,step,tokens,elapsed,mh,a.microbatch,preemptions,transition),True);row={"nominal_tokens":a.target,"training_tokens":tokens,"step":step,"train_loss":loss_value,"data_d_validation_loss":balanced_loss,**{f"{s}_validation_loss":losses[s] for s in SOURCES},"cumulative_training_seconds":elapsed,"session_training_seconds":elapsed-prior,"tokens_per_second":(tokens-session_start)/max(elapsed-prior,1e-9),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"learning_rate":opt.param_groups[0]["lr"],"checkpoint_sha256":digest,"backend":"fp32_compile","threads":a.threads,"microbatch":a.microbatch,"preemptions":preemptions,"data_manifest_sha256":mh,"dataset_transition":sha256_file(a.transition)};exists=a.curve.exists();a.curve.parent.mkdir(parents=True,exist_ok=True)
  with a.curve.open("a",newline="") as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader() if not exists else None;w.writerow(row)
  print(json.dumps(row),flush=True)
 return 0
if __name__=="__main__":raise SystemExit(main())
