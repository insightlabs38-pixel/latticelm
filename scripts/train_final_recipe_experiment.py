"""Exact-resume from-scratch trainer for bounded final-recipe experiments."""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, resource, signal, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from latticelm.config import LatticeConfig
from latticelm.data_d import Int32Shard, SourceStream, sha256_file, verify_top_manifest
from latticelm.final_recipe import make_optimizer, wsd_lr_scale
from latticelm.model import build_model

ROOT=Path(__file__).resolve().parents[1]; STOP=False


class WeightedMixture:
    def __init__(self, streams, weights, batch_size):
        self.streams=dict(streams); self.counter=0
        counts={s:round(w*20) for s,w in weights.items()}; delta=20-sum(counts.values());counts[next(iter(counts))]+=delta
        self.cycle=tuple(s for s in weights for _ in range(counts[s]));self.batch_size=batch_size
    def batch(self):
        labels=tuple(self.cycle[(self.counter+i)%len(self.cycle)] for i in range(self.batch_size));self.counter+=self.batch_size
        pairs=[self.streams[s].one() for s in labels];return np.stack([x for x,_ in pairs]),np.stack([y for _,y in pairs]),labels
    def state_dict(self):return {"counter":self.counter,"draws":{s:x.draws for s,x in self.streams.items()}}
    def load_state_dict(self,x):
        self.counter=int(x["counter"])
        for s,v in self.streams.items():v.draws=int(x["draws"][s])


def next_hash(mix):
    state=mix.state_dict();x,y,_=mix.batch();mix.load_state_dict(state);return hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()


def atomic_checkpoint(root,payload,terminal=False):
    root.mkdir(parents=True,exist_ok=True);latest=root/"latest.pt";previous=root/"previous.pt";partial=root/".latest.pt.partial"
    torch.save(payload,partial);checksum=sha256_file(partial);partial.with_suffix(".sha256").write_text(checksum+"\n")
    if latest.exists():os.replace(latest,previous)
    if latest.with_suffix(".sha256").exists():os.replace(latest.with_suffix(".sha256"),previous.with_suffix(".sha256"))
    os.replace(partial,latest);os.replace(partial.with_suffix(".sha256"),latest.with_suffix(".sha256"))
    if terminal:
        milestone=root/"milestone.pt"
        if not milestone.exists(): os.link(latest,milestone);milestone.with_suffix(".sha256").write_text(checksum+"\n")
    return checksum


def arrays(manifest):
    top=json.loads(manifest.read_text());base=manifest.parent;train={s:[] for s in top["mixture_definition"]};valid={s:[] for s in train}
    for field,target in (("shards",train),("validation_shards",valid)):
        for child in top[field]:
            m=json.loads((base/child["manifest_path"]).read_text());target[m["source"]].append(Int32Shard(base/m["path"],m).tokens)
    return top,train,valid


def evaluate(model, values, context, batch=4):
    model.eval();losses=[]
    with torch.inference_mode():
        for start in range(0,min(len(values)-context-1,8*batch*context),batch*context):
            chunks=[np.asarray(values[start+i*context:start+(i+1)*context+1]) for i in range(batch)]
            x=torch.tensor(np.stack([z[:-1] for z in chunks]),dtype=torch.long);y=torch.tensor(np.stack([z[1:] for z in chunks]),dtype=torch.long)
            _,loss=model(x,y);losses.append(float(loss))
    model.train();return sum(losses)/len(losses)


def run(a):
    global STOP
    cfg=LatticeConfig.from_json(a.config);torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    top=verify_top_manifest(a.manifest,a.tokenizer);manifest_hash=sha256_file(a.manifest);top,train,valid=arrays(a.manifest)
    streams={s:SourceStream(train[s],cfg.context_length,cfg.seed+17*i) for i,s in enumerate(train)};mix=WeightedMixture(streams,top["mixture_definition"],cfg.batch_size)
    model=build_model(cfg);params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not params<50_000_000:raise RuntimeError(f"parameter cap exceeded: {params}")
    if a.backend=="compile":model.compile(mode="max-autotune-no-cudagraphs",fullgraph=False)
    optimizer,partition=make_optimizer(model,cfg.optimizer,cfg.learning_rate,cfg.weight_decay,(cfg.adam_beta1,cfg.adam_beta2));base_lrs=[g["lr"] for g in optimizer.param_groups]
    exp=ROOT/"artifacts/final_recipe_experiments"/a.experiment;result_path=exp/"result.json";latest=exp/"latest.pt"
    if result_path.exists():return json.loads(result_path.read_text())
    step=tokens=0;prior=0.;best=float("inf");preemptions=0;loss_value=float("nan");checksum=""
    if a.fresh and latest.exists():raise RuntimeError("immutable experiment directory already exists")
    if a.resume:
        found=next((p for p in (latest,exp/"previous.pt") if p.exists() and p.with_suffix(".sha256").exists() and sha256_file(p)==p.with_suffix(".sha256").read_text().strip()),None)
        if not found:raise RuntimeError("no valid resume checkpoint")
        ck=torch.load(found,map_location="cpu",weights_only=False)
        if ck["config"]!=cfg.to_dict() or ck["manifest_sha256"]!=manifest_hash or ck["tokenizer_sha256"]!=sha256_file(a.tokenizer):raise RuntimeError("resume identity mismatch")
        model.load_state_dict(ck["model"],strict=True);optimizer.load_state_dict(ck["optimizers"]);mix.load_state_dict(ck["mixture"]);random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"]);torch.set_rng_state(ck["torch_rng"])
        if next_hash(mix)!=ck["next_batch_sha256"]:raise RuntimeError("exact resume next-batch mismatch")
        step=int(ck["step"]);tokens=int(ck["tokens"]);prior=float(ck["training_seconds"]);best=float(ck["best_validation"]);preemptions=int(ck["preemptions"])+1;loss_value=float(ck.get("train_loss",float("nan")));checksum=sha256_file(found)
    step_tokens=cfg.batch_size*cfg.context_length;total_steps=math.ceil(a.target_tokens/step_tokens);started=time.perf_counter();exp.mkdir(parents=True,exist_ok=True)
    log=exp/"train.jsonl"
    while tokens<a.target_tokens:
        step+=1;scale=wsd_lr_scale(step,total_steps,cfg.warmup_fraction,cfg.stable_fraction,cfg.decay_fraction)
        for group,base_lr in zip(optimizer.param_groups,base_lrs):group["lr"]=base_lr*scale
        x,y,_=mix.batch();x=torch.tensor(x,dtype=torch.long);y=torch.tensor(y,dtype=torch.long);increment=min(step_tokens,a.target_tokens-tokens)
        optimizer.zero_grad();logits,_=model(x);loss=F.cross_entropy(logits.reshape(-1,cfg.vocab_size)[:increment],y.reshape(-1)[:increment]);loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError(f"nonfinite at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);optimizer.step();tokens+=increment;loss_value=float(loss.detach());elapsed=prior+time.perf_counter()-started
        with log.open("a") as f:f.write(json.dumps({"step":step,"tokens":tokens,"loss":loss_value,"lr_scale":scale,"seconds":elapsed})+"\n")
        checkpoint_due=tokens==a.target_tokens or tokens//a.checkpoint_tokens != (tokens-increment)//a.checkpoint_tokens or STOP
        if checkpoint_due:
            vals={s:evaluate(model,np.concatenate(v),cfg.context_length) for s,v in valid.items()};best=min(best,sum(vals.values())/len(vals))
            payload={"model":model.state_dict(),"optimizers":optimizer.state_dict(),"config":cfg.to_dict(),"step":step,"tokens":tokens,"train_loss":loss_value,"training_seconds":elapsed,"best_validation":best,"mixture":mix.state_dict(),"next_batch_sha256":next_hash(mix),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"manifest_sha256":manifest_hash,"tokenizer_sha256":sha256_file(a.tokenizer),"preemptions":preemptions,"optimizer_partition":partition}
            checksum=atomic_checkpoint(exp,payload,tokens==a.target_tokens)
        if STOP:return {"status":"STOPPED_SAFE","tokens":tokens}
    wall=prior+time.perf_counter()-started;vals={s:evaluate(model,np.concatenate(v),cfg.context_length) for s,v in valid.items()}
    result={"schema":"final-recipe-experiment-v1","status":"VALID","experiment_id":a.experiment,"parent_decision":a.parent_decision,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"config":cfg.to_dict(),"config_sha256":hashlib.sha256(json.dumps(cfg.to_dict(),sort_keys=True).encode()).hexdigest(),"tokenizer":str(a.tokenizer),"tokenizer_sha256":sha256_file(a.tokenizer),"manifest":str(a.manifest),"manifest_sha256":manifest_hash,"seed":cfg.seed,"parameter_count":params,"hardware":"GCP c4a-standard-16","thread_count":a.threads,"pytorch_version":torch.__version__,"compile_settings":a.backend,"optimizer":cfg.optimizer,"optimizer_partition":partition,"training_tokens":tokens,"checkpoint":str(exp/"milestone.pt"),"checkpoint_sha256":checksum,"elapsed_seconds":wall,"tokens_per_second":tokens/wall,"train_loss":loss_value,"source_validation":vals,"validation_loss":sum(vals.values())/len(vals),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"classification":"MODELING_RESULT"}
    tmp=result_path.with_name(".result.json.tmp");tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");os.replace(tmp,result_path);return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True);p.add_argument("--experiment",required=True);p.add_argument("--parent-decision",required=True);p.add_argument("--target-tokens",type=int,required=True);p.add_argument("--threads",type=int,default=16);p.add_argument("--backend",choices=("eager","compile"),default="compile");p.add_argument("--checkpoint-tokens",type=int,default=10_000_000);g=p.add_mutually_exclusive_group(required=True);g.add_argument("--fresh",action="store_true");g.add_argument("--resume",action="store_true");a=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__("STOP",True));signal.signal(signal.SIGINT,lambda *_:globals().__setitem__("STOP",True));print(json.dumps(run(a),indent=2))
if __name__=="__main__":main()
