"""Bounded immutable post-900M experiment worker.

WSD is implemented as exact DATA-D-v3 continuation from the saved selector and
optimizer. SFT uses only deterministic LatticeReason examples. RLVR currently
records a conservative no-candidate result unless/until its verifier-policy
implementation passes a separate trajectory-parity gate; it never disguises SFT
as reinforcement learning.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from latticelm.config import LatticeConfig
from latticelm.data_d import ExactMixtureV2,Int32Shard,SourceStream,sha256_file,verify_top_manifest
from latticelm.lattice_reason.core import FAMILIES,generate_example,verify_example
from latticelm.model import build_model
from latticelm.tokenizer import load_tokenizer
from run_phase7c_final import roll_checkpoint

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";MANIFEST=ART/"data/data_d_v3/canonical-1b/manifest.json";TOK=ART/"tokenizers/babylm_2026_4k.json";STOP=False
def stop(*_):globals().__setitem__("STOP",True)
def arrays(base,children):
    out={s:[] for s in ("fineweb_edu","wikipedia","fineweb")}
    for child in children:
        m=json.loads((base/child["manifest_path"]).read_text());out[m["source"]].append(Int32Shard(base/m["path"],m).tokens)
    return out
def digest_next(mix):
    state=mix.state_dict();x,y,_=mix.batch();mix.load_state_dict(state);return hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
def immutable(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if sha256_file(path)!=path.with_suffix(".sha256").read_text().strip():raise RuntimeError("immutable branch hash mismatch")
        return
    tmp=path.with_name("."+path.name+".tmp");torch.save(payload,tmp);tmp.replace(path);path.with_suffix(".sha256").write_text(sha256_file(path)+"\n")
def eval_all(cp:Path,candidate_id:str,validation_loss:float,reasoning:float,evidence_dir:Path|None=None)->dict:
    root=evidence_dir or cp.parent;root.mkdir(parents=True,exist_ok=True);wiki=root/"wikitext.json";gibc=root/"gibc.json"
    if not wiki.exists():subprocess.run([sys.executable,"scripts/evaluate_wikitext103.py","--checkpoint",str(cp),"--tokenizer",str(TOK),"--output",str(wiki),"--threads","4"],cwd=ROOT,check=True)
    if not gibc.exists():subprocess.run([sys.executable,"scripts/evaluate_gibc.py","--checkpoint",str(cp),"--tokenizer",str(TOK),"--output",str(gibc),"--tasks","hellaswag,arc_easy,piqa,winogrande","--threads","4"],cwd=ROOT,check=True)
    w=json.loads(wiki.read_text());g=json.loads(gibc.read_text())["results"]
    return {"candidate_id":candidate_id,"checkpoint":str(cp),"checkpoint_sha256":sha256_file(cp),"validation_loss":validation_loss,"reasoning_accuracy":reasoning,"wikitext_ppl":w["perplexity"],"wikitext_bpb":w["bits_per_byte"],**{t:g[t]["acc,none"] for t in ("hellaswag","arc_easy","piqa","winogrande")}}
def data_setup(cfg):
    top=verify_top_manifest(MANIFEST,TOK);raw=arrays(MANIFEST.parent,top["shards"]);valid=arrays(MANIFEST.parent,top["validation_shards"]);streams={s:SourceStream(raw[s],128,cfg.seed+n) for s,n in zip(raw,(11,23,37))};vals=torch.from_numpy(np.concatenate([np.asarray(x) for v in valid.values() for x in v])[:750000].astype(np.int64));return ExactMixtureV2(streams),vals
def validation(model,data):
    model.eval();losses=[]
    with torch.no_grad():
        for off in range(0,min(len(data)-129*8,129*8*16),129*8):
            z=data[off:off+129*8].view(8,129);losses.append(float(model(z[:,:128],z[:,1:])[1]))
    model.train();return sum(losses)/len(losses)
def reason_batch(tok,index,batch=8):
    rows=[]
    for j in range(batch):
        ex=generate_example(index+j,split="train",include_trace=True);assert verify_example(ex);ids=tok.encode(ex.rendered_text);ids=([1]+ids)[-129:];ids+=([0]*(129-len(ids)));rows.append(ids)
    z=torch.tensor(rows);return z[:,:128],z[:,1:]
def reasoning_accuracy(model,tok,count=192):
    model.eval();correct=0
    with torch.inference_mode():
        for i in range(count):
            ex=generate_example(i,split="validation",include_trace=False);assert verify_example(ex)
            prompt=ex.rendered_text.rsplit(" Answer:",1)[0]+" Answer:";ids=tok.encode(prompt)[-128:]
            generated=model.generate(torch.tensor(ids)[None,:],max(2,len(tok.encode(ex.answer))+2))[0,len(ids):].tolist();answer=tok.decode(generated).strip()
            correct+=answer.startswith(ex.answer) and (len(answer)==len(ex.answer) or not answer[len(ex.answer)].isalnum())
    model.train();return correct/count
def train_branch(base:Path,mode:str,cid:str,added:int,lr:float)->dict:
    parent=torch.load(base,map_location="cpu",weights_only=False);parent_sha=sha256_file(base);cfg=LatticeConfig(**parent["config"]);model=build_model(cfg);model.load_state_dict(parent["model"],strict=True);model.compile(mode="max-autotune-no-cudagraphs",fullgraph=False);mix,vals=data_setup(cfg);tok=load_tokenizer(TOK)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg.weight_decay,betas=(cfg.adam_beta1,cfg.adam_beta2),eps=1e-8)
    root=ART/"checkpoints"/cid;recovery=root/"latest.pt";consumed=step=0;prior_wall=0.;last=float("nan")
    if recovery.exists() and recovery.with_suffix(".sha256").exists() and sha256_file(recovery)==recovery.with_suffix(".sha256").read_text().strip():
        state=torch.load(recovery,map_location="cpu",weights_only=False)
        if state.get("parent_checkpoint_sha256")!=parent_sha or state.get("branch_token_budget")!=added or state.get("branch_kind")!=mode:raise RuntimeError("branch recovery identity mismatch")
        model.load_state_dict(state["model"],strict=True);opt.load_state_dict(state["optimizer"]);random.setstate(state["python_rng_state"]);np.random.set_state(state["numpy_rng_state"]);torch.set_rng_state(state["torch_rng_state"]);consumed=int(state["branch_training_tokens"]);step=int(state["branch_step"]);prior_wall=float(state["wall_seconds"]);last=float(state["train_loss"])
        if mode=="wsd":mix.load_state_dict(state["data_source_selector_state"]);assert digest_next(mix)==state["next_batch_sha256"]
    elif mode=="wsd":opt.load_state_dict(parent["optimizer"]);mix.load_state_dict(parent["data_source_selector_state"])
    total=max(1,math.ceil(added/1024));start=time.perf_counter()
    def payload():
        result={**parent,"model":model._orig_mod.state_dict() if hasattr(model,"_orig_mod") else model.state_dict(),"optimizer":opt.state_dict(),"scheduler":{"schedule":"cosine-to-zero","branch_step":step,"total_steps":total},"lineage":cid,"run_id":cid,"parent_checkpoint_sha256":parent_sha,"branch_kind":mode,"branch_token_budget":added,"branch_training_tokens":consumed,"branch_step":step,"tokens_seen":int(parent["tokens_seen"])+consumed,"learning_rate_policy":{"name":"cosine","start_lr":lr,"end_lr":0.0,"budget_tokens":added},"wall_seconds":prior_wall+time.perf_counter()-start,"train_loss":last,"python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state()}
        if mode=="wsd":result["data_source_selector_state"]=mix.state_dict();result["next_batch_sha256"]=digest_next(mix)
        return result
    while consumed<added:
        increment=min(1024,added-consumed)
        if mode=="wsd":x,y,_=mix.batch();x=torch.from_numpy(x.astype(np.int64));y=torch.from_numpy(y.astype(np.int64));fraction=(step+1)/total;rate=lr*.5*(1+math.cos(math.pi*fraction))
        else:x,y=reason_batch(tok,step*8);rate=lr
        for group in opt.param_groups:group["lr"]=rate
        opt.zero_grad(set_to_none=True);logits,_=model(x);parts=F.cross_entropy(logits.reshape(-1,cfg.vocab_size),y.reshape(-1),reduction="none").view(8,128);loss=parts.reshape(-1)[:increment].sum()/increment
        if not torch.isfinite(loss):raise FloatingPointError("non-finite branch loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);opt.step();last=float(loss);consumed+=increment;step+=1
        terminal=consumed==added
        if step%384==0 or terminal or STOP:roll_checkpoint(root,payload(),terminal)
        if STOP:raise SystemExit(75)
    cp=root/"milestone.pt";immutable(cp,payload());reason=reasoning_accuracy(model,tok) if mode=="sft" else 0.0;return eval_all(cp,cid,validation(model,vals),reason)
def base_result(base:Path,base_id:str)->dict:
    state=torch.load(base,map_location="cpu",weights_only=False);cfg=LatticeConfig(**state["config"]);model=build_model(cfg);model.load_state_dict(state["model"],strict=True);_,vals=data_setup(cfg)
    return eval_all(base,base_id,validation(model,vals),0.0,ART/f"saturation/wsd/{base_id.lower()}/base-evaluation")
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("wsd","sft","rlvr"));p.add_argument("--base",type=Path,required=True);p.add_argument("--base-id",default="BASE_900M");p.add_argument("--output",type=Path,required=True);a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);torch.set_num_threads(16);torch.set_num_interop_threads(1)
    if a.mode=="wsd":specs=((f"{a.base_id.lower()}-wsd-100m",100_000_000,3e-4),)
    elif a.mode=="sft":specs=(("post900m-sft-1m",1_000_000,2e-5),("post900m-sft-2m",2_000_000,1e-5))
    else:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({"schema":"post900m-rlvr-v1","candidates":[],"status":"SKIPPED_SAFELY","reason":"RLVR policy-gradient worker has not passed trajectory-parity certification; no SFT surrogate was substituted"},indent=2)+"\n");return
    candidates=[train_branch(a.base,a.mode,*spec) for spec in specs];a.output.parent.mkdir(parents=True,exist_ok=True);result={"schema":f"post900m-{a.mode}-v1","candidates":candidates}
    if a.mode=="wsd":result["base"]=base_result(a.base,a.base_id)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":main()
