"""Run a matched frozen-Co4-S data-regime experiment with two validations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import subprocess
import time

import numpy as np
import torch

from latticelm.config import LatticeConfig
from latticelm.model import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/data/phase7a"


def load(name: str) -> torch.Tensor:
    return torch.from_numpy(np.memmap(DATA/f"{name}.int32", mode="r", dtype=np.int32)).long()


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


class SequentialSource:
    """Seeded, without-replacement context sampler, resumable by draw count."""
    def __init__(self, tokens: torch.Tensor, context: int, seed: int):
        self.tokens=tokens; self.context=context; self.seed=seed; self.draws=0
        self.blocks=(len(tokens)-1)//context
    def one(self) -> tuple[torch.Tensor, torch.Tensor]:
        epoch, position = divmod(self.draws, self.blocks)
        # Affine permutation when multiplier is coprime with block count.
        rng=random.Random(self.seed+epoch); a=rng.randrange(1,self.blocks)
        while math.gcd(a,self.blocks)!=1: a=(a+1)%self.blocks or 1
        b=rng.randrange(self.blocks); block=(a*position+b)%self.blocks
        self.draws += 1; start=block*self.context
        return self.tokens[start:start+self.context], self.tokens[start+1:start+self.context+1]


def evaluate(model, tokens, config) -> float:
    generator=torch.Generator().manual_seed(424242); losses=[]
    model.eval()
    with torch.inference_mode():
        for _ in range(16):
            starts=torch.randint(0,len(tokens)-config.context_length-1,(config.batch_size,),generator=generator)
            x=torch.stack([tokens[i:i+config.context_length] for i in starts.tolist()])
            y=torch.stack([tokens[i+1:i+config.context_length+1] for i in starts.tolist()])
            losses.append(float(model(x,y)[1]))
    model.train(); return sum(losses)/len(losses)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--regime",choices=["DATA-A","DATA-B","DATA-C"],required=True)
    p.add_argument("--max-tokens",type=int,required=True); p.add_argument("--resume"); p.add_argument("--experiment",required=True)
    a=p.parse_args(); config=LatticeConfig.from_json(ROOT/"configs/phase7a_co4_s.json")
    if a.max_tokens % (config.batch_size*config.context_length): p.error("max tokens must be a multiple of 1024")
    max_steps=a.max_tokens//(config.batch_size*config.context_length); torch.set_num_threads(config.num_threads)
    random.seed(config.seed); torch.manual_seed(config.seed)
    baby=load("babylm_train"); web=load("finewebedu_train"); own={"DATA-A":load("babylm_validation"),"DATA-B":load("finewebedu_validation")}
    common=load("common_validation")
    if a.regime=="DATA-C":
        web_validation=load("finewebedu_validation")
        each=min(len(web_validation),len(load("babylm_validation"))//3)
        own["DATA-C"]=torch.cat((load("babylm_validation")[:3*each],web_validation[:each]))
    sources={"baby":SequentialSource(baby,config.context_length,config.seed+11),
             "web":SequentialSource(web,config.context_length,config.seed+29)}
    model=build_model(config); optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,
        weight_decay=config.weight_decay,betas=(config.adam_beta1,config.adam_beta2))
    start_step=0; previous_wall=0.; curves=[]
    if a.resume:
        ck=torch.load(a.resume,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        start_step=ck["step"]; previous_wall=ck["wall_seconds"]; random.setstate(ck["random_state"]); torch.set_rng_state(ck["torch_rng_state"])
        for key,value in ck["source_draws"].items(): sources[key].draws=value
    milestones={256,512,768,1024,1536,2048,2560,3072,10240}
    start=time.perf_counter(); train_loss=float("nan"); source_tokens={"baby":sources["baby"].draws*128,"web":sources["web"].draws*128}
    log=ROOT/"artifacts/logs"/f"{a.experiment}.jsonl"
    for step in range(start_step+1,max_steps+1):
        kinds=(["baby"]*8 if a.regime=="DATA-A" else ["web"]*8 if a.regime=="DATA-B" else ["baby"]*6+["web"]*2)
        pairs=[sources[k].one() for k in kinds]; x=torch.stack([z[0] for z in pairs]); y=torch.stack([z[1] for z in pairs])
        for k in kinds: source_tokens[k]+=config.context_length
        _,loss=model(x,y); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip); optimizer.step(); train_loss=float(loss)
        if step in milestones or step==max_steps:
            own_loss=evaluate(model,own[a.regime],config); common_loss=evaluate(model,common,config); wall=previous_wall+time.perf_counter()-start
            row={"experiment_id":a.experiment,"dataset_regime":a.regime,
                 "source_ratios":"100/0" if a.regime=="DATA-A" else "0/100" if a.regime=="DATA-B" else "75/25",
                 "seed":config.seed,"model_parameters":sum(p.numel() for p in model.parameters()),"train_tokens":step*1024,
                 "babylm_tokens":source_tokens["baby"],"finewebedu_tokens":source_tokens["web"],
                 "unique_documents_seen":None,"wall_seconds":wall,"tokens_per_second":step*1024/wall,
                 "train_loss":train_loss,"own_validation_loss":own_loss,"common_validation_loss":common_loss,
                 "common_validation_perplexity":math.exp(common_loss),"peak_ram_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 "checkpoint_hash":"pending" if step==max_steps else "screening_not_persisted","status":"completed"}
            curves.append(row)
            with log.open("a") as f: f.write(json.dumps(row)+"\n")
    wall=previous_wall+time.perf_counter()-start; ck_path=ROOT/"artifacts/checkpoints"/f"{a.experiment}_step{max_steps}.pt"
    ck_path.parent.mkdir(exist_ok=True); torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":max_steps,
        "tokens_seen":max_steps*1024,"config":config.to_dict(),"source":a.regime,"wall_seconds":wall,
        "source_draws":{k:v.draws for k,v in sources.items()},"random_state":random.getstate(),"torch_rng_state":torch.get_rng_state()},ck_path)
    checkpoint_hash=sha256(ck_path); curves[-1]["checkpoint_hash"]=checkpoint_hash
    # Correct the just-written final log row and append-only phase ledger.
    lines=log.read_text().splitlines(); lines[-1]=json.dumps(curves[-1]); log.write_text("\n".join(lines)+"\n")
    ledger=ROOT/"artifacts/data_learning_curves.csv"; fields=list(curves[0]); exists=ledger.exists()
    with ledger.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); (None if exists else w.writeheader()); w.writerows(curves)
    print(json.dumps(curves[-1],indent=2))


if __name__=="__main__": main()
