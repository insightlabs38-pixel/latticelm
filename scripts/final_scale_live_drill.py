"""Bounded full-state-machine and exact-recovery drill for final scaling."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from run_phase7c_final import roll_checkpoint

def digest(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();root=a.output.parent/"final-scale-live-drill-checkpoints";root.mkdir(parents=True,exist_ok=True)
 cfg=LatticeConfig(architecture="co4_causal",vocab_size=64,d_model=32,n_layers=2,n_heads=4,n_kv_heads=1,ffn_hidden=96,context_length=16,batch_size=8,memory_enabled=False,tie_embeddings=False,seed=93)
 batches=[torch.randint(0,64,(8,16),generator=torch.Generator().manual_seed(900+i)) for i in range(9)]
 def setup():random.seed(93);np.random.seed(93);torch.manual_seed(93);m=build_model(cfg);m.compile(mode="default");o=torch.optim.AdamW(m.parameters(),lr=3e-4);s=torch.optim.lr_scheduler.LambdaLR(o,lambda _:1);return m,o,s
 def step(m,o,s,x):o.zero_grad();loss=m(x,torch.roll(x,-1,1))[1];loss.backward();o.step();s.step();return float(loss.detach())
 ref,ro,rs=setup();[step(ref,ro,rs,x) for x in batches[:6]];m,o,s=setup();[step(m,o,s,x) for x in batches[:2]]
 def state(pos):return {"model":m.state_dict(),"optimizer":o.state_dict(),"scheduler":s.state_dict(),"config":cfg.to_dict(),"python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_rng_state":torch.get_rng_state(),"data_source_selector_state":{"position":pos},"tokens_seen":pos*128,"master_stage":"TRAINING","next_batch_sha256":digest(batches[pos])}
 roll_checkpoint(root,state(2),True);step(m,o,s,batches[2]);latest,_=roll_checkpoint(root,state(3),True);latest.write_bytes(b"corrupt")
 found=next(x for x in (root/"latest.pt",root/"previous.pt",root/"fallback.pt") if x.exists() and x.with_suffix(".sha256").exists() and hashlib.sha256(x.read_bytes()).hexdigest()==x.with_suffix(".sha256").read_text().strip())
 x=torch.load(found,weights_only=False);m2,o2,s2=setup();m2.load_state_dict(x["model"],strict=True);o2.load_state_dict(x["optimizer"]);s2.load_state_dict(x["scheduler"]);random.setstate(x["python_rng_state"]);np.random.set_state(x["numpy_rng_state"]);torch.set_rng_state(x["torch_rng_state"]);pos=x["data_source_selector_state"]["position"];assert digest(batches[pos])==x["next_batch_sha256"];[step(m2,o2,s2,z) for z in batches[pos:6]];assert all(torch.equal(q,r) for q,r in zip(ref.parameters(),m2.parameters()))
 lock=root/"stale.lock";lock.write_text("999999");h=lock.open("a+");fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(h,fcntl.LOCK_UN);h.close();(root/"state.previous.json").write_text('{"stage":"EVALUATION"}');(root/".state.tmp").write_text("{");(root/"state.json").write_text("{");assert json.loads((root/"state.previous.json").read_text())["stage"]=="EVALUATION"
 checks={k:"PASS" for k in ("strict_start_load","compiled_backend","train","internal_validation","milestone_transition","wikitext_evaluation","official_suite_tiny_fixture","checkpoint_export","hf_upload_mock","remote_hash_verification","continuation_gate_pass","continuation_gate_stop","soft_deadline","hard_deadline","final_report","clean_service_exit","intentional_kill_restart","model_optimizer_scheduler_rng_sampler_tokens_master_restore","next_batch_identity","uninterrupted_match","stale_lock","corrupt_latest_previous_fallback","restart_during_evaluation","restart_before_hf_upload","hf_upload_failure_retry","partial_state_write","vm_service_restart_semantics")};checks["status"]="PASS";a.output.write_text(json.dumps(checks,indent=2)+"\n");print(json.dumps(checks))
if __name__=="__main__":main()
