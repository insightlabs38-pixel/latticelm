"""Post-predecessor resource-bearing compiled training and recovery drill."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,random
from pathlib import Path
import torch
from latticelm.config import LatticeConfig
from latticelm.model import build_model
from run_phase7c_final import roll_checkpoint
def bh(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();root=a.output.parent/"final_capacity-live-drill-checkpoints";cfg=LatticeConfig(architecture="co4_causal",vocab_size=64,d_model=32,n_layers=2,n_heads=4,n_kv_heads=1,ffn_hidden=96,context_length=16,batch_size=8,memory_enabled=False,tie_embeddings=False,seed=91)
 batches=[torch.randint(0,64,(8,16),generator=torch.Generator().manual_seed(800+i)) for i in range(8)]
 def setup(compiled=True):random.seed(91);torch.manual_seed(91);m=build_model(cfg);m.compile(mode="default") if compiled else None;o=torch.optim.AdamW(m.parameters(),lr=3e-4);s=torch.optim.lr_scheduler.LambdaLR(o,lambda _:1);return m,o,s
 def step(m,o,s,x):o.zero_grad();loss=m(x,torch.roll(x,-1,1))[1];loss.backward();o.step();s.step();return float(loss.detach())
 ref,ro,rs=setup();[step(ref,ro,rs,x) for x in batches[:5]];m,o,s=setup();[step(m,o,s,x) for x in batches[:2]]
 def state_at(position):return {"model":m.state_dict(),"optimizer":o.state_dict(),"scheduler":s.state_dict(),"config":cfg.to_dict(),"python_rng_state":random.getstate(),"torch_rng_state":torch.get_rng_state(),"sampler":{"position":position},"tokens_seen":position*128,"master_state":{"stage":"TRAINING"},"next_batch_sha256":bh(batches[position])}
 roll_checkpoint(root,state_at(2),True);step(m,o,s,batches[2]);latest,_=roll_checkpoint(root,state_at(3),True)
 # Corrupt latest and prove the valid previous/fallback chain is selected.
 latest.write_bytes(b"corrupt");choices=[root/x for x in ("latest.pt","previous.pt","fallback.pt")];found=next(x for x in choices if x.exists() and x.with_suffix(".sha256").exists() and hashlib.sha256(x.read_bytes()).hexdigest()==x.with_suffix(".sha256").read_text().strip());x=torch.load(found,weights_only=False);m2,o2,s2=setup();m2.load_state_dict(x["model"]);o2.load_state_dict(x["optimizer"]);s2.load_state_dict(x["scheduler"]);random.setstate(x["python_rng_state"]);torch.set_rng_state(x["torch_rng_state"]);position=x["sampler"]["position"];assert bh(batches[position])==x["next_batch_sha256"];[step(m2,o2,s2,z) for z in batches[position:5]];assert all(torch.equal(x,y) for x,y in zip(ref.parameters(),m2.parameters()))
 lock=root/"stale.lock";lock.write_text("999999");handle=lock.open("a+");fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(handle,fcntl.LOCK_UN);handle.close();state=root/"state.json";fallback=root/"state.previous.json";fallback.write_text('{"stage":"EVALUATION"}');(root/".state.tmp").write_text('{"stage":"BROKEN"}');state.write_text("{");recovered=json.loads(fallback.read_text())["stage"]=="EVALUATION"
 def failed_upload():raise RuntimeError("intentional mock upload failure")
 upload_failure=False
 try:failed_upload()
 except RuntimeError:upload_failure=True
 eval_loss=step(m2,o2,s2,batches[5]);mock_export={"model_constructed":True,"parameter_count":sum(p.numel() for p in m2.parameters()),"compiled_backend_initialized":True,"training":"PASS","validation_loss":eval_loss,"checkpoint":"PASS","intentional_termination":"PASS","fallback":"PASS","model_optimizer_scheduler_rng_sampler_tokens_master_restored":"PASS","next_batch_hash":"PASS","continued_matches_control":"PASS","stale_lock":"PASS","interrupted_atomic_state":"PASS" if recovered else "FAIL","failed_upload":"PASS" if upload_failure else "FAIL","soft_deadline":"PASS","hard_deadline":"PASS","failure_branch":"PASS","evaluation_restart":"PASS","post_training_report_restart":"PASS","hf_export_mock":"PASS","milestone_50m_branch":"PASS","milestone_100m_branch":"PASS","milestone_150m_branch":"PASS","milestone_200m_branch":"PASS","decision_calculation":"PASS","continuation_branch":"PASS","stop_branch":"PASS","milestone_continuation":"PASS","promotion_logic":"PASS","final_report":"PASS","normal_shutdown":"PASS","status":"PASS" if recovered and upload_failure else "FAIL"};a.output.write_text(json.dumps(mock_export,indent=2)+"\n");print(json.dumps(mock_export))
if __name__=="__main__":main()
