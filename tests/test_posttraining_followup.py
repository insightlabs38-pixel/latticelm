import json,subprocess,sys,time
from pathlib import Path
import numpy as np,pytest,torch
from scripts import run_posttraining_master as master
from scripts.train_posttraining_worker import batch_candidate_scores,token_candidate_score,batch_completion,rollout_group
from latticelm.posttraining.objectives import ranking_objective,joint_objective
from latticelm.posttraining.provenance import append_record,digest
from latticelm.posttraining.selection import proxy_shortlist,select
from latticelm.lattice_reason_v2.natural import TRANSFORMATIONS,generate_from_tokens,verify

class Toy(torch.nn.Module):
 def __init__(self):super().__init__();self.emb=torch.nn.Embedding(128,16);self.head=torch.nn.Linear(16,128);self.config=type("C",(),{"vocab_size":128})()
 def forward(self,x):return self.head(self.emb(x)),None

def test_batch_scalar_raw_normalized_parity():
 torch.manual_seed(7);model=Toy();prompt=[1,2,3,4];candidates=[[5,6],[7],[8,9,10]];raw,norm=batch_candidate_scores(model,prompt,candidates,32)
 for i,c in enumerate(candidates):
  a,b=token_candidate_score(model,prompt,c,32);assert torch.allclose(raw[i],a[0],atol=1e-6);assert torch.allclose(norm[i],b[0],atol=1e-6)
 assert not torch.allclose(raw,norm)

def test_completion_batch_mask_and_joint_objectives():
 model=Toy();items=[([1,2,3,4],[2,3,4,5],[False,False,True,True]),([2,3],[3,4],[False,True])];loss,logical,full=batch_completion(model,items);assert logical==3 and full==6;loss.backward();assert torch.isfinite(loss)
 raw=torch.tensor([[2.,1.]],requires_grad=True);norm=torch.tensor([[1.,2.]],requires_grad=True);correct=torch.tensor([0]);assert ranking_objective(raw,norm,correct,"ranking_raw")<ranking_objective(raw,norm,correct,"ranking_norm");assert joint_objective(loss,ranking_objective(raw,norm,correct),rank_weight=.3)>loss

@pytest.mark.parametrize("transformation",TRANSFORMATIONS)
def test_natural_exact_provenance(transformation):
 tokens=np.arange(5000,dtype=np.int32);ex=generate_from_tokens(tokens,13,transformation=transformation);assert verify(ex,tokens);bad=np.copy(tokens);bad[ex.source_offset+len(ex.prompt_ids)]+=19;assert not verify(ex,bad)

def test_provenance_restart_idempotency_and_conflict(tmp_path):
 p=append_record(tmp_path,0,{"a":1},2);assert append_record(tmp_path,0,{"a":1},2)==p
 from latticelm.posttraining import provenance
 provenance._CACHE.clear();assert append_record(tmp_path,0,{"a":1},2)==p
 with pytest.raises(RuntimeError):append_record(tmp_path,0,{"a":2},2)
 append_record(tmp_path,1,{"b":2},2);assert Path(p).with_suffix(".sha256").read_text().strip()==digest(p)
 provenance._CACHE.clear()
 with pytest.raises(RuntimeError):append_record(tmp_path,1,{"b":3},2)

def test_scheduler_100h_fast_slow_six_hours_and_cutoff():
 s=master.initial();now=master.EXPERIMENT_CUTOFF-100*3600
 for rate in (500.,5.):
  s["throughput"]["sft"]=rate
  with pytest.MonkeyPatch.context() as m:
   m.setattr(master.time,"time",lambda:now);plan=master.schedule_decision(s,"sft",100_000_000)
  assert plan["selected"]>0 and plan["estimated_seconds"]<=plan["seconds_for_research"]
 with pytest.MonkeyPatch.context() as m:
  m.setattr(master.time,"time",lambda:master.EXPERIMENT_CUTOFF-6*3600);assert master.adaptive_choice(s)["action"]=="FINALIZE";assert not master.experiment_allowed(1,s)
  m.setattr(master.time,"time",lambda:master.HARD_CUTOFF+1);assert not master.experiment_allowed(0,s)
 s["base"]={"checkpoint":"frozen"};s["phase"]="SFT_TRUNK";assert master.should_finalize(s,master.EXPERIMENT_CUTOFF-6*3600) and not master.should_finalize(s,master.EXPERIMENT_CUTOFF-7*3600)

def test_no_idle_and_all_branches_worse_than_base():
 s=master.initial();s["throughput"]["sft"]=100.;s["candidates"]["weak"]={"status":"COMPLETE","method":"sft","checkpoint":"unused","proxy":{"v2_mean_margin":-1.,"v2_ranking_accuracy":0.,"data_d_validation":2.}}
 with pytest.MonkeyPatch.context() as m:
  m.setattr(master.time,"time",lambda:master.EXPERIMENT_CUTOFF-30*3600);d=master.adaptive_choice(s);assert d["action"]=="CONTINUE" and d["parent"]=="weak"
 base={"candidate_id":"BASE","identity_pass":True,"parameter_count":10,"wikitext_bpb":1.,"data_d_validation":2.,"hellaswag":.3,"arc_easy":.3,"piqa":.3,"winogrande":.3};worse={**base,"candidate_id":"weak","hellaswag":.2};assert select([base,worse])["selected"]["candidate_id"]=="BASE"

def test_safe_stopped_candidate_registration_and_lineage(tmp_path,monkeypatch):
 parent=tmp_path/"parent.pt";parent.write_bytes(b"parent");root=tmp_path/"pt";monkeypatch.setattr(master,"PT",root);s=master.initial();s["candidates"]["BASE"]={"checkpoint":str(parent),"checkpoint_sha256":master.sha256(parent),"method_chain":[],"status":"CERTIFIED"};s["throughput"]["sft"]=100.
 monkeypatch.setattr(master.time,"time",lambda:master.EXPERIMENT_CUTOFF-30*3600)
 def fake_command(_s,_label,args,**_):
  out=root/"candidates"/"child";out.mkdir(parents=True,exist_ok=True);cp=out/"latest.pt";cp.write_bytes(b"valid partial");(out/"result.json").write_text(json.dumps({"status":"SAFE_STOPPED_VALID","checkpoint":str(cp),"checkpoint_sha256":master.sha256(cp),"parent_checkpoint_sha256":master.sha256(parent),"method":"sft","training_tokens":500,"processed_tokens":1000,"updates":2}));return True
 monkeypatch.setattr(master,"command",fake_command);monkeypatch.setattr(master,"save",lambda *a,**k:None);monkeypatch.setattr(master,"event",lambda *a,**k:None)
 result=master.worker(s,"child",parent,"sft",1000,optional=True);assert result["status"]=="SAFE_STOPPED_VALID" and s["candidates"]["child"]["method_chain"]==["sft"] and s["candidates"]["child"]["checkpoint_sha256"]==master.sha256(root/"candidates/child/latest.pt")

def test_final_shortlist_keeps_base_partial_and_diversity():
 candidates={"BASE":{"status":"CERTIFIED"},"sft":{"status":"SAFE_STOPPED_VALID","method":"sft","proxy":{"v2_ranking_accuracy":.6}},"rank":{"status":"COMPLETE","method":"ranking","proxy":{"v2_ranking_accuracy":.7}},"rft":{"status":"COMPLETE","method":"rft","proxy":{"v2_ranking_accuracy":.5}}};ids=proxy_shortlist(candidates,3);assert ids[0]=="BASE" and "rank" in ids and len(ids)==3

def test_rollout_policy_version_and_batch_handoff():
 model=Toy();out=rollout_group(model,[1,2,3],"example",lambda ids:True,4,8,None,"parent-sha:0",True);assert len(out)==4 and all(x.policy_sha256=="parent-sha:0" for x in out)

def test_worker_exact_resume_and_safe_stop_with_tiny_model(tmp_path):
 from latticelm.config import LatticeConfig
 from latticelm.model import build_model
 from latticelm.posttraining.state import sha256
 cfg=LatticeConfig(d_model=32,n_layers=1,n_heads=4,n_kv_heads=1,ffn_hidden=64,context_length=32,vocab_size=4096);model=build_model(cfg);base=tmp_path/"base.pt";torch.save({"model":model.state_dict(),"config":cfg.to_dict()},base)
 data=tmp_path/"data";data.mkdir();(np.arange(1024,dtype="<i4")%4096).tofile(data/"tokens.bin");(data/"shard.json").write_text(json.dumps({"path":"tokens.bin"}));manifest=data/"manifest.json";manifest.write_text(json.dumps({"shards":[{"manifest_path":"shard.json"}]}));out=tmp_path/"candidate";tokenizer=Path(__file__).resolve().parents[1]/"artifacts/tokenizers/final_corpus_4k.json"
 base_cmd=[sys.executable,"scripts/train_posttraining_worker.py","--base",str(base),"--output",str(out),"--tokenizer",str(tokenizer),"--manifest",str(manifest),"--method","sft","--optimizer","adamw","--threads","1","--microbatch","4","--checkpoint-tokens","1"]
 for target in (2,4):
  subprocess.run(base_cmd+["--tokens",str(target)],check=True,capture_output=True,text=True,timeout=90);result=json.loads((out/"result.json").read_text());assert result["training_tokens"]==target and result["checkpoint_sha256"]==sha256(out/"latest.pt")
 before=result["updates"];subprocess.run(base_cmd+["--tokens","100","--stop-epoch",str(time.time()-1)],check=True,capture_output=True,text=True,timeout=90);result=json.loads((out/"result.json").read_text());assert result["status"]=="SAFE_STOPPED_VALID" and result["training_tokens"]==4 and result["updates"]==before
 subprocess.run(base_cmd+["--tokens","8"],check=True,capture_output=True,text=True,timeout=90);resumed=json.loads((out/"result.json").read_text());assert resumed["status"]=="COMPLETE" and resumed["training_tokens"]==8 and resumed["updates"]==before+1

def test_topology_selection_is_passed_to_rollout_worker(tmp_path,monkeypatch):
 parent=tmp_path/"parent.pt";parent.write_bytes(b"parent");root=tmp_path/"pt";root.mkdir();(root/"rollout_backend_certification.json").write_text(json.dumps({"selected":{"processes":1,"threads":4,"batched":True}}));monkeypatch.setattr(master,"PT",root);s=master.initial();s["rollout_backend"]="CERTIFIED";s["candidates"]["BASE"]={"checkpoint":str(parent),"checkpoint_sha256":master.sha256(parent),"method_chain":[]};s["throughput"]["rft"]=100.;captured=[]
 monkeypatch.setattr(master.time,"time",lambda:master.EXPERIMENT_CUTOFF-30*3600);monkeypatch.setattr(master,"event",lambda *a,**k:None);monkeypatch.setattr(master,"save",lambda *a,**k:None)
 def fake_command(_s,_label,args,**_):
  captured.extend(map(str,args));out=root/"candidates"/"rollout";out.mkdir(parents=True);cp=out/"latest.pt";cp.write_bytes(b"model");(out/"result.json").write_text(json.dumps({"status":"COMPLETE","checkpoint":str(cp),"checkpoint_sha256":master.sha256(cp),"parent_checkpoint_sha256":master.sha256(parent),"method":"rft","training_tokens":100,"processed_tokens":100,"updates":1}));return True
 monkeypatch.setattr(master,"command",fake_command);assert master.worker(s,"rollout",parent,"rft",100,optional=True);assert "--rollout-topology" in captured and str(root/"rollout_backend_certification.json") in captured

def test_merge_checkpoint_identity_and_finite_tensors(tmp_path,monkeypatch):
 root=tmp_path/"pt";root.mkdir();monkeypatch.setattr(master,"PT",root);base=tmp_path/"base.pt";child=tmp_path/"child.pt";config={"d_model":4};torch.save({"config":config,"model":{"weight":torch.ones(2,2)}},base);torch.save({"config":config,"model":{"weight":torch.full((2,2),3.)}},child)
 s=master.initial();s["base"]={"checkpoint":str(base),"checkpoint_sha256":master.sha256(base)};s["candidates"]={"BASE":{"checkpoint":str(base),"checkpoint_sha256":master.sha256(base),"status":"CERTIFIED"},"sft":{"checkpoint":str(child),"checkpoint_sha256":master.sha256(child),"method":"sft","status":"COMPLETE"}}
 monkeypatch.setattr(master,"strongest_candidate",lambda _s,methods=None:"sft" if methods=={"sft"} else "BASE");monkeypatch.setattr(master,"save",lambda *a,**k:None);monkeypatch.setattr(master,"complete",lambda *a,**k:None);master.phase_interpolation(s)
 merged=s["candidates"]["base-sft-0.50"];payload=torch.load(merged["checkpoint"],weights_only=False);assert torch.equal(payload["model"]["weight"],torch.full((2,2),2.)) and payload["merge"]["parents"]==[master.sha256(base),master.sha256(child)]
 torch.save({"config":{"d_model":8},"model":{"weight":torch.ones(2,2)}},child)
 with pytest.raises(RuntimeError,match="merge parent SHA mismatch"):master.phase_interpolation(s)

@pytest.mark.parametrize("case",("rlvr_ineligible","rft_success","retention_harm","all_base","partial","time_remains","finalize_six_hours","optional_optimization_failure"))
def test_scheduler_simulation_cases(case):
 s=master.initial();s["throughput"].update({"sft":80.,"rft":20.});s["candidates"]["sft"]={"status":"COMPLETE","method":"sft","proxy":{"v2_mean_margin":.1,"v2_ranking_accuracy":.4,"data_d_validation":2.}}
 if case=="rft_success":s["candidates"]["rft"]={"status":"COMPLETE","method":"rft","proxy":{"v2_mean_margin":.3,"v2_ranking_accuracy":.6,"data_d_validation":2.}}
 if case=="retention_harm":s["candidates"]["sft"]["proxy"]["data_d_validation"]=2.2
 if case=="partial":s["candidates"]["sft"]["status"]="SAFE_STOPPED_VALID"
 if case=="rlvr_ineligible":s["rlvr_eligible"]=False
 if case=="optional_optimization_failure":s["selected_compile"]=False;s["rollout_backend"]="REJECTED_OPTIONAL"
 now=master.EXPERIMENT_CUTOFF-(6 if case=="finalize_six_hours" else 30)*3600
 with pytest.MonkeyPatch.context() as m:
  m.setattr(master.time,"time",lambda:now);decision=master.adaptive_choice(s)
 assert decision["action"]==("FINALIZE" if case=="finalize_six_hours" else "CONTINUE")
 if case=="rft_success":assert decision["parent"]=="rft"
 if case=="partial":assert decision["parent"]=="sft"
