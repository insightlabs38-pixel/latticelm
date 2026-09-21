import json
import pytest
import scripts.run_posttraining_master as master
from latticelm.posttraining.selection import select
from latticelm.posttraining.state import Lock,atomic_json,sha256
def test_atomic_backup_and_immutable_hash(tmp_path):
 p=tmp_path/"s.json";b=tmp_path/"prev.json";atomic_json(p,{"x":1},b);atomic_json(p,{"x":2},b);assert json.loads(b.read_text())=={"x":1} and sha256(p);atomic_json(tmp_path/"i.json",{"x":1},immutable=True)
def test_lock_rejects_duplicate(tmp_path):
 with Lock(tmp_path/"lock"):
  with pytest.raises(BlockingIOError):
   with Lock(tmp_path/"lock"):pass
def test_deadlines_and_dry_run(monkeypatch):
 result=master.dry_run();assert result["passive_wait"] and master.EXPERIMENT_CUTOFF<master.HARD_CUTOFF;monkeypatch.setattr(master.time,"time",lambda:master.EXPERIMENT_CUTOFF+1);assert not master.experiment_allowed()
def test_wait_uses_semantic_handoff(tmp_path,monkeypatch):
 p=tmp_path/"prod.json";p.write_text(json.dumps({"terminal_state":None}));monkeypatch.setattr(master,"PRODUCTION_STATE",p);assert master.phase_wait(master.initial()) is False
def test_optional_method_fallthrough_and_base_selection():
 base={"candidate_id":"BASE","identity_pass":True,"parameter_count":10,"wikitext_bpb":1.,"data_d_validation":2.,"hellaswag":.3,"arc_easy":.3,"piqa":.3,"winogrande":.3};bad={**base,"candidate_id":"bad","identity_pass":False,"hellaswag":.9};decision=select([base,bad]);assert decision["selected"]["candidate_id"]=="BASE" and decision["rejected"][0]["candidate"]["candidate_id"]=="bad"
def test_master_state_contains_recovery_fields():
 s=master.initial();assert s["phase"]=="WAIT_FOR_PRODUCTION_COMPLETE" and "BASE" in s["candidates"] and s["gibc_evaluations"]==0 and s["terminal_state"] is None
