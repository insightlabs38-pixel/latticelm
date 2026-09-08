from __future__ import annotations
import json,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pytest
import scripts.run_capacity24_master as master
from latticelm.config import LatticeConfig
from latticelm.model import build_model

def test_geometry_and_frozen_family():
 cfg=LatticeConfig.from_json(Path("configs/capacity24_co4.json"));model=build_model(cfg)
 assert model.parameter_breakdown()["total"]==24_097_200
 assert (cfg.architecture,cfg.context_length,cfg.vocab_size,cfg.tie_embeddings)==("co4_causal",128,4096,False)
 assert not cfg.memory_enabled

def test_launch_wait_does_not_start_budget():
 state=master.initial();assert state["current_stage"]=="WAITING_FOR_PREDECESSOR"
 assert state["experiment_budget_start_time"] is None and state["soft_deadline"] is None and state["hard_deadline"] is None
 assert state["predecessor"]["unit"]=="latticelm-systems-data-20260907.service"

def test_backend_promotion_and_fallback():
 good=[{"backend":"fp32_compile","relative_speedup_percent":"13.4","status":"PASS","finite_gradients":"True","loss_sanity":"True"}]
 assert master.select_backend(good,{"stats":{"unique_graphs":1}})[0]=="fp32_compile"
 bad=[{**good[0],"relative_speedup_percent":"9.9"}];assert master.select_backend(bad,{"stats":{"unique_graphs":1}})[0]=="fp32_eager"
 assert master.select_backend(good,{"stats":{"unique_graphs":7}})[0]=="fp32_eager"

def test_wait_terminates_without_spending_budget(monkeypatch):
 states=iter([{"ActiveState":"active"},{"ActiveState":"inactive","Result":"success","ExecMainStatus":"0"},{"ActiveState":"inactive","Result":"success","ExecMainStatus":"0"}]);monkeypatch.setattr(master,"systemd_props",lambda:next(states));monkeypatch.setattr(master.time,"sleep",lambda _:None);monkeypatch.setattr(master,"save",lambda *a,**k:None);monkeypatch.setattr(master,"event",lambda *a,**k:None);s=master.initial();master.wait_predecessor(s);assert s["experiment_budget_start_time"] is None

def test_hard_deadline_and_failure_branch(monkeypatch):
 s=master.initial();s["hard_deadline"]=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
 with pytest.raises(TimeoutError):master.run(s,"deadline",["true"])

def test_corrupt_state_valid_fallback_and_stale_lock(tmp_path,monkeypatch):
 monkeypatch.setattr(master,"STATE",tmp_path/"state");monkeypatch.setattr(master,"BACKUP",tmp_path/"fallback");monkeypatch.setattr(master,"LOCK",tmp_path/"lock");master.STATE.write_text("{");master.BACKUP.write_text('{"current_stage":"EVALUATION"}');master.LOCK.write_text("999999")
 assert master.load()["current_stage"]=="EVALUATION"
 with master.Lock():assert master.LOCK.read_text()

def test_live_drill_and_all_late_stages_are_mandatory():
 text=Path("scripts/capacity24_live_drill.py").read_text();required=("model_constructed","compiled_backend_initialized","training","validation_loss","checkpoint","intentional_termination","fallback","next_batch_hash","failed_upload","evaluation_restart","post_training_report_restart","milestone_continuation","promotion_logic","final_report","normal_shutdown","soft_deadline","hard_deadline","failure_branch")
 assert all(x in text for x in required)
 master_text=Path("scripts/run_capacity24_master.py").read_text()
 for stage in ("LIVE_CERTIFICATION","CALIBRATION","TRAIN_","EVALUATE_","CAPACITY_ANALYSIS","FINAL_HYGIENE","COMPLETE","BLOCKED"):assert stage in master_text

def test_forbidden_experiments_absent():
 text=Path("scripts/run_capacity24_master.py").read_text().lower()
 for forbidden in ("schedulefree","triton","bf16","lattice_reason","32m training"):assert forbidden not in text

def test_complete_production_stage_traversal_with_fixtures(tmp_path,monkeypatch):
 stages=[];labels=[];s=master.initial();monkeypatch.setattr(master,"ROOT",tmp_path);monkeypatch.setattr(master,"ART",tmp_path);monkeypatch.setattr(master,"STATE",tmp_path/"state.json")
 monkeypatch.setattr(master,"wait_predecessor",lambda state:stages.append("WAITING_FOR_PREDECESSOR"));monkeypatch.setattr(master,"gate",lambda state:"fp32_compile")
 def save(state,stage=None,**kw):
  if stage:state["current_stage"]=stage;stages.append(stage)
  state.update(kw)
 monkeypatch.setattr(master,"save",save)
 def run(state,label,cmd,allow75=False):
  labels.append(label)
  if label=="live-smoke-recovery":(tmp_path/"capacity24_live_drill.json").write_text('{"status":"PASS"}')
  if label=="calibration":(tmp_path/"capacity24_calibration.json").write_text('{"status":"PASS","selected":{"threads":12,"microbatch":8}}')
  if label.startswith("wiki-"):Path(cmd[cmd.index("--output")+1]).write_text('{"perplexity":70,"bits_per_byte":2}')
  if label=="gibc-150m":Path(cmd[cmd.index("--output")+1]).write_text('{"results":{}}')
  return 0
 monkeypatch.setattr(master,"run",run);monkeypatch.setattr(master,"publish",lambda state,target,wiki,gibc=None:{"wikitext":wiki,"gibc":gibc});monkeypatch.setattr(master,"analyze",lambda state,results:{"capacity_result":"FLAT","recommend_32m":False});monkeypatch.setattr(master.subprocess,"run",lambda *a,**k:type("R",(),{"returncode":0})());s["soft_deadline"]=(datetime.now(timezone.utc)+timedelta(hours=21)).isoformat();s["hard_deadline"]=(datetime.now(timezone.utc)+timedelta(hours=22)).isoformat();master.execute(s)
 assert labels==["live-smoke-recovery","calibration","train-50000000","wiki-50000000","train-100000000","wiki-100000000","train-150000000","wiki-150000000","gibc-150m","final-tests"]
 for stage in ("CALIBRATION","TRAIN_50M","EVALUATE_50M","TRAIN_100M","EVALUATE_100M","TRAIN_150M","EVALUATE_150M","CAPACITY_ANALYSIS","FINAL_HYGIENE","COMPLETE"):assert stage in stages

def test_predecessor_gate_failure_writes_blocker(tmp_path,monkeypatch):
 s=master.initial();seen=[];monkeypatch.setattr(master,"ART",tmp_path);monkeypatch.setattr(master,"wait_predecessor",lambda state:None);monkeypatch.setattr(master,"gate",lambda state:(_ for _ in ()).throw(RuntimeError("fixture rejection")));monkeypatch.setattr(master,"save",lambda state,stage=None,**kw:seen.append(stage));master.execute(s)
 assert seen==["BLOCKED"] and "24M training was not launched" in (tmp_path/"capacity24_blocker_report.md").read_text()
