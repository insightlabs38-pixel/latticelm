from pathlib import Path
import scripts.run_final_scale_master as master

def test_frozen_capacity_and_milestones():
 assert master.PARAMS==32_678_640 and master.START==205_000_000 and master.FINAL_TARGET==900_000_000
 assert master.NORMALIZED=={"M1":512_000_000,"M2":1_024_000_000,"M3":1_537_000_000,"M4":2_049_000_000}
 assert master.NORMALIZED["M2"]>1_009_592_508
def test_contract_and_no_search():
 text=Path("scripts/run_final_scale_master.py").read_text().lower()
 for required in ("capacity search closed: yes","final_scale_report.md","final_scale_curve.csv","final_scale_milestones.csv","final_scale_decision.md","final_scale_benchmark_deltas.csv","final_scale_runtime.csv","soft_deadline","hard_deadline","previous.pt","fallback.pt","remote hash verification"):
  assert required in text
 for forbidden in ("schedulefree","triton","bf16","lattice_reason","40m","49m","dpo","rlhf","hpo"):
  assert forbidden not in text
def test_drill_contract():
 text=Path("scripts/final_scale_live_drill.py").read_text()
 for key in ("continuation_gate_pass","continuation_gate_stop","restart_during_evaluation","restart_before_hf_upload","hf_upload_failure_retry","partial_state_write","vm_service_restart_semantics","uninterrupted_match"):
  assert key in text
