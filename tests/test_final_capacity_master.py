from pathlib import Path
from latticelm.config import LatticeConfig
from latticelm.model import build_model
import scripts.run_final_capacity_master as master

def test_final_geometry_and_family():
    cfg=LatticeConfig.from_json(Path("configs/final_capacity32_co4.json"));model=build_model(cfg)
    assert model.parameter_breakdown()["total"]==32_678_640
    assert (cfg.architecture,cfg.context_length,cfg.vocab_size,cfg.tie_embeddings)==("co4_causal",128,4096,False)
    assert not cfg.memory_enabled

def test_master_contract_and_forbidden_work():
    text=(Path("scripts/run_final_capacity_master.py").read_text()+Path("scripts/train_final_capacity.py").read_text()).lower()
    for required in ("audit_24m","50_000_000","100_000_000","150_000_000","matched","final_capacity_surface.csv","final_lineage_status.md","capacity_search_closed","soft_deadline","hard_deadline","fallback.pt","hf hash verification"):
        assert required.lower() in text
    for forbidden in ("schedulefree","triton","bf16","lattice_reason","40m","49m","dpo","rlhf"):
        assert forbidden not in text

def test_live_drill_covers_all_branches():
    text=Path("scripts/final_capacity_live_drill.py").read_text()
    for required in ("milestone_50m_branch","milestone_100m_branch","milestone_150m_branch","milestone_200m_branch","decision_calculation","continuation_branch","stop_branch","failed_upload","evaluation_restart","post_training_report_restart","interrupted_atomic_state","stale_lock","normal_shutdown"):
        assert required in text

def test_backend_is_frozen_to_certified_choice():
    good=[{"backend":"fp32_compile","relative_speedup_percent":"13.4","status":"PASS","finite_gradients":"True","loss_sanity":"True"}]
    assert master.select_backend(good,{"stats":{"unique_graphs":1}})[0]=="fp32_compile"
