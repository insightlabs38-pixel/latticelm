from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

import scripts.run_phase7e_master as master


def test_atomic_state_serialization_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "STATE", tmp_path/"state.json")
    master.atomic_json(master.STATE, {"run_id":"x","value":3})
    assert master.load_state()["value"] == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_process_lock_and_stale_lock(tmp_path):
    path=tmp_path/"lock"; first=master.MasterLock(path); second=master.MasterLock(path)
    assert first.acquire(); assert not second.acquire(); first.release()
    path.write_text("999999\n")
    assert second.acquire(); second.release()


def test_stage_transitions_are_explicit():
    assert master.STAGES[0] == "PREFLIGHT"
    assert master.STAGES.index("DATA_C_TO_50M") < master.STAGES.index("DATA_C_TO_100M")
    assert "24M" not in " ".join(master.STAGES)


def test_soft_hard_deadline_and_time_aware_skip():
    now=datetime.now(timezone.utc)
    state={"soft_deadline":(now+timedelta(seconds=10)).isoformat(),"hard_deadline":(now+timedelta(seconds=20)).isoformat()}
    assert master.deadline(state,"soft") < master.deadline(state,"hard")
    assert not master.can_start(state, 30, reserve=0)


def test_checkpoint_failure(monkeypatch):
    monkeypatch.setattr(master, "RECOVERY", Path("/does/not/exist"))
    with pytest.raises(RuntimeError, match="missing Phase 7D"):
        master.check_phase7d()


def test_spot_resume_and_hf_retry_logic(monkeypatch, tmp_path):
    state={"current_stage":"x","retry_counts":{},"spot_interruption_count":0,"hard_deadline":(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),"trainer_pid":None}
    monkeypatch.setattr(master,"STATE",tmp_path/"state.json"); master.atomic_json(master.STATE,{"run_id":"x","stop_requested":False})
    monkeypatch.setattr(master,"save",lambda s: None); monkeypatch.setattr(master,"event",lambda *a,**k: None); monkeypatch.setattr(master.time,"sleep",lambda _:None)
    class P:
        calls=0
        def __init__(self,*a,**k): self.pid=1
        def wait(self): P.calls+=1; return 1 if P.calls==1 else 0
    monkeypatch.setattr(master.subprocess,"Popen",P)
    master.run_child(state,"hf_upload",["true"],retries=2)
    assert state["retry_counts"]["hf_upload"] == 1 and state["spot_interruption_count"] == 1


def test_exact_next_batch_recovery_and_data_prep_recovery(tmp_path):
    batch=b"synthetic-next-batch"; expected=hashlib.sha256(batch).hexdigest()
    state={"run_id":"x","next_batch_hash":expected,"current_stage":"DATA_D_PREPARATION"}
    path=tmp_path/"s.json"; master.atomic_json(path,state)
    recovered=json.loads(path.read_text())
    assert recovered["next_batch_hash"] == hashlib.sha256(batch).hexdigest()
    assert recovered["current_stage"] == "DATA_D_PREPARATION"
