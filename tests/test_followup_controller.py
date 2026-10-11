"""Focused validity/recovery tests for the final-week control plane."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import followup_common as common
import posthack_followup_master as master


def test_deadline_projection_accounts_for_resume_and_overhead():
    assert common.projected_seconds(200_000_000, 4000, .08, 900) == 54_900
    assert common.parse_time("2026-10-17T20:10:00Z") == common.parse_time("2026-10-17T16:10:00-04:00")
    with pytest.raises(ValueError): common.parse_time("2026-10-17T20:10:00")


def test_failure_classification_does_not_spend_code_budget_for_kill():
    assert common.failure_class(-9,None) == "INTERRUPTED_INFRA"
    assert common.failure_class(None,None) == "INTERRUPTED_INFRA"
    assert common.failure_class(1,None) == "FAILED_CODE"
    assert common.failure_class(1,{"classification":"FAILED_DATA"}) == "FAILED_DATA"


def test_queue_requires_certificate_and_respects_deadline(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"RESEARCH",tmp_path)
    r={"experiment_id":"X","ready":True,"launch_certificate":"certificate.json","token_budget":100,"planning_tps":10}
    state={"runs":{},"scientific_stop_epoch":2000}
    assert master.choose([r],state,now=0)==(None,None)
    common.atomic_json(tmp_path/"certificate.json",{"passed":True,"registration_sha256":common.object_hash(r),"source_hashes":{}})
    assert master.choose([r],state,now=0)==(r,"RUNNING")
    assert master.choose([r],state,now=1999)==(None,None)
    state["runs"]["X"]={"status":"FAILED_CODE"}
    assert master.choose([r],state,now=0)==(None,None)


def test_certificate_cannot_be_reused_for_a_changed_registration(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"RESEARCH",tmp_path)
    r={"launch_certificate":"certificate.json", "source_hashes":{"trainer":"abc"}, "token_budget":100}
    common.atomic_json(tmp_path/"certificate.json",{"passed":True,"registration_sha256":common.object_hash(r),"source_hashes":r["source_hashes"]})
    assert master.valid_certificate(r)
    r["token_budget"] = 101
    assert not master.valid_certificate(r)


def test_resume_queue_uses_saved_tokens(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"RESEARCH",tmp_path)
    r={"experiment_id":"X","ready":True,"smoke":True,"token_budget":10000,"planning_tps":10,"evaluation_seconds":0}
    state={"runs":{},"scientific_stop_epoch":100}
    with patch.object(master,"verified_checkpoint",return_value={"logical_tokens":9900}):
        assert master.choose([r],state,only="X",now=0)==(r,"RUNNING")


def test_completion_needs_endpoint_checkpoint_receipt_and_evaluation(tmp_path,monkeypatch):
    common.atomic_json(tmp_path/"train_receipt.json",{"status":"TRAINED","logical_tokens":100,"checkpoint_sha256":"ok"})
    record={"token_budget":100}
    with patch.object(common,"verified_checkpoint",return_value={"logical_tokens":100,"sha256":"ok"}):
        assert not common.complete_evidence(tmp_path,record)
        common.atomic_json(tmp_path/"evaluation_receipt.json",{"status":"COMPLETED","checkpoint_sha256":"wrong"})
        assert not common.complete_evidence(tmp_path,record)
        common.atomic_json(tmp_path/"evaluation_receipt.json",{"status":"COMPLETED","checkpoint_sha256":"ok"})
        assert common.complete_evidence(tmp_path,record)


def test_atomic_replacement_failure_keeps_prior_state(tmp_path):
    path=tmp_path/"state.json";common.atomic_json(path,{"value":1})
    with patch.object(common.os,"replace",side_effect=OSError("interruption")):
        with pytest.raises(OSError):common.atomic_json(path,{"value":2})
    assert json.loads(path.read_text())=={"value":1}


def test_pid_reuse_and_boot_mismatch_rejected():
    actual={"pid":42,"boot_id":"new","start_ticks":"20","process_state":"S"}
    with patch.object(common,"process_identity",return_value=actual):
        assert not common.alive({"pid":42,"boot_id":"old","start_ticks":"20"},"X")
        assert not common.alive({"pid":42,"boot_id":"new","start_ticks":"19"},"X")
