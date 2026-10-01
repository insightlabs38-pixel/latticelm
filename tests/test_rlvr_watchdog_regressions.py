import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import evaluate_posttraining_proxy as proxy
from scripts import watch_rlvr_gate as watchdog


def _torch_interop_smoke(_):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return int(torch.ones(1).item())


def test_rollout_spawn_worker_can_configure_interop_threads():
    # Mirrors the certification pool's spawn context and reproduces the former
    # failure mode if a worker inherits initialized PyTorch parallel state.
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        assert list(pool.map(_torch_interop_smoke, [None])) == [1]


def test_single_candidate_natural_example_is_excluded_from_ranking():
    class Model:
        pass
    with patch.object(proxy, "batch_candidate_scores", return_value=(__import__('numpy').array([1.]), __import__('numpy').array([1.]))):
        row = proxy.score(Model(), [1], [[2]], 0, 8, "natural", {"example_id": "natural-1", "transformation": "identity"})
    assert row["ranking_eligible"] is False
    assert "raw_margin" not in row
    report = proxy.report([row])
    assert report["natural"]["examples"] == 0
    assert report["ranking_exclusions"]["natural_single_candidate"] == 1
    assert report["ranking_exclusions"]["natural_nonranking_examples"] == 1


def test_ambiguous_master_aborts(tmp_path, monkeypatch):
    state = {"schema": "posttraining-master-state-v1", "phase": "SIMPO_TOURNAMENT"}
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state))
    monkeypatch.setattr(watchdog, "STATE", p)
    monkeypatch.setattr(watchdog, "proc_rows", lambda: [
        {"pid": 1, "ppid": 0, "state": "S", "argv": ["python", watchdog.MASTER], "cwd": str(watchdog.ROOT)},
        {"pid": 2, "ppid": 0, "state": "S", "argv": ["python", watchdog.MASTER], "cwd": str(watchdog.ROOT)},
    ])
    with pytest.raises(RuntimeError, match="exactly one master"):
        watchdog.identify()


def test_no_simpo_child_aborts(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"schema": "posttraining-master-state-v1", "phase": "SIMPO_TOURNAMENT"}))
    monkeypatch.setattr(watchdog, "STATE", p)
    monkeypatch.setattr(watchdog, "proc_rows", lambda: [{"pid": 1, "ppid": 0, "state": "S", "argv": ["python", watchdog.MASTER], "cwd": str(watchdog.ROOT)}])
    with pytest.raises(RuntimeError, match="SIMPO child"):
        watchdog.identify()


def test_state_change_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"phase":"SIMPO_TOURNAMENT"}')
    before = watchdog.digest(path)
    path.write_text('{"phase":"RLVR_ELIGIBILITY"}')
    with pytest.raises(RuntimeError, match="changed unexpectedly"):
        watchdog.require_unchanged(path, before)


def test_certification_pass_state_patch_preserves_scientific_fields(tmp_path, monkeypatch):
    cert_path = tmp_path / "cert.json"
    cert_path.write_text('{"status":"PASS"}')
    monkeypatch.setattr(watchdog, "CERT", cert_path)
    state = {"rollout_backend": "REJECTED_OPTIONAL", "capability_map": {"x": 0}, "rft_eligible": False, "optional_failures": [{"label": "old failure"}]}
    cert = {"selected": {"processes": 1, "online_worker_supported": True}}
    updated = watchdog.recovery_state(state, cert, simpo={"status": "COMPLETE"}, now="fixed-time")
    assert updated["rollout_backend"] == "CERTIFIED"
    assert updated["rollout_topology"] == cert["selected"]
    assert updated["capability_map"] == state["capability_map"]
    assert updated["rft_eligible"] is False
    assert updated["optional_failures"] == state["optional_failures"]
    assert updated["rollout_recertification"]["status"] == "PASS"


def test_certification_failure_does_not_mark_backend_certified():
    state = {"rollout_backend": "REJECTED_OPTIONAL", "rollout_topology": None}
    updated = watchdog.recovery_state(state, failure="benchmark failed", now="fixed-time")
    assert updated["rollout_backend"] == "REJECTED_OPTIONAL"
    assert updated["rollout_topology"] is None
    assert updated["rollout_recertification"]["status"] == "FAIL"


def test_completed_simpo_result_is_reused_after_restart(tmp_path, monkeypatch):
    from scripts import run_posttraining_master as master
    out = tmp_path / "candidates" / "simpo"
    out.mkdir(parents=True)
    result = {"status": "COMPLETE", "training_tokens": 2500000, "checkpoint": "already-trained.pt"}
    (out / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(master, "PT", tmp_path)
    assert master.worker({}, "simpo", Path("unused-parent"), "simpo", 2500000) == result


def test_watchdog_lock_allows_only_one_owner(tmp_path):
    import fcntl
    lock = tmp_path / "watchdog.lock"
    first = lock.open("a+")
    second = lock.open("a+")
    fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(BlockingIOError):
        fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
    first.close()
    second.close()


def test_valid_safe_stopped_simpo_is_persisted_as_terminal_candidate(tmp_path, monkeypatch):
    cert_path = tmp_path / "cert.json"
    cert_path.write_text("{}")
    monkeypatch.setattr(watchdog, "CERT", cert_path)
    state = {"candidates": {"BASE": {"status": "CERTIFIED"}}}
    worker_result = {"status": "SAFE_STOPPED_VALID", "method": "simpo", "training_tokens": 1234, "checkpoint": "simpo.pt", "checkpoint_sha256": "sha", "method_chain": ["joint", "simpo"], "wall_seconds": 60}
    updated = watchdog.recovery_state(state, {"selected": {}}, simpo={"status": "SAFE_STOPPED_VALID", "requested_tokens": 2500000, "worker_result": worker_result}, now="fixed-time")
    assert updated["candidates"]["simpo"]["training_tokens"] == 1234
    assert updated["candidates"]["simpo"]["status"] == "SAFE_STOPPED_VALID"
    assert updated["watchdog_simpo_outcome"]["status"] == "SAFE_STOPPED_VALID"
    assert "worker_result" not in updated["watchdog_simpo_outcome"]
    assert updated["simpo_early_stop"]["actual_training_tokens"] == 1234


def test_safe_stopped_candidate_is_reused_without_resume(tmp_path, monkeypatch):
    from scripts import run_posttraining_master as master
    import hashlib
    out = tmp_path / "candidates" / "simpo"
    out.mkdir(parents=True)
    checkpoint = out / "latest.pt"
    checkpoint.write_bytes(b"valid test checkpoint")
    result = {"schema": "posttraining-worker-result-v2", "status": "SAFE_STOPPED_VALID",
              "method": "simpo", "training_tokens": 12, "checkpoint": str(checkpoint),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (out / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(master, "PT", tmp_path)
    assert master.worker({}, "simpo", Path("unused-parent"), "simpo", 1668397) == result


def test_expensive_method_does_not_inherit_sft_throughput():
    from scripts import run_posttraining_master as master
    plan = master.schedule_decision({"throughput": {"sft": 10000}}, "simpo", 1_000_000, minimum=1)
    assert plan["measured_logical_tokens_per_second"] == 1.0
