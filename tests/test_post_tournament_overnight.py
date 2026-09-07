from datetime import datetime, timedelta, timezone

import scripts.run_post_tournament_overnight as master


def test_predecessor_wait_active_then_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(master, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(master, "EVENTS", tmp_path / "events.jsonl")
    states=iter([{"ActiveState":"active","SubState":"running"},{"ActiveState":"inactive","SubState":"dead","Result":"success"}])
    monkeypatch.setattr(master.time,"sleep",lambda _:None)
    state={"run_id":"x","current_stage":"STARTING"}
    master.wait_for_predecessor(state,0,lambda:next(states))
    assert state["predecessor_state"]["ActiveState"]=="inactive"


def test_deadline_gate_is_conservative():
    now=datetime.now(timezone.utc)
    state={"soft_deadline":(now+timedelta(hours=13)).isoformat(),"hard_deadline":(now+timedelta(hours=14)).isoformat(),"observed_throughput":5000}
    assert master.can_start(state,25_000_000,50_000_000)
    assert not master.can_start(state,0,1_000_000_000)


def test_branch_selection_ignores_tiny_deltas():
    a={**master.CONTROL,"run_id":"LR-10","lr_fraction":.1}
    b={**master.CONTROL,"run_id":"LR-25","lr_fraction":.25}
    for task in master.TASKS:a[task]+=.001;b[task]+=.002
    classification,_=master.tournament_choice([master.CONTROL,a,b])
    assert classification=="D"


def test_exact_resume_rejects_missing_fields(tmp_path,monkeypatch):
    monkeypatch.setattr(master,"ART",tmp_path)
    root=tmp_path/"checkpoints/LR-10";root.mkdir(parents=True);path=root/"latest.pt"
    import torch
    torch.save({"run_id":"LR-10","tokens_seen":10},path)
    path.with_suffix(".sha256").write_text(master.digest(path)+"\n")
    assert master.checkpoint("LR-10") is None
