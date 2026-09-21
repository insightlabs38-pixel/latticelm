import hashlib,json
from pathlib import Path
import torch
import scripts.run_production_checkpoint_evaluation_companion as companion
def payload(tokens=1_000_007):return {"schema":"final-production-checkpoint-v1","model":{},"optimizers":{},"config":{"learning_rate":.0008},"config_sha256":"c"*64,"tokens":tokens,"step":tokens//8192,"manifest_sha256":"m"*64,"tokenizer_sha256":"t"*64,"next_batch_sha256":"n"*64,"validation_loss":2.9,"source_validation":{"fineweb":2.9}}
def write_checkpoint(root,tokens=1_000_007):
 run=root/"final_production_run";run.mkdir();path=run/"latest.pt";torch.save(payload(tokens),path);digest=hashlib.sha256(path.read_bytes()).hexdigest();(run/"latest.sha256").write_text(digest+"\n");return path,digest
def test_partial_or_bad_sidecar_is_rejected(tmp_path,monkeypatch):
 path,digest=write_checkpoint(tmp_path);monkeypatch.setattr(companion,"RUN",path.parent);assert companion.latest_checkpoint();(path.parent/"latest.sha256").write_text("bad\n");assert companion.latest_checkpoint() is None;(path.parent/"latest.sha256").write_text(digest+"\n");path.write_bytes(path.read_bytes()+b"partial");assert companion.latest_checkpoint() is None
def test_first_checkpoint_at_or_after_and_exact_preference(tmp_path,monkeypatch):
 events=tmp_path/"events.jsonl";events.write_text("\n".join(json.dumps(x) for x in [{"event":"CHECKPOINT","at":1,"tokens":999_999_999,"sha256":"a"},{"event":"CHECKPOINT","at":3,"tokens":1_000_020_000,"sha256":"c"},{"event":"CHECKPOINT","at":2,"tokens":1_000_000_000,"sha256":"b"}])+"\n");monkeypatch.setattr(companion,"EVENTS",events);assert companion.first_checkpoint_at_or_after(1_000_000_000)["tokens"]==1_000_000_000
def test_restart_state_is_idempotent_and_terminal_permanent(tmp_path,monkeypatch):
 state=companion.new_state();state["completed_targets"]=[1_000_000_000,1_250_000_000,1_500_000_000];state["status"]="COMPLETE";path=tmp_path/"state.json";path.write_text(json.dumps(state));monkeypatch.setattr(companion,"STATE",path);loaded=companion.load_state();assert companion.next_target(loaded) is None and loaded["status"]=="COMPLETE"
def test_pin_uses_target_directory_and_records_identity(tmp_path,monkeypatch):
 path,digest=write_checkpoint(tmp_path);monkeypatch.setattr(companion,"ART",tmp_path);monkeypatch.setattr(companion,"STATE",tmp_path/"state.json");monkeypatch.setattr(companion,"EVENT_LOG",tmp_path/"events.jsonl");state=companion.new_state();record={"event":"CHECKPOINT","at":20,"tokens":1_000_007,"sha256":digest,"effective_tok_s":2800};identity=companion.pin_checkpoint(state,1_000_000_000,record,(path,payload(),digest));assert identity["target_tokens"]==1_000_000_000;assert Path(identity["checkpoint"]).parent.name=="production_milestone_eval_1000m";assert hashlib.sha256(Path(identity["checkpoint"]).read_bytes()).hexdigest()==digest
def test_migration_does_not_mark_scheduled_targets_complete(tmp_path,monkeypatch):
 path=tmp_path/"state.json";path.write_text(json.dumps({"schema":"production-eval-companion-v1","status":"COMPLETE","checkpoint":{"tokens":750_000_000}}));monkeypatch.setattr(companion,"STATE",path);state=companion.load_state();assert state["completed_targets"]==[] and state["legacy_evaluations"][0]["tokens"]==750_000_000
def test_schedule_scale_and_monotonic_targets():assert companion.expected_lr_scale(1_000_000_000,1_900_000_000)==1.0 and companion.TARGETS==tuple(sorted(companion.TARGETS))
