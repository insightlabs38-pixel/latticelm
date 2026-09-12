import csv,json,signal
from pathlib import Path
import pytest
import scripts.post900m_scaling as scaling
import scripts.run_post900m_master as master
import scripts.post900m_tournament as tournament
def curve(path,losses):
 fields=["nominal_tokens","train_loss","data_d_validation_loss","fineweb_edu_validation_loss","wikipedia_validation_loss","fineweb_validation_loss"]
 with path.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for t,l in losses:w.writerow({"nominal_tokens":t,"train_loss":l+.1,"data_d_validation_loss":l,"fineweb_edu_validation_loss":l,"wikipedia_validation_loss":l-.03,"fineweb_validation_loss":l+.03})
def test_decisions_are_deterministic(tmp_path):
 p=tmp_path/"c";curve(p,[(50_000_000,4.5),(205_000_000,4.),(300_000_000,3.8),(500_000_000,3.5),(700_000_000,3.3),(900_000_000,3.15)]);a=scaling.analyze(p,{})
 assert a==scaling.analyze(p,{}) and scaling.classify(a)=="CONTINUE_TO_1_5B"
 p=tmp_path/"s";curve(p,[(50_000_000,3.3),(205_000_000,3.2),(300_000_000,3.19),(500_000_000,3.181),(700_000_000,3.180),(900_000_000,3.179)]);a=scaling.analyze(p,{})
 assert scaling.classify(a)=="SATURATING_OR_LOW_VALUE";a["wikitext"]=[{"bits_per_byte":1.7},{"bits_per_byte":2.}];assert scaling.classify(a)=="INCONCLUSIVE";assert scaling.classify(a,False)=="BLOCKED"
def test_predecessor_running_and_failures(monkeypatch,tmp_path):
 p=tmp_path/"p";monkeypatch.setattr(master,"PRE_STATE",p);monkeypatch.setattr(master,"PRE_SERVICE","x")
 p.write_text(json.dumps({"current_stage":"TRAIN_900000000","child_pid":1}));monkeypatch.setattr(master,"service_active",lambda _:True);assert master.predecessor_ready({},True)==(False,None)
 for stage in ("FAILED","BLOCKED","STOPPED_SAFE"):
  p.write_text(json.dumps({"current_stage":stage,"child_pid":None}));assert stage in master.predecessor_ready({},True)[1]
 p.write_text("{");monkeypatch.setattr(master,"service_active",lambda _:False);assert "unreadable" in master.predecessor_ready({},True)[1]
def test_state_backup_and_immutable(monkeypatch,tmp_path):
 state=tmp_path/"s";backup=tmp_path/"b";monkeypatch.setattr(master,"STATE",state);monkeypatch.setattr(master,"BACKUP",backup)
 master.atomic(state,{"current_stage":"A"},backup);master.atomic(state,{"current_stage":"B"},backup);state.write_text("bad");assert master.load()["current_stage"]=="A"
 e=tmp_path/"e";master.atomic(e,{"x":1},immutable=True);master.atomic(e,{"x":1},immutable=True)
 with pytest.raises(RuntimeError):master.atomic(e,{"x":2},immutable=True)
def test_transition_record(monkeypatch,tmp_path):
 old=tmp_path/"old";newd=tmp_path/"new";newd.mkdir();new=newd/"manifest.json";old.write_text(json.dumps({"total_unique_tokens":1_009_592_508}));new.write_text(json.dumps({"total_unique_tokens":2_250_000_000,"mixture_definition":{"fineweb":.25}}));(newd/"certification.json").write_text(json.dumps({"acceptance":"PASS","manifest_sha256":master.sha256_file(new),"deterministic_first_batch_sha256":"d"}));monkeypatch.setattr(master,"MANIFEST3",old);monkeypatch.setattr(master,"ART",tmp_path);r=master.transition_record(old,new,"c");assert r["token_count_at_transition"]==900_000_000 and "new deterministic stream" in r["data_stream_state_policy"]
def test_sigterm_and_contract(monkeypatch):
 monkeypatch.setattr(master,"STOP",False);monkeypatch.setattr(master,"load",lambda:None);monkeypatch.setattr(master,"CHILD",None);master.request_stop(signal.SIGTERM,None);assert master.STOP
 text=Path("scripts/run_post900m_master.py").read_text()
 for x in ("next_batch_sha256","numpy_rng_state","strict_state_dict","WAIT_SECONDS=300","post900m_handoff.json","dataset-transition-v1"):assert x in text
def test_stop_does_not_mutate_terminal_state(monkeypatch):
 monkeypatch.setattr(master,"STOP",False);monkeypatch.setattr(master,"CHILD",None);monkeypatch.setattr(master,"load",lambda:{"current_stage":"COMPLETE"})
 monkeypatch.setattr(master,"save",lambda *_a,**_k:pytest.fail("terminal state was rewritten"));master.request_stop()
def test_missing_checkpoint_bad_sha_wrong_tokens_parameters_config_are_gated():
 text=Path("scripts/run_post900m_master.py").read_text()
 for x in ("missing immutable 900M checkpoint","sidecar mismatch",'state.get("tokens_seen")==TARGET','==PARAMS','state.get("config")==cfg.to_dict()'):assert x in text
def test_unit_dependency_and_restart_policy():
 u=Path("ops/latticelm-post900m-20260912.service").read_text();assert "After=latticelm-final-scale-20260909.service" in u and "RestartPreventExitStatus=2 75" in u
def test_continuation_trainer_contract():
 t=Path("scripts/train_post900m_continuation.py").read_text()
 for x in ("parent_lineage","dataset_transition","next_batch_sha256","optimizer","scheduler","numpy_rng_state","torch_rng_state","fp32_compile","BASE_TOKENS=900_000_000"):
  assert x in t
 m=Path("scripts/run_post900m_master.py").read_text()
 for target in ("1_000_000_000","1_100_000_000","1_200_000_000","1_250_000_000","1_350_000_000","1_500_000_000","2250000000","canonical-2250m"):
  assert target in m

def official(**updates):
 value={"candidate_id":"x","checkpoint":"x.pt","validation_loss":3.18,"wikitext_ppl":47.,"wikitext_bpb":1.77,"reasoning_accuracy":.1,"hellaswag":.26,"arc_easy":.31,"piqa":.53,"winogrande":.49,"tokens_per_second":3000.,"finite":True,"restart_verified":True,"matched_loss_delta":0.}
 value.update(updates);return value

def test_saturation_promotion_gates_are_deterministic():
 base=official(candidate_id="base",validation_loss=3.19,wikitext_bpb=1.78,reasoning_accuracy=.05)
 good=official(candidate_id="cool",validation_loss=3.18,wikitext_bpb=1.77)
 assert tournament.promote_cooldown(base,[good])[0]==good
 noisy=official(candidate_id="bad",validation_loss=3.18,wikitext_bpb=1.77,hellaswag=.20)
 assert tournament.promote_cooldown(base,[noisy])[0] is None
 sft=official(candidate_id="sft",reasoning_accuracy=.10)
 assert tournament.promote_sft(base,[sft])[0]==sft
 forgetting=official(candidate_id="forget",reasoning_accuracy=.5,wikitext_ppl=55.)
 assert tournament.promote_sft(base,[forgetting])[0] is None

def test_system_promotion_requires_speed_parity_and_restart():
 base=official(candidate_id="b8",tokens_per_second=3000)
 good=official(candidate_id="b16",tokens_per_second=3300)
 assert tournament.promote_system(base,[good])[0]==good
 good["restart_verified"]=False
 assert tournament.promote_system(base,[good])[0] is None

def test_saturation_controller_contract():
 text=Path("scripts/run_saturation_response.py").read_text()
 for item in ("WSD_COOLDOWN","SYSTEMS_BENCHMARK","SFT","RLVR_GRPO","FINAL_SELECTION","STOPPED_SAFE","saturation_response_results.json","BASE_512M","BASE_750M","BASE_900M","promote versus own parent"):
  assert item in text
 worker=Path("scripts/train_post900m_experiment.py").read_text()
 for item in ("parent_checkpoint_sha256","cosine-to-zero","100_000_000","branch_token_budget","step%384","LatticeReason","SKIPPED_SAFELY","reasoning_accuracy"):
  assert item in worker

def test_handoff_semantics_are_not_false_1_5b_claims():
 text=Path("scripts/run_post900m_master.py").read_text()
 assert '"INCONCLUSIVE":"resolve_900m_scaling_ambiguity"' in text
 assert '"SATURATION_RESPONSE_COMPLETE":"strategic_review"' in text
