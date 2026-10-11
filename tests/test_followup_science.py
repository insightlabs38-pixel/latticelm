"""Small validity tests for the promotion funnel; no experimental training."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import followup_analysis as analysis
import followup_funnel as funnel
import followup_common as common
import posthack_followup_master as master
import followup_historical as historical
from followup_plan import full_count
from latticelm.config import LatticeConfig
from latticelm.full_co4 import FullCo4LM


@pytest.mark.parametrize('order',[['modulated'],['modulated','modulated'],['modulated','standard'],['standard','standard']])
def test_analytical_exact_parameter_counts(order):
    cfg=LatticeConfig(vocab_size=32,d_model=16,n_heads=4,n_kv_heads=4,n_layers=len(order),context_length=16,ffn_hidden=64,tie_embeddings=False)
    m=FullCo4LM(cfg,block_types=order)
    assert full_count(cfg.to_dict(),order)==sum(p.numel() for p in m.parameters())


def test_historical_import_overrides_fresh_defaults_without_duplicate_kwargs(tmp_path,monkeypatch):
    monkeypatch.setattr(funnel,'ROOT',tmp_path)
    seen={}
    def make(name,cfg,budget,rate,control,**extra):
        seen.update(extra);return {'experiment_id':name,'parameter_count':32,'source_hashes':{},'code_commit':'X',**extra}
    with patch.object(funnel,'registrations',return_value=[]),patch.object(funnel,'make_record',side_effect=make),patch.object(funnel,'register'),patch.object(funnel,'decision'):
        r=funnel.ensure('CONTROL-X',{'config':{'seed':1337}},100,'control',auto_preflight=False,initialization_policy='import')
    assert r['auto_preflight'] is False and seen['initialization_policy']=='import'


def test_imported_endpoint_goes_to_evaluation_without_training_certificate(tmp_path,monkeypatch):
    monkeypatch.setattr(master,'RESEARCH',tmp_path)
    r={'experiment_id':'CONTROL-X','ready':True,'historical_experiment':'X','token_budget':100,'evaluation_seconds':5}
    with patch.object(master,'complete_evidence',return_value=True):
        assert master.choose([r],{'runs':{},'scientific_stop_epoch':1000,'hard_deadline_epoch':2000},now=0)==(r,'EVALUATING')
    with patch.object(master,'complete_evidence',return_value=False):
        assert master.choose([r],{'runs':{},'scientific_stop_epoch':1000},now=0)==(None,None)


def test_missing_legacy_source_is_not_accepted_as_current_trainer():
    with pytest.raises(ValueError,match='not recoverable'):
        historical.verify_historical_control('A8-CO4-S1337',{'trainer_sha256':None},'sha')


def test_primary_controls_wait_for_explicit_choice(tmp_path,monkeypatch):
    monkeypatch.setattr(funnel,'RESEARCH',tmp_path)
    assert funnel.primary_control('DENSE',1337) is None
    common.atomic_json(tmp_path/'HISTORICAL_8M_CONTROL_POLICY.json',{'selection':'waiting_for_source_snapshots'})
    assert funnel.primary_control('DENSE',1337) is None
    common.atomic_json(tmp_path/'HISTORICAL_8M_CONTROL_POLICY.json',{'selection':'fresh_matched_controls'})
    assert funnel.primary_control('DENSE',1337)=='F8-DENSE-FULLCONTROL-100M-S1337'
    assert funnel.primary_control('CO4',2026)=='F8-CO4-FULLCONTROL-100M-S2026'


def test_r32_import_binds_audited_source_and_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(historical,'ROOT',tmp_path)
    monkeypatch.setattr(historical,'RESEARCH',tmp_path)
    trainer=tmp_path/'trainer.py';trainer.write_text('frozen')
    h=common.sha256(trainer)
    # The key is intentionally the actual trainer identity, not any current
    # unrelated source hash; the immutable endpoint also must match.
    target=tmp_path/'scripts/posthack_train.py';target.parent.mkdir();target.write_text('frozen')
    common.atomic_json(tmp_path/'R32_CO4_PROVENANCE_AUDIT.json',{'passed':True,'control':'R32-CO4-CONTROL',
      'checkpoint_sha256':'correct','source_hashes':{'scripts/posthack_train.py':h}})
    assert historical.verify_historical_control('R32-CO4-CONTROL',{'trainer_sha256':h},'correct')['passed']
    with pytest.raises(ValueError,match='endpoint differs'):
        historical.verify_historical_control('R32-CO4-CONTROL',{'trainer_sha256':h},'changed')
    with pytest.raises(ValueError,match='trainer differs'):
        historical.verify_historical_control('R32-CO4-CONTROL',{'trainer_sha256':'unknown'},'correct')
    target.write_text('changed')
    with pytest.raises(ValueError,match='source family changed'):
        historical.verify_historical_control('R32-CO4-CONTROL',{'trainer_sha256':h},'correct')


def test_paired_loss_interval_and_pairing_rejection():
    a={'paired_losses':{'s':[2,3,4,5]}};b={'paired_losses':{'s':[3,4,5,6]}}
    e=analysis.paired_loss(a,b,{'s':1.},draws=50)
    assert e['candidate_minus_control']==-1 and e['ci95']==[-1.,-1.]
    with pytest.raises(ValueError):analysis.paired_loss(a,{'paired_losses':{'s':[1]}},{'s':1.})


def effect(control,loss=-.1,bpb=-.01,quality=True,efficiency=False):
    return {'control':control,'paired_loss':{'candidate_minus_control':loss},'paired_bpb':{'candidate_minus_control':bpb},
      'quality_gain':quality,'efficiency_signal':efficiency,'clear_lm_regression':False,
      'quality_consistent_with_tie':False,'sample_efficiency_strong_threshold_tokens':[],
      'depth_ratio':1,'wall_ratio':1,'capability':{'any_clear_regression':False}}


def test_two_seed_gate_rejects_opposing_quality_direction():
    a=effect('DENSE-S1337');b=effect('DENSE-S2026',loss=.1,bpb=.01,quality=False,efficiency=True)
    assert not analysis.two_seed_consistency([a],[b])['passed']
    b=effect('DENSE-S2026',quality=False)
    assert analysis.two_seed_consistency([a],[b])['passed']
    b['capability']['any_clear_regression']=True
    assert not analysis.two_seed_consistency([a],[b])['passed']


def test_capability_veto_is_paired_and_corrected_over_all_metrics():
    a={};b={}
    for task in ('hellaswag','arc_easy','piqa','winogrande'):
        a[task]=[{'doc_id':i,'target_hash':str(i),'acc':0,'acc_norm':0} for i in range(40)]
        b[task]=[{'doc_id':i,'target_hash':str(i),'acc':1,'acc_norm':1} for i in range(40)]
    e=analysis.paired_tasks(a,b)
    assert e['any_clear_regression'] and e['tests_corrected']==8
    b['piqa'][0]['target_hash']='different'
    with pytest.raises(ValueError):analysis.paired_tasks(a,b)


def test_preflight_queue_transitions_and_deadline(tmp_path,monkeypatch):
    monkeypatch.setattr(master,'RESEARCH',tmp_path)
    r={'experiment_id':'NEW','launch_certificate':'cert.json','ready':True,'auto_preflight':True,'config':{'batch_size':32,'context_length':256},'planning_tps':1000,'token_budget':50_000_000}
    state={'runs':{},'scientific_stop_epoch':10000}
    assert master.choose([r],state,now=0)==(r,'VERIFYING')
    assert master.choose([r],state,now=9999)==(None,None)


def test_explicit_extension_metadata_uses_new_phase_horizon(tmp_path,monkeypatch):
    monkeypatch.setattr(funnel,'RESEARCH',tmp_path);monkeypatch.setattr(funnel,'ROOT',tmp_path)
    cfg={'seed':1337,'batch_size':32,'context_length':256,'learning_rate':.0008,'warmup_fraction':.02,'stable_fraction':.83,'decay_fraction':.15}
    parent={'config':cfg,'parent_control':'CONTROL','schedule':{'peak_lr':.0008}}
    cp={'logical_tokens':100_000_000,'step':12208,'file':'parent.pt','sha256':'sha','wall_seconds':1000}
    with patch.object(funnel,'registered',return_value=parent),patch.object(funnel,'verified_checkpoint',return_value=cp),patch.object(funnel,'ensure') as ensure:
        assert funnel.extension('PARENT')=='PARENT-EXT200M'
    extra=ensure.call_args.kwargs
    assert extra['schedule_total_steps']==12208 and extra['schedule_start_step']==12208
    assert extra['schedule_token_budget']==100_000_000
    assert extra['schedule']['sample_lrs']['12208']==0
    assert extra['schedule']['sample_lrs']['6104']==.0008


def test_master_adopts_live_orphan_without_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(master,'RESEARCH',tmp_path)
    name='ORPHAN';run=tmp_path/'runs'/name
    identity={'pid':9999999,'boot_id':'boot','start_ticks':'42','step':20,'logical_tokens':163840,'phase':'TRAINING'}
    common.atomic_json(run/'heartbeat.json',identity)
    state={'active':None}
    monkeypatch.setattr(master,'STATE',tmp_path/'state.json')
    with patch.object(master,'alive',side_effect=lambda data,_:data['pid']==9999999),patch.object(master,'decision'):
        master.recover_unrecorded_children(state,[{'experiment_id':name}])
    assert state['active']['pid']==9999999 and state['active']['phase']=='RUNNING'


def test_complete_evidence_rejects_modified_evaluation_output(tmp_path):
    output=tmp_path/'result.json';output.write_text('{}')
    common.atomic_json(tmp_path/'train_receipt.json',{'status':'TRAINED','logical_tokens':100,'checkpoint_sha256':'ok'})
    common.atomic_json(tmp_path/'evaluation_receipt.json',{'status':'COMPLETED','checkpoint_sha256':'ok','outputs':[{'file':output.name,'sha256':common.sha256(output),'size_bytes':2}]})
    with patch.object(common,'verified_checkpoint',return_value={'logical_tokens':100,'sha256':'ok'}):
        assert common.complete_evidence(tmp_path,{'token_budget':100})
        output.write_text('[]')
        assert not common.complete_evidence(tmp_path,{'token_budget':100})
