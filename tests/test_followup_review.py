"""Regression cases discovered in the independent prompt/code review."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import followup_common as common
import followup_evaluate as evaluate
import followup_analysis as analysis
import posthack_followup_master as master
import followup_archive as archive


def test_public_evaluator_identity_is_bound_to_checkpoint_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(common,'RESEARCH',tmp_path/'research')
    identity={'pid':42,'boot_id':'boot','start_ticks':'17'}
    cmd=[b'python',str(tmp_path/'scripts/evaluate_gibc.py').encode(),b'--checkpoint',
         str(tmp_path/'research/runs/X/checkpoint.pt').encode()]
    with patch.object(common,'process_identity',return_value={**identity,'process_state':'R'}),patch.object(Path,'read_bytes',side_effect=lambda:b'\0'.join(cmd)):
        assert common.alive(identity,'X')
        assert not common.alive(identity,'Y')
        cmd[1]=b'/unrelated/evaluate_gibc.py'
        assert not common.alive(identity,'X')


def test_clean_partial_trainer_exit_is_infrastructure_interruption(tmp_path,monkeypatch):
    monkeypatch.setattr(master,'RESEARCH',tmp_path)
    common.atomic_json(tmp_path/'runs/X/train_receipt.json',{'status':'INTERRUPTED_INFRA','logical_tokens':20,'checkpoint_sha256':'ok'})
    active={'phase':'RUNNING'};record={'experiment_id':'X'}
    with patch.object(master,'verified_checkpoint',return_value={'logical_tokens':20,'sha256':'ok'}):
        assert master.disappeared_classification(active,record,0,None)=='INTERRUPTED_INFRA'
    with patch.object(master,'verified_checkpoint',return_value={'logical_tokens':19,'sha256':'other'}):
        assert master.disappeared_classification(active,record,0,None)=='FAILED_CODE'
    assert master.disappeared_classification(active,record,0,{'classification':'FAILED_DATA'})=='FAILED_DATA'


def test_evaluator_recovers_after_predictions_commit_before_result(tmp_path):
    output=tmp_path/'capability_full.json';temporary=tmp_path/'.capability_full.json.tmp'
    samples=tmp_path/'.capability_samples.json.tmp'
    common.atomic_json(temporary,{'checkpoint_sha256':'same'});common.atomic_json(samples,{'task':[{'doc_id':1}]})
    original=evaluate.os.replace
    def interrupted(source,destination):
        if Path(destination)==output:raise OSError('simulated preemption between publications')
        return original(source,destination)
    with patch.object(evaluate.os,'replace',side_effect=interrupted):
        with pytest.raises(OSError):evaluate.publish_outputs(tmp_path,temporary,output,True)
    assert (tmp_path/'capability_samples.json').exists() and temporary.exists() and not output.exists()
    evaluate.publish_outputs(tmp_path,temporary,output,True)
    assert output.exists() and not temporary.exists()


def test_legacy_result_before_predictions_publication_is_recoverable(tmp_path):
    # Previous wrapper versions committed results before predictions. The
    # surviving staged predictions are preserved and can still be committed.
    output=tmp_path/'capability_full.json';temporary=tmp_path/'.capability_full.json.tmp'
    common.atomic_json(output,{'checkpoint_sha256':'same'})
    common.atomic_json(temporary,{'checkpoint_sha256':'same'})
    common.atomic_json(tmp_path/'.capability_samples.json.tmp',{'task':[{'doc_id':1}]})
    evaluate.publish_outputs(tmp_path,temporary,output,True)
    assert (tmp_path/'capability_samples.json').exists()


def test_capability_pairs_reject_changed_documents_or_prompts():
    a={task:[{'doc_id':1,'doc_hash':'doc','prompt_hash':'prompt','target_hash':'target','acc':1}]
       for task in ('hellaswag','arc_easy','piqa','winogrande')}
    b=json.loads(json.dumps(a));b['piqa'][0]['doc_hash']='different'
    with pytest.raises(ValueError,match='Unpaired'):analysis.paired_tasks(a,b)
    b=json.loads(json.dumps(a));b['piqa'][0]['prompt_hash']='different'
    with pytest.raises(ValueError,match='Unpaired'):analysis.paired_tasks(a,b)


def test_real_registration_cannot_complete_without_required_output_hashes(tmp_path):
    common.atomic_json(tmp_path/'train_receipt.json',{'status':'TRAINED','logical_tokens':100,'checkpoint_sha256':'ok'})
    common.atomic_json(tmp_path/'evaluation_receipt.json',{'status':'COMPLETED','checkpoint_sha256':'ok','outputs':[]})
    with patch.object(common,'verified_checkpoint',return_value={'logical_tokens':100,'sha256':'ok'}):
        assert not common.complete_evidence(tmp_path,{'schema':'followup-run-registration-v1','token_budget':100})


def test_compressed_index_roundtrip_is_verified_and_cached(tmp_path,monkeypatch):
    import gzip
    monkeypatch.setattr(archive,'ROOT',tmp_path);monkeypatch.setattr(archive,'RESEARCH',tmp_path/'research')
    monkeypatch.setattr(archive,'cached_hash',common.sha256)
    base=tmp_path/'post_hackathon_research/research_d';base.mkdir(parents=True)
    compressed=base/'documents.jsonl.gz';payload=b'{"document":"preserved"}\n'
    with gzip.open(compressed,'wb') as handle:handle.write(payload)
    original=tmp_path/'original';original.write_bytes(payload)
    compressed.with_name(compressed.name+'.source.sha256').write_text(common.sha256(original)+'\n')
    archive.verify_compressed_indexes()
    with patch.object(gzip,'open',side_effect=AssertionError('unchanged gzip should not be decompressed again')):
        archive.verify_compressed_indexes()
    compressed.with_name(compressed.name+'.source.sha256').write_text('0'*64)
    with pytest.raises(ValueError,match='different bytes'):archive.verify_compressed_indexes()


def test_scarce_time_uses_actual_remaining_work():
    r={'experiment_id':'F16-D4-S2026-FRESH150M','token_budget':150_000_000,'planning_tps':7000,'evaluation_seconds':1800}
    state={'runs':{},'scientific_stop_epoch':80_000}
    assert master.scarce_time([r],state,0)
    state['scientific_stop_epoch']=200_000
    assert not master.scarce_time([r],state,0)


def test_scarce_queue_puts_source_compatibility_before_d4(tmp_path,monkeypatch):
    monkeypatch.setattr(master,'RESEARCH',tmp_path)
    d4={'experiment_id':'F16-D4-S2026-FRESH150M','token_budget':150_000_000,'planning_tps':7000,'evaluation_seconds':1800,
       'ready':True,'auto_preflight':True,'launch_certificate':'d4.json','config':{'batch_size':32,'context_length':256}}
    full={'experiment_id':'FCB-C1-BASE-S1337','token_budget':5_000_000,'planning_tps':800,'evaluation_seconds':60,
       'ready':True,'auto_preflight':True,'launch_certificate':'full.json','config':{'batch_size':32,'context_length':256},
       'full_co4':{'block_types':['modulated','modulated']}}
    state={'runs':{},'scientific_stop_epoch':80_000}
    assert master.choose([d4,full],state,now=0)==(full,'VERIFYING')
    state['scientific_stop_epoch']=200_000
    assert master.choose([d4,full],state,now=0)==(d4,'VERIFYING')


def test_metric_reader_tolerates_only_torn_final_append(tmp_path):
    path=tmp_path/'metrics.jsonl';path.write_bytes(b'{"step":1}\n{"step":')
    assert common.metric_rows(path)==[{'step':1}]
    path.write_bytes(b'{"step":1}\n{"step":\n')
    with pytest.raises(ValueError,match='committed'):common.metric_rows(path)
    path.write_bytes(b'broken\n{"step":2}')
    with pytest.raises(ValueError,match='committed'):common.metric_rows(path)


def test_restart_promotes_known_good_fallback_without_deleting_weights(tmp_path):
    head=tmp_path/'bad.pt';head.write_bytes(b'damaged')
    prior=tmp_path/'good.pt';prior.write_bytes(b'valid')
    good={'file':prior.name,'sha256':common.sha256(prior),'size_bytes':5,'logical_tokens':20}
    bad={'file':head.name,'sha256':'wrong','size_bytes':7,'logical_tokens':30}
    common.atomic_json(tmp_path/'checkpoint_manifest.json',{'rolling':[bad,good],'milestones':[]})
    common.reconcile_rolling_manifest(tmp_path)
    assert json.loads((tmp_path/'checkpoint_manifest.json').read_text())['rolling']==[good]
    assert head.exists() and prior.exists() and (tmp_path/'CHECKPOINT_FALLBACK_RECOVERY.json').exists()


def test_source_validation_recovery_from_committed_endpoint(tmp_path,monkeypatch):
    import torch
    import followup_train
    import train_final_production
    monkeypatch.setattr(evaluate,'ROOT',tmp_path)
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    record={'manifest_path':manifest.name,'sampling_manifest_sha256':common.sha256(manifest),'config':{'context_length':16},'threads':1}
    item={'file':'checkpoint.pt','sha256':'current','step':100,'logical_tokens':1000}
    torch.save({'model':{'weight':torch.ones(1)}},tmp_path/item['file'])
    common.atomic_json(tmp_path/'latest_validation.json',{'checkpoint_sha256':'previous'})
    model=type('Model',(),{'load_state_dict':lambda self,*_,**__:None})()
    with patch.object(followup_train,'build_model',return_value=model),patch.object(followup_train,'evaluate_sources',return_value=({'s':2.},{'s':[2.]})),patch.object(train_final_production,'load_arrays',return_value=({'mixture_definition':{'s':1.}},None,None)):
        result=evaluate.ensure_source_validation(record,tmp_path,item)
    assert result['checkpoint_sha256']=='current' and result['frozen_mixture_weighted_loss']==2.
    assert (tmp_path/'SOURCE_VALIDATION_RECOVERY.json').exists()


def test_differential_replication_is_causal_and_has_finite_gradients():
    import torch
    from latticelm.config import LatticeConfig
    from latticelm.posthack_variants import DifferentialCo4LM
    torch.set_num_threads(1);torch.manual_seed(32)
    cfg=LatticeConfig(vocab_size=32,d_model=32,ffn_hidden=64,n_heads=4,n_kv_heads=2,n_layers=2,
                      context_length=16,architecture='co4_causal',real_gqa=True,qk_norm=True,dropout=0.)
    model=DifferentialCo4LM(cfg).eval();tokens=torch.randint(0,32,(2,15));output=model(tokens)[0]
    changed=tokens.clone();changed[:,7:]=torch.randint(0,32,changed[:,7:].shape)
    torch.testing.assert_close(output[:,:7],model(changed)[0][:,:7],atol=1e-6,rtol=0)
    torch.testing.assert_close(output[:,:7],model(tokens[:,:7])[0],atol=1e-6,rtol=0)
    output.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_wall_accounting_includes_resume_downtime(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'RESEARCH',tmp_path)
    record={'experiment_id':'X'}
    receipt={'at':'2026-10-11T02:00:00+00:00','wall_seconds':1800}
    common.atomic_json(tmp_path/'runs/X/initialization_receipt.json',{'at':'2026-10-11T01:00:00+00:00'})
    result=analysis.training_wall(record,receipt)
    assert result['retained_training_seconds']==1800
    assert result['observed_elapsed_seconds']==3600


def test_wall_accounting_does_not_invent_missing_start(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'RESEARCH',tmp_path)
    assert analysis.training_wall({'experiment_id':'X'},{'at':'2026-10-11T02:00:00+00:00','wall_seconds':1800})['observed_elapsed_seconds'] is None


def test_wall_accounting_rejects_reversed_times(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'RESEARCH',tmp_path)
    common.atomic_json(tmp_path/'runs/X/initialization_receipt.json',{'at':'2026-10-11T03:00:00+00:00'})
    with pytest.raises(ValueError,match='Nonpositive'):
        analysis.training_wall({'experiment_id':'X'},{'at':'2026-10-11T02:00:00+00:00','wall_seconds':1800})


def test_closeout_publication_failure_does_not_mark_campaign_finalized(tmp_path,monkeypatch):
    import followup_finalize as finalize
    monkeypatch.setattr(finalize,'RESEARCH',tmp_path)
    state={'runs':{}}
    report_states=[]
    with patch.object(finalize,'registrations',return_value=[]),patch.object(finalize,'reports',side_effect=lambda s,**kw:report_states.append(s.get('finalized',False))),patch.object(archive,'archive',return_value={'files':{}}),patch.object(archive,'representative_loads',return_value={'passed':True}),patch.object(finalize,'publish_git',side_effect=RuntimeError('remote publication failed')):
        with pytest.raises(RuntimeError,match='publication failed'):finalize.finalize(state)
    assert not state.get('finalized') and report_states[-1] is False
