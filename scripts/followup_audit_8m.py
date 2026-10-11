"""Bounded, per-run historical audits against recovered exact source snapshots.

This reads historical evidence and writes new certificates. It never launches
the old trainer, updates historical files, or changes a scientific run.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
from followup_common import ROOT,RESEARCH,atomic_json,metric_rows,sha256,utc


def payload_keys(source):
    tree=ast.parse(source)
    payloads=[node.value for node in ast.walk(tree) if isinstance(node,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='payload' for t in node.targets)
              and isinstance(node.value,ast.Dict)]
    if len(payloads)!=1:raise ValueError('Ambiguous checkpoint payload in recovered trainer')
    return {key.value for key in payloads[0].keys}


def original_root_path(source,name):
    """Read the recovered trainer's explicit ROOT/path constant."""
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            value=node.value
            if isinstance(value,ast.BinOp) and isinstance(value.op,ast.Div) and isinstance(value.left,ast.Name) and value.left.id=='ROOT' and isinstance(value.right,ast.Constant):
                return value.right.value
    raise ValueError('Unrecoverable original path constant: '+name)


def bind_matched_implementation(result):
    """Bind exact shared sources and the actual baseline optimizer partition."""
    import torch
    torch.set_num_threads(1)
    from latticelm.config import LatticeConfig
    from latticelm.model import build_model
    from latticelm.final_recipe import muon_parameter_partition
    snapshot=RESEARCH/'provenance_recovery/recovered'/result['executed_trainer_sha256']/'snapshot'
    paths=('src/latticelm/model.py','src/latticelm/config.py','src/latticelm/data_d.py',
           'src/latticelm/co4_reference.py','src/latticelm/memory.py','scripts/train_final_production.py')
    for rel in paths:
        if sha256(ROOT/rel)!=sha256(snapshot/rel):raise ValueError('Matched mathematical source differs: '+rel)
    def function(source,name):
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    archived=(snapshot/'src/latticelm/final_recipe.py').read_text();current=(ROOT/'src/latticelm/final_recipe.py').read_text()
    for name in ('make_optimizer','wsd_lr_scale'):
        if function(archived,name)!=function(current,name):raise ValueError('Matched optimizer/schedule function differs: '+name)
    old=ROOT/'post_hackathon_research';record=next(r for r in metric_rows(old/'RUN_REGISTRY.jsonl') if r['experiment_id']==result['control'])
    model=build_model(LatticeConfig(**result['config']))
    if muon_parameter_partition(model)[2]!=record['optimizer_partition']:raise ValueError('Matched optimizer partition differs')
    cfg=model.config
    if cfg.dropout or cfg.memory_enabled:raise ValueError('Old evaluator training-mode behavior needs further audit')
    probe=torch.arange(8,dtype=torch.long)[None,:]
    with torch.no_grad():
        model.train();train_output=model(probe)[0];model.eval();eval_output=model(probe)[0]
    if not torch.equal(train_output,eval_output):raise ValueError('Baseline train/eval mode changes outputs')
    if result['control']!='A8-CO4-S1337':
        old_tree=ast.parse((snapshot/'src/latticelm/posthack_variants.py').read_text())
        new_tree=ast.parse((ROOT/'src/latticelm/posthack_variants.py').read_text())
        branches=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_posthack_model').body[0] for tree in (old_tree,new_tree)]
        if ast.dump(branches[0],include_attributes=False)!=ast.dump(branches[1],include_attributes=False):raise ValueError('Baseline variant factory differs')
    result['checks'].update(matched_exact_shared_source_bytes=True,matched_optimizer_and_wsd_function_ast=True,
        matched_actual_baseline_optimizer_partition=True,baseline_train_eval_mode_invariant=True)
    result['matched_current_source_hashes']={p:sha256(ROOT/p) for p in (*paths,'src/latticelm/final_recipe.py','src/latticelm/posthack_variants.py')}
    result['matched_implementation_note']='Original 841 optimizer partition source lacks the later router-name exclusion. Actual baseline parameter partitions agree exactly; make_optimizer and WSD ASTs are identical. Model/config/data/memory/Co4/shared sampler bytes match exactly. Recovered baseline factories return the same original build_model. Old validation left eval mode active, which is output-invariant for these dropout-zero, memory-disabled baselines.'
    result['matched_binding_at']=utc()
    atomic_json(RESEARCH/'provenance_recovery'/f"{result['control']}_AUDIT.json",result)
    return result


def audit(name,snapshot,chain):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    # Each invocation is a fresh process: original snapshot imports cannot be
    # shadowed by already-imported current mathematical implementations.
    sys.path[:0]=[str(snapshot/'src'),str(snapshot/'scripts')]
    from latticelm.config import LatticeConfig
    from latticelm.data_d import SourceStream
    from latticelm.final_recipe import make_optimizer,wsd_lr_scale
    from latticelm.model import build_model
    from train_final_production import WeightedMixture,next_batch_hash
    old=ROOT/'post_hackathon_research';run=old/'runs'/name
    records=[r for r in metric_rows(old/'RUN_REGISTRY.jsonl') if r['experiment_id']==name]
    if len(records)!=1:raise ValueError('Ambiguous historical registration')
    record=records[0];events=[e for e in metric_rows(old/'controller_events.jsonl') if e.get('experiment_id')==name]
    cfg=LatticeConfig(**record['config']);target=record['token_budget'];quantum=cfg.batch_size*cfg.context_length
    steps=math.ceil(target/quantum);receipt=json.loads((run/'result.json').read_text());digest=sha256(run/'latest.pt')
    checkpoint=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
    checks={}
    def check(condition,label):
        checks[label]=bool(condition)
        if not condition:raise ValueError(name+': '+label)
    counts=Counter(e['event'] for e in events)
    check(counts['TRAIN_START']==counts['TRAIN_END']==1 and counts['RESUMED']==0,'fresh_uninterrupted_event_history')
    start=next(e for e in events if e['event']=='TRAIN_START')
    check(digest==receipt['checkpoint_sha256']==(run/'latest.sha256').read_text().strip(),'endpoint_hash_receipt')
    check(receipt['status']=='COMPLETED' and checkpoint['logical_tokens']==checkpoint['target_tokens']==target==100_000_000,'complete_target')
    check(checkpoint['step']==receipt['steps']==steps,'complete_steps')
    check(json.loads(json.dumps(checkpoint['config']))==record['config'],'actual_registered_config')
    trainer=snapshot/'scripts/posthack_train.py';trainer_text=trainer.read_text();trainer_hash=sha256(trainer)
    check(trainer_hash==chain['executed_trainer_sha256'],'exact_executed_source_recovered')
    check(set(checkpoint)==payload_keys(trainer_text),'checkpoint_matches_executed_payload_schema')
    if name=='A8-CO4-S1337':
        check(record['source_file_sha256']['scripts/posthack_train.py']==trainer_hash,'registered_first_trainer_hash')
        check(start.get('trainer_sha256') is None and checkpoint.get('trainer_sha256') is None,'original_trainer_omitted_source_hash_as_recovered')
        amendments=[e for e in events if e['event']=='SOURCE_COMPATIBILITY_AMENDMENT']
        check(len(amendments)==1 and 'first active run retains loaded original trainer' in amendments[0]['reason'],'recorded_loaded_first_trainer_amendment')
    else:
        check(start['trainer_sha256']==trainer_hash,'launch_exact_trainer_hash')
        disk_hash=checkpoint['trainer_sha256']
        if disk_hash!=trainer_hash:
            check(name=='A8-DENSE-S1337' and disk_hash==chain['checkpoint_disk_trainer_sha256'],'explained_launch_disk_hash_difference')
            check('"trainer_sha256": sha256(Path(__file__))' in trainer_text,'checkpoint_rehashes_disk_not_loaded_code')
            disk_source=ROOT/chain['checkpoint_disk_trainer_path']
            check(sha256(disk_source)==disk_hash,'exact_changed_disk_source_recovered')
            check(set(checkpoint)!=payload_keys(disk_source.read_text()),'payload_identifies_loaded_launch_version_not_new_disk_version')
            tree=ast.parse(trainer_text)
            check(not any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in ('exec','eval','reload')) or (isinstance(n.func,ast.Attribute) and n.func.attr=='reload')) for n in ast.walk(tree)),'no_dynamic_trainer_reload')
        else:check(True,'launch_and_endpoint_trainer_agree')
    check(start['target_tokens']==target and start['batch_size']==cfg.batch_size and start['context_length']==cfg.context_length,'launch_budget_batch_context')
    check(record.get('variant','baseline')=='baseline' and record.get('sampling_mode','baseline')=='baseline','baseline_no_curriculum_variant')
    if name!='A8-CO4-S1337':
        from latticelm.posthack_variants import build_posthack_model
        build=build_posthack_model
    else:build=lambda config,variant='baseline':build_model(config)
    manifest_path=record.get('manifest_path',original_root_path(trainer_text,'MANIFEST'))
    tokenizer_path=record.get('tokenizer_path',original_root_path(trainer_text,'TOKENIZER'))
    manifest=ROOT/manifest_path;top=json.loads(manifest.read_text())
    check(sha256(manifest)==record['sampling_manifest_sha256']==checkpoint['manifest_sha256'],'data_manifest_hash')
    check(sha256(ROOT/tokenizer_path)==record['tokenizer_sha256']==checkpoint['tokenizer_sha256'],'tokenizer_hash')
    # Reuse only content hashes bound to the current complete stat identity.
    # An uncached/changed shard is hashed, never accepted based on its filename.
    cache=json.loads((RESEARCH/'.archive-hash-cache.json').read_text())
    arrays={source:[] for source in top['mixture_definition']};shards=[]
    for item in top['shards']+top['validation_shards']:
        child=manifest.parent/item['manifest_path'];meta=json.loads(child.read_text());path=manifest.parent/meta['path']
        check(sha256(child)==item['manifest_sha256'],'child_manifest_'+item['manifest_path'])
        stat=path.stat();key=json.dumps((str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
        actual=cache[key] if key in cache else sha256(path)
        check(actual==meta['sha256'] and stat.st_size==meta['token_count']*4,'data_shard_'+meta['path'])
        shards.append({'path':str(path.relative_to(ROOT)),'sha256':actual,'stat_identity':[stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns],'reused_verified_hash':key in cache})
        if item in top['shards']:arrays[meta['source']].append(np.memmap(path,mode='r',dtype='<i4'))
    streams={source:SourceStream(arrays[source],cfg.context_length,cfg.seed+17*i) for i,source in enumerate(arrays)}
    mixture=WeightedMixture(streams,top['mixture_definition'],cfg.batch_size)
    check(tuple(checkpoint.get('sampling_cycle',mixture.cycle))==mixture.cycle and not checkpoint.get('sampling_annealed',False),'saved_sampling_cycle')
    rows=metric_rows(run/'metrics.jsonl');draws=Counter();max_lr_error=0.
    for index,row in enumerate(rows,1):
        check(row['step']==index,'metric_steps_contiguous')
        expected=cfg.learning_rate*wsd_lr_scale(index,steps,cfg.warmup_fraction,cfg.stable_fraction,cfg.decay_fraction)
        max_lr_error=max(max_lr_error,abs(row['lr']-expected))
        check(row['logical_tokens']==min(index*quantum,target) and row['processed_tokens']==index*quantum,'whole_run_token_accounting')
        offset=(index-1)*cfg.batch_size;labels=tuple(mixture.cycle[(offset+i)%len(mixture.cycle)] for i in range(cfg.batch_size))
        check(tuple(row['source_labels'])==labels,'whole_run_source_order');draws.update(labels)
    check(len(rows)==steps and max_lr_error<=1e-15,'one_complete_100m_wsd_no_lr_restart')
    check(checkpoint['mixture']=={'counter':steps*cfg.batch_size,'draws':dict(draws)},'saved_sampler_cursors')
    mixture.load_state_dict(checkpoint['mixture'])
    check(next_batch_hash(mixture)==checkpoint['next_batch_sha256'],'exact_saved_next_batch_replay')
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    model=build(cfg,'baseline');count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(count==record['parameter_count'],'exact_parameter_count')
    optimizer,partition=make_optimizer(model,cfg.optimizer,cfg.learning_rate,cfg.weight_decay,(cfg.adam_beta1,cfg.adam_beta2))
    check(partition==record['optimizer_partition']==checkpoint['optimizer_partition'],'optimizer_partition')
    check(all(not state['state'] for state in optimizer.state_dict()),'fresh_replay_optimizer_states_empty')
    groups=[g for state in checkpoint['optimizers'] for g in state['param_groups']]
    check(all(g['lr']==0 for g in groups),'final_optimizer_lr_zero')
    check(all(not isinstance(value,torch.Tensor) or torch.isfinite(value).all() for value in checkpoint['model'].values()),'finite_endpoint_model')
    # Strict compatibility uses a separate model, preserving the fresh replay.
    fresh_state={k:v.clone() for k,v in model.state_dict().items()};model.load_state_dict(checkpoint['model'],strict=True);model.load_state_dict(fresh_state,strict=True);del fresh_state
    fresh_streams={source:SourceStream(arrays[source],cfg.context_length,cfg.seed+17*i) for i,source in enumerate(arrays)}
    fresh=WeightedMixture(fresh_streams,top['mixture_definition'],cfg.batch_size);base_lrs=[g['lr'] for g in optimizer.param_groups];replay=[]
    for step in range(1,4):
        for group,base in zip(optimizer.param_groups,base_lrs):group['lr']=base*wsd_lr_scale(step,steps,cfg.warmup_fraction,cfg.stable_fraction,cfg.decay_fraction)
        x,y,_=fresh.batch();optimizer.zero_grad();logits,_=model(torch.from_numpy(x.astype(np.int64,copy=False)))
        loss=F.cross_entropy(logits.reshape(-1,cfg.vocab_size),torch.from_numpy(y.astype(np.int64,copy=False)).reshape(-1));loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip);optimizer.step()
        replay.append({'step':step,'loss':float(loss.detach()),'recorded_loss':rows[step-1]['train_loss'],'absolute_error':abs(float(loss.detach())-rows[step-1]['train_loss'])})
    check(max(r['absolute_error'] for r in replay)<=1e-5,'short_fresh_replay_corrobates_initial_trajectory')
    evidence=[run/'result.json',run/'metrics.jsonl',run/'latest.sha256',run/'train.log',old/'RUN_REGISTRY.jsonl',old/'controller_events.jsonl']
    evidence += [ROOT/p for p in chain['evidence_paths']]
    source_paths=[snapshot/p for p in ('scripts/posthack_train.py','scripts/train_final_production.py','src/latticelm/model.py','src/latticelm/config.py','src/latticelm/final_recipe.py','src/latticelm/data_d.py','src/latticelm/co4_reference.py','src/latticelm/memory.py')]
    if (snapshot/'src/latticelm/posthack_variants.py').exists():source_paths.append(snapshot/'src/latticelm/posthack_variants.py')
    result={'schema':'historical-8m-control-provenance-v1','at':utc(),'passed':True,'control':name,'checkpoint_sha256':digest,'executed_trainer_sha256':trainer_hash,'checkpoint_disk_trainer_sha256':checkpoint.get('trainer_sha256'),'executed_snapshot_chain_verified':True,'snapshot_chain':chain,'parameter_count':count,'config':record['config'],'manifest_sha256':checkpoint['manifest_sha256'],'tokenizer_sha256':checkpoint['tokenizer_sha256'],'checks':checks,'source_snapshot_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in source_paths},'evidence_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in evidence},'data_shards':shards,'event_counts':dict(counts),'total_steps':steps,'maximum_lr_error':max_lr_error,'next_batch_sha256':checkpoint['next_batch_sha256'],'fresh_replay':replay,'runtime_torch_version':torch.__version__,'evidence_limit':'No initial production checkpoint. Exact source chain, launch/source amendments, payload schema, contiguous whole-run WSD/source order, optimizer partition, saved next batch and three fresh replay steps corroborate provenance. Replay uses one CPU thread vs historical 12/16, with 1e-5 loss tolerance for FP32 reduction differences; tolerance does not establish source equivalence.'}
    atomic_json(RESEARCH/'provenance_recovery'/f'{name}_AUDIT.json',result)
    return {'control':name,'passed':True,'trainer':trainer_hash,'parameter_count':count,'maximum_lr_error':max_lr_error,'fresh_replay':replay}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment-id',required=True);p.add_argument('--snapshot');p.add_argument('--chain');p.add_argument('--bind-current',action='store_true');args=p.parse_args()
    if args.bind_current:
        result=json.loads((RESEARCH/'provenance_recovery'/f'{args.experiment_id}_AUDIT.json').read_text())
        if not result.get('passed'):raise ValueError('Cannot bind failed historical audit')
        bind_matched_implementation(result);print('Matched frozen implementation bound: '+args.experiment_id);raise SystemExit(0)
    if not args.snapshot or not args.chain:p.error('--snapshot and --chain are required for the historical audit')
    chain=json.loads(Path(args.chain).read_text())
    try:print(json.dumps(audit(args.experiment_id,Path(args.snapshot).resolve(),chain),indent=2))
    except Exception as error:
        atomic_json(RESEARCH/'provenance_recovery'/f'{args.experiment_id}_AUDIT.json',{'at':utc(),'control':args.experiment_id,'passed':False,'reason':str(error),'snapshot_chain':chain})
        raise
