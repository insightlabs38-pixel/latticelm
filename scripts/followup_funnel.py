"""Data-driven, bounded follow-up queue. Decisions and settings are frozen first."""
from __future__ import annotations
import copy
import json
import math
import os
import subprocess
from pathlib import Path
import torch
from followup_common import ROOT,RESEARCH,atomic_json,complete_evidence,decision,registered,registrations,register,sha256,utc,verified_checkpoint
from latticelm.final_recipe import wsd_lr_scale
from followup_prepare import make_record
from followup_plan import shallow_control
from followup_analysis import comparison,paired_loss,read

EXTRA_SOURCES=['src/latticelm/full_co4.py','src/latticelm/full_co4_reference.py']


def ensure(name,recipe,budget,role,control='',seed=1337,**extra):
    existing={r['experiment_id']:r for r in registrations()}
    if name in existing:return existing[name]
    cfg=copy.deepcopy(recipe['config']);cfg['seed']=seed
    full=copy.deepcopy(recipe.get('full_co4'))
    r=make_record(name,cfg,budget,800 if full else 10000,control,
        variant=recipe.get('variant','baseline'),role=role,auto_preflight=extra.pop('auto_preflight',True),
        full_co4=full,architecture_specific_tuning=bool(full),
        hypothesis=extra.pop('hypothesis',role),initialization_policy=extra.pop('initialization_policy','fresh seed/data/optimizer/sampler; no warm-start from a shorter WSD run'),**extra)
    if full:
        r['source_hashes'].update({p:sha256(ROOT/p) for p in EXTRA_SOURCES})
        if recipe.get('parameter_count_analytical') and r['parameter_count']!=recipe['parameter_count_analytical']:
            raise RuntimeError('Analytical/measured full-Co4 parameter counts differ')
        r['co4_stages']=full['block_types'].count('modulated');r['total_blocks']=len(full['block_types'])
    if r.get('trainer_script'):
        r['source_hashes'][('scripts/'+r['trainer_script'])]=sha256(ROOT/'scripts'/r['trainer_script'])
    # Every mathematical source must be recoverable from the stated revision.
    import hashlib
    for path,digest in r['source_hashes'].items():
        content=subprocess.check_output(['git','show',r['code_commit']+':'+path],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=digest:raise RuntimeError('Commit does not contain registered science source: '+path)
    register(r);decision('CONDITIONAL_RUN_REGISTERED',experiment_id=name,role=role,parameter_count=r['parameter_count'])
    return r


def done(name):
    if not (RESEARCH/'runs'/name/'evaluation_receipt.json').exists():return False
    return complete_evidence(RESEARCH/'runs'/name,registered(name))


def trained(name):return (RESEARCH/'runs'/name/'train_receipt.json').exists()


def frozen_decision(path,value):
    target=RESEARCH/'decisions'/path
    if target.exists():return read(target)
    atomic_json(target,{'at':utc(),**value});decision('SCIENTIFIC_GATE',path=str(target.relative_to(RESEARCH)),**value)
    return value


def historical_rows():return {r['experiment_id']:r for r in map(json.loads,(ROOT/'post_hackathon_research/RUN_REGISTRY.jsonl').read_text().splitlines())}


def import_control(old_name):
    name='CONTROL-'+old_name;run=RESEARCH/'runs'/name
    if (run/'train_receipt.json').exists():return name
    rows=historical_rows();old=rows[old_name];original=ROOT/'post_hackathon_research/runs'/old_name/'latest.pt'
    checkpoint=torch.load(original,map_location='cpu',weights_only=False)
    digest=sha256(original)
    if checkpoint['logical_tokens']!=old['token_budget'] or json.loads(json.dumps(checkpoint['config']))!=old['config']:raise ValueError('Historical control identity mismatch')
    cfg=old['config'];recipe={'config':cfg,'variant':old.get('variant','baseline')}
    r=ensure(name,recipe,old['token_budget'],'historical control; fresh original run, no new training',seed=cfg['seed'],
        historical_experiment=old_name,historical_checkpoint_sha256=digest,auto_preflight=False,evaluation_mode='full',
        initialization_policy='Imported immutable completed historical checkpoint; no training/continuation allowed',evaluation_seconds=1800)
    if checkpoint['manifest_sha256']!=r['sampling_manifest_sha256'] or checkpoint['tokenizer_sha256']!=r['tokenizer_sha256']:raise ValueError('Historical control data pairing mismatch')
    from followup_train import build_model,evaluate_sources
    from train_final_production import load_arrays
    model=build_model(r);model.load_state_dict(checkpoint['model'],strict=True)
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=r['parameter_count']:raise ValueError('Control parameter mismatch')
    if checkpoint.get('trainer_sha256')!=sha256(ROOT/'scripts/posthack_train.py'):raise ValueError('Historical trainer provenance mismatch')
    _,_,valid=load_arrays(ROOT/r['manifest_path'])
    losses,paired=evaluate_sources(model,valid,cfg['context_length'])
    run.mkdir(parents=True,exist_ok=True);destination=run/'historical-control.pt'
    if not destination.exists():os.link(original,destination)
    item={'file':destination.name,'sha256':digest,'size_bytes':destination.stat().st_size,'step':checkpoint['step'],
          'logical_tokens':checkpoint['logical_tokens'],'processed_tokens':checkpoint['processed_tokens'],'wall_seconds':checkpoint['wall_seconds'],'at':utc()}
    atomic_json(run/'checkpoint_manifest.json',{'schema':'followup-checkpoint-manifest-v1','rolling':[item],'milestones':[item]})
    weights=read(ROOT/r['manifest_path'])['mixture_definition']
    atomic_json(run/'latest_validation.json',{'at':utc(),'step':checkpoint['step'],'logical_tokens':checkpoint['logical_tokens'],
       'source_losses':losses,'paired_losses':paired,'checkpoint_sha256':digest,'frozen_mixture_weighted_loss':sum(weights[s]*losses[s] for s in weights),
       'scope':'Recomputed through common validation pipeline; historical metrics remain unchanged'})
    history=ROOT/'post_hackathon_research/runs'/old_name/'evaluations.jsonl'
    if history.exists():
        from followup_common import atomic_bytes
        atomic_bytes(run/'historical_learning_curve.jsonl',history.read_bytes())
    atomic_json(run/'train_receipt.json',{'status':'TRAINED','experiment_id':name,'logical_tokens':checkpoint['logical_tokens'],
       'processed_tokens':checkpoint['processed_tokens'],'step':checkpoint['step'],'checkpoint_sha256':digest,'wall_seconds':checkpoint['wall_seconds'],
       'imported_historical_control':True,'historical_experiment':old_name,'historical_checkpoint_sha256':digest,'at':utc()})
    return name


def learned(name):
    directory=RESEARCH/'runs'/name;history=directory/'validation_history.jsonl'
    if not history.exists():return False
    rows=[json.loads(line) for line in history.read_text().splitlines()]
    return len(rows)>1 and rows[-1]['frozen_mixture_weighted_loss']<rows[0]['frozen_mixture_weighted_loss']


def late_improvement(name):
    path=RESEARCH/'runs'/name/'validation_history.jsonl'
    if not path.exists():return False
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    late=[r for r in rows if r['logical_tokens']>=50_000_000]
    if len(late)<2:return False
    weights=read(ROOT/registered(name)['manifest_path'])['mixture_definition']
    return paired_loss(late[-1],late[0],weights)['ci95'][1]<0


def extension(parent_name,total=200_000_000):
    name=parent_name+'-EXT200M';parent=registered(parent_name);run=RESEARCH/'runs'/parent_name;item=verified_checkpoint(run)
    if not item or item['logical_tokens']!=100_000_000:raise ValueError('Extension requires verified primary 100M endpoint')
    recipe={'config':parent['config'],'full_co4':parent.get('full_co4'),'variant':parent.get('variant','baseline')}
    start=item['step'];phase_steps=math.ceil((total-item['logical_tokens'])/(parent['config']['batch_size']*parent['config']['context_length']))
    cfg=parent['config'];schedule={**parent['schedule'],'warmup_steps':max(1,round(phase_steps*cfg['warmup_fraction'])),
      'decay_steps':max(1,round(phase_steps*cfg['decay_fraction'])),
      'sample_lrs':{str(s):cfg['learning_rate']*wsd_lr_scale(s,phase_steps,cfg['warmup_fraction'],cfg['stable_fraction'],cfg['decay_fraction']) for s in (1,phase_steps//2,phase_steps)}}
    control=parent.get('parent_control','')
    r=ensure(name,recipe,total,'explicit paired extension phase',control+'-EXT200M' if control else '',seed=cfg['seed'],
      trainer_script='followup_continuation_train.py',continuation_checkpoint=str((run/item['file']).relative_to(ROOT)),
      continuation_checkpoint_sha256=item['sha256'],initial_logical_tokens=item['logical_tokens'],initial_wall_seconds=item['wall_seconds'],
      schedule_start_step=start,schedule_token_budget=total-item['logical_tokens'],
      schedule_total_steps=phase_steps,schedule=schedule,
      schedule_phase='New, explicitly registered 100M WSD phase; preserve all parent state; identical LR restart for candidate and matched controls',
      evaluation_mode='full',evaluation_seconds=7200,initialization_policy='Restore exact parent model/optimizer/RNG/sampler; explicit new schedule phase',
      hypothesis='Paired extension explores late learning; it is not an uninterrupted 200M run')
    return name


def advance(state):
    torch.set_num_threads(1)
    plan=read(RESEARCH/'FOLLOWUP_EXPERIMENT_PLAN.json');rows=historical_rows()
    # Comparator endpoints are evaluated once through the same pipeline. Import
    # executes only while the controller has no active heavy worker.
    import_control('R32-CO4-CONTROL')
    for seed in (1337,2026):
        for architecture in ('DENSE','CO4'):
            old=f'A8-{architecture}-S{seed}'
            if old in rows:import_control(old)
    import_control('R16-S2026-CONTROL')
    dense='F32-DENSE-S1337-FRESH300M';co4='CONTROL-R32-CO4-CONTROL'
    if done(dense) and done(co4):
        frozen_decision('dense-300m-closure.json',{'run':False,'evidence':comparison(dense,co4),
          'interpretation':'Dense-minus-MOD at one matched fresh 300M trajectory; do not extrapolate to 1.9B.'})
    d4='F16-D4-S2026-FRESH150M';control16='CONTROL-R16-S2026-CONTROL'
    if done(d4) and done(control16):
        evidence=comparison(d4,control16)
        frozen_decision('d4-replication.json',{'run':bool(evidence['quality_gain']), 'evidence':evidence,
          'prior_direction':'Small unfavorable aggregate DATA-D change with favorable BPB; inspect source deltas and annealed sampler logs.',
          'bpb_direction_repeated':evidence['paired_bpb']['candidate_minus_control']<0,
          'policy':'No further mixture tournament; coherent LM/BPB evidence and no statistically clear task regression required for promotion.'})
    budgets=plan['stage_budgets'];weights=read(ROOT/'artifacts/data/data_d_v4/canonical-2250m/manifest.json')['mixture_definition']
    # Compatibility controls precede novel families; all compatibility tuning
    # decisions use training/held-out data only, never downstream benchmarks.
    for family,entry in plan['families'].items():
        control=f'FC{family}-SHALLOW-C1-S1337'
        ensure(control,entry['shallow_dense'],budgets['compatibility'],'C1 matched shallow dense compatibility control',evaluation_mode='validation_only',evaluation_seconds=60)
        for index,arm in enumerate(entry['tuning_arms']):
            recipe=copy.deepcopy(entry['main']);recipe['config']['learning_rate']*=arm['global_lr_multiplier'];recipe['full_co4']['initialization_scale']=arm['initialization_scale']
            name=f'FC{family}-C1-{arm["key"]}-S1337'
            if index:
                base=f'FC{family}-C1-BASE-S1337';base_failed=state['runs'].get(base,{}).get('status')=='FAILED_SCIENCE'
                if not done(control) or not (done(base) or base_failed):continue
                needs_tuning=base_failed
                if done(base):
                    paired=paired_loss(read(RESEARCH/'runs'/base/'latest_validation.json'),read(RESEARCH/'runs'/control/'latest_validation.json'),weights)
                    needs_tuning=paired['ci95'][0]>0 or not learned(base)
                gate=frozen_decision(f'{family}-tuning.json',{'family':family,'run':bool(needs_tuning),'reason':'Bounded LR/init tuning only when clearly poorly tuned or unstable; compatibility validation only.'})
                if not gate['run']:continue
            ensure(name,recipe,budgets['compatibility'],'C1 source-grounded compatibility/tuning; not a quality conclusion',control,
                   evaluation_mode='validation_only',evaluation_seconds=60,tuning_arm=arm,family=family)
        base=f'FC{family}-C1-BASE-S1337';gate_path=RESEARCH/'decisions'/f'{family}-tuning.json'
        if not done(control) or not gate_path.exists():continue
        tuning=read(gate_path)['run']
        arms=[f'FC{family}-C1-{arm["key"]}-S1337' for arm in entry['tuning_arms']] if tuning else [base]
        if not all(done(a) or state['runs'].get(a,{}).get('status')=='FAILED_SCIENCE' for a in arms):continue
        viable=[a for a in arms if done(a) and learned(a)]
        if not viable:
            frozen_decision(f'{family}-C1-selection.json',{'family':family,'run':False,'reason':'No learning setting among the at-most-four predefined compatibility arms'});continue
        best=min(viable,key=lambda a:read(RESEARCH/'runs'/a/'latest_validation.json')['frozen_mixture_weighted_loss'])
        selection=frozen_decision(f'{family}-C1-selection.json',{'family':family,'run':True,'selected_compatibility_run':best,'criterion':'held-out loss only; frozen before C2'})
        source=registered(selection['selected_compatibility_run']);recipe={'config':source['config'],'full_co4':source['full_co4']}
        shallow=f'FC{family}-SHALLOW-50M-S1337'
        ensure(shallow,entry['shallow_dense'],budgets['screen'],'C2 matched shallow dense',evaluation_mode='screen',evaluation_seconds=1800)
        for arch in ('DENSE','CO4'):
            old=rows[f'A8-{arch}-S1337'];ensure(f'F8-{arch}-FULLCONTROL-50M-S1337',{'config':old['config']},budgets['screen'],'C2 matched deeper/current-MOD control',evaluation_mode='screen',evaluation_seconds=1800)
        name=f'FC{family}-50M-S1337'
        ensure(name,recipe,budgets['screen'],'C2 distinct source-grounded full-Co4 screen',shallow,evaluation_mode='screen',evaluation_seconds=3600,family=family,selected_setting=selection)
        controls=[shallow,'F8-DENSE-FULLCONTROL-50M-S1337','F8-CO4-FULLCONTROL-50M-S1337']
        if not done(name) or not all(done(c) for c in controls):continue
        evidence=[comparison(name,c) for c in controls]
        promising=not any(e['capability']['any_clear_regression'] for e in evidence) and any(e['quality_gain'] or e['efficiency_signal'] for e in evidence)
        screen=frozen_decision(f'{family}-C2-promotion.json',{'family':family,'run':bool(promising),'evidence':evidence,'reason':'Separate quality/sample/wall/depth axes under the user-confirmed conservative policy'})
        if not screen['run']:continue
        main=f'FC{family}-100M-S1337';shallow100=f'FC{family}-SHALLOW-100M-S1337'
        ensure(shallow100,entry['shallow_dense'],budgets['primary_confirmation'],'Primary matched shallow dense',evaluation_mode='full',evaluation_seconds=7200)
        ensure(main,recipe,budgets['primary_confirmation'],'C3 primary source-grounded confirmation',shallow100,evaluation_mode='full',evaluation_seconds=7200,family=family,selected_setting=selection)
    candidates=[]
    for family in plan['families']:
        main=f'FC{family}-100M-S1337';shallow=f'FC{family}-SHALLOW-100M-S1337'
        controls=[shallow,'CONTROL-A8-DENSE-S1337','CONTROL-A8-CO4-S1337']
        if not done(main) or not all(done(c) for c in controls):continue
        evidence=[comparison(main,c) for c in controls]
        if not any(e['capability']['any_clear_regression'] for e in evidence) and any(e['quality_gain'] or e['efficiency_signal'] for e in evidence):
            candidates.append((family,main,evidence))
    # Wait for every promoted C3 family before selecting the single winner.
    promoted=[f for f in plan['families'] if (RESEARCH/'decisions'/f'{f}-C2-promotion.json').exists() and read(RESEARCH/'decisions'/f'{f}-C2-promotion.json')['run']]
    families_resolved=all((RESEARCH/'decisions'/f'{f}-C2-promotion.json').exists() or
      ((RESEARCH/'decisions'/f'{f}-C1-selection.json').exists() and not read(RESEARCH/'decisions'/f'{f}-C1-selection.json')['run']) for f in plan['families'])
    if candidates and families_resolved and all(done(f'FC{f}-100M-S1337') for f in promoted):
        winner=min(candidates,key=lambda x:read(RESEARCH/'runs'/x[1]/'latest_validation.json')['frozen_mixture_weighted_loss'])
        choice=frozen_decision('full-winner.json',{'family':winner[0],'run':True,'seed1':winner[1],'evidence':winner[2],
            'choice_policy':'Select lowest held-out loss among non-regressing viable C3 families; keep wall/depth/compute axes separate in the report'})
        family=choice['family'];entry=plan['families'][family];first=registered(choice['seed1']);recipe={'config':first['config'],'full_co4':first['full_co4']}
        second=f'FC{family}-100M-S2026';shallow=f'FC{family}-SHALLOW-100M-S2026'
        ensure(shallow,entry['shallow_dense'],100_000_000,'Second-seed matched shallow dense',seed=2026,evaluation_mode='full',evaluation_seconds=7200)
        ensure(second,recipe,100_000_000,'C4 exactly one second primary full-Co4 seed',shallow,seed=2026,evaluation_mode='full',evaluation_seconds=7200,family=family)
        second_controls=[shallow,'CONTROL-A8-DENSE-S2026','CONTROL-A8-CO4-S2026']
        if done(second) and all(done(c) for c in second_controls):
            evidence=[comparison(second,c) for c in second_controls]
            from followup_analysis import two_seed_consistency
            consistency=two_seed_consistency(choice['evidence'],evidence)
            consistent=consistency['passed']
            confirmed=frozen_decision('full-confirmation.json',{'run':bool(consistent),'family':family,'seeds':[1337,2026],
              'evidence_seed1':choice['evidence'],'evidence_seed2':evidence,'consistency':consistency,'reason':'Directional consistency on the same scientific axis/comparator across the two planned seeds; no extra confirmation seeds'})
            if confirmed['run']:
                if late_improvement(choice['seed1']) or late_improvement(second):
                    for seed,parent in [(1337,choice['seed1']),(2026,second)]:
                        for n in [parent,f'FC{family}-SHALLOW-100M-S{seed}',f'CONTROL-A8-DENSE-S{seed}',f'CONTROL-A8-CO4-S{seed}']:extension(n)
                ladder=entry['depth_ladder']['1'];control=f'FC{family}-DEPTH1-DENSE-100M-S1337'
                ensure(control,ladder['shallow_dense'],100_000_000,'Optional depth-1 matched dense',evaluation_mode='full',evaluation_seconds=7200)
                settings=copy.deepcopy(ladder['candidate']);settings['config']['learning_rate']=first['config']['learning_rate'];settings['full_co4']['initialization_scale']=first['full_co4']['initialization_scale']
                ensure(f'FC{family}-DEPTH1-100M-S1337',settings,100_000_000,'Optional depth-1 source-family ladder',control,evaluation_mode='full',evaluation_seconds=7200)
                if late_improvement(choice['seed1']):
                    ladder=entry['depth_ladder']['4'];control=f'FC{family}-DEPTH4-DENSE-100M-S1337'
                    ensure(control,ladder['shallow_dense'],100_000_000,'Conditional depth-4 matched dense',evaluation_mode='full',evaluation_seconds=7200)
                    settings=copy.deepcopy(ladder['candidate']);settings['config']['learning_rate']=first['config']['learning_rate'];settings['full_co4']['initialization_scale']=first['full_co4']['initialization_scale']
                    ensure(f'FC{family}-DEPTH4-100M-S1337',settings,100_000_000,'Conditional justified depth-4 source-family ladder',control,evaluation_mode='full',evaluation_seconds=7200)
                scale=copy.deepcopy(entry['scale_16m']['candidate']);scale['config']['learning_rate']=first['config']['learning_rate'];scale['full_co4']['initialization_scale']=first['full_co4']['initialization_scale']
                control=f'FC{family}-16M-DENSE-150M-S1337';name=f'FC{family}-16M-150M-S1337'
                ensure(control,entry['scale_16m']['shallow_dense'],150_000_000,'Optional first 16M matched shallow dense',evaluation_mode='full',evaluation_seconds=7200)
                ensure(name,scale,150_000_000,'Optional one first 16M source-family run',control,evaluation_mode='full',evaluation_seconds=7200)
                if done(name) and done(control):
                    effect=comparison(name,control)
                    scale_gate=frozen_decision('full-16m-confirmation.json',{'run':bool(effect['quality_gain'] or effect['efficiency_signal']),'evidence':effect})
                    if scale_gate['run']:
                        c=f'FC{family}-16M-DENSE-150M-S2026'
                        ensure(c,entry['scale_16m']['shallow_dense'],150_000_000,'Conditional second 16M matched shallow dense',seed=2026,evaluation_mode='full',evaluation_seconds=7200)
                        ensure(f'FC{family}-16M-150M-S2026',scale,150_000_000,'Conditional one additional 16M source-family seed',c,seed=2026,evaluation_mode='full',evaluation_seconds=7200)
    diff='F8-DIFF-S7331-FRESH100M';mod='F8-MOD-S7331-FRESH100M'
    if done(diff) and done(mod):
        evidence=comparison(diff,mod);replicated=evidence['quality_gain']
        diff_gate=frozen_decision('differential-replication.json',{'run':bool(replicated),'evidence':evidence})
        if diff_gate['run']:
            cfg=copy.deepcopy(rows['R16-S2026-CONTROL']['config']);cfg['seed']=7331
            ensure('F16-MOD-S7331-150M',{'config':cfg},150_000_000,'Conditional Differential 16M matched control',seed=7331,evaluation_mode='full',evaluation_seconds=1800)
            ensure('F16-DIFF-S7331-150M',{'config':cfg,'variant':'differential_co4'},150_000_000,'Conditional Differential 16M replication','F16-MOD-S7331-150M',seed=7331,evaluation_mode='full',evaluation_seconds=1800)
            confirmation=RESEARCH/'decisions/full-confirmation.json'
            if confirmation.exists() and read(confirmation)['run']:
                family=read(confirmation)['family'];parent=registered(f'FC{family}-100M-S1337')
                full=copy.deepcopy(parent['full_co4']);full['differential']=True
                ensure(f'FC{family}-DIFFERENTIAL-50M-S1337',{'config':parent['config'],'full_co4':full},50_000_000,
                       'Conditional source-family + Differential combination; one screen only',f'FC{family}-50M-S1337',evaluation_mode='screen',evaluation_seconds=3600)
