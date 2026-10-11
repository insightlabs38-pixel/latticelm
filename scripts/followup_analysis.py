"""Paired uncertainty and separate scientific axes; no percentage cutoffs."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from followup_common import ROOT,RESEARCH,registered,utc,atomic_json,sha256


def read(path):return json.loads(Path(path).read_text())


def paired_loss(candidate,control,weights,draws=2000):
    rng=np.random.default_rng(2026);samples=np.zeros(draws);estimate=0.;sources={}
    for source,weight in weights.items():
        a=np.asarray(candidate['paired_losses'][source],dtype=float);b=np.asarray(control['paired_losses'][source],dtype=float)
        if a.shape!=b.shape or not len(a):raise ValueError('Unpaired validation blocks: '+source)
        differences=a-b;estimate+=weight*differences.mean()
        samples+=weight*differences[rng.integers(0,len(a),(draws,len(a)))].mean(1)
        sources[source]=float(differences.mean())
    return {'candidate_minus_control':float(estimate),'ci95':np.quantile(samples,[.025,.975]).tolist(),'source_deltas':sources,
            'unit':'four-sequence held-out block; stratified by source','limitation':'fixed held-out subset and evaluation-item uncertainty, not training-seed uncertainty'}


def paired_bpb(candidate,control,draws=1000):
    for key in ('dataset','config','split','dataset_fingerprint','tokenizer_sha256','stride','context_length'):
        if candidate.get(key)!=control.get(key):raise ValueError('WikiText pairing mismatch: '+key)
    a=candidate['paired_raw_rows'];b=control['paired_raw_rows']
    if len(a)!=len(b):raise ValueError('WikiText row counts differ')
    differences=[];bytecounts=[]
    for x,y in zip(a,b):
        if (x['row_id'],x['utf8_bytes'])!=(y['row_id'],y['utf8_bytes']):raise ValueError('WikiText row identity mismatch')
        if x['utf8_bytes']:
            differences.append(x['negative_log_likelihood_nats']-y['negative_log_likelihood_nats']);bytecounts.append(x['utf8_bytes'])
    differences=np.asarray(differences);bytecounts=np.asarray(bytecounts);rng=np.random.default_rng(2026);results=[]
    for offset in range(0,draws,32):
        idx=rng.integers(0,len(differences),(min(32,draws-offset),len(differences)))
        results.extend((differences[idx].sum(1)/(bytecounts[idx].sum(1)*math.log(2))).tolist())
    return {'candidate_minus_control':candidate['bits_per_byte']-control['bits_per_byte'],
            'ci95':np.quantile(results,[.025,.975]).tolist(),'unit':'paired raw WikiText rows; ratio bootstrap',
            'limitation':'row uncertainty; adjacent paragraphs are not independent documents; no training-seed inference'}


def paired_tasks(candidate,control):
    output={};tasks=('hellaswag','arc_easy','piqa','winogrande')
    test_count=sum(all(metric in row for rows in (candidate[task],control[task]) for row in rows)
                   for task in tasks for metric in ('acc','acc_norm'))
    for task in tasks:
        def index(rows):
            result={}
            for row in rows:
                key=(str(row['doc_id']),row['target_hash'])
                if key in result:raise ValueError('Duplicate capability identity')
                result[key]=row
            return result
        a=index(candidate[task]);b=index(control[task])
        if not a or set(a)!=set(b):raise ValueError('Unpaired capability samples: '+task)
        metrics={}
        for metric in ('acc','acc_norm'):
            if not all(metric in a[k] and metric in b[k] for k in a):continue
            diff=np.asarray([float(a[k][metric])-float(b[k][metric]) for k in a])
            if not np.isin(diff,[-1.,0.,1.]).all():raise ValueError('Capability metric is not item correctness')
            wins=int((diff>0).sum());losses=int((diff<0).sum());n=wins+losses
            p=float(binomtest(min(wins,losses),n,.5).pvalue) if n else 1.
            metrics[metric]={'delta':float(diff.mean()),'n':len(diff),'discordant_wins':wins,'discordant_losses':losses,'paired_exact_p':p,
                             'statistically_clear_regression':bool(diff.mean()<0 and p<.05/max(1,test_count))}
        output[task]=metrics
    return {'per_task':output,'any_clear_regression':any(m['statistically_clear_regression'] for t in output.values() for m in t.values()),
            'tests_corrected':test_count,'policy':'Paired exact test; Bonferroni across all available task/metric tests. Screens are not final evidence.'}


def two_seed_consistency(first,second):
    """Require the same comparison and directional axis; do not pool seed noise."""
    axes=[];matched=[]
    veto=any(e['capability']['any_clear_regression'] for e in first+second)
    for a in first:
        expected=a['control'].replace('S1337','S2026')
        b=next((e for e in second if e['control']==expected),None)
        if b is None:raise ValueError('Missing second-seed comparator: '+expected)
        matched.append(expected)
        if a['clear_lm_regression'] or b['clear_lm_regression']:continue
        if all(e['paired_loss']['candidate_minus_control']<0 and e['paired_bpb']['candidate_minus_control']<0 for e in (a,b)) and (a['quality_gain'] or b['quality_gain']):
            axes.append({'control':expected,'axis':'coherent loss/BPB quality direction'})
        if a['sample_efficiency_strong_threshold_tokens'] and b['sample_efficiency_strong_threshold_tokens']:
            axes.append({'control':expected,'axis':'sample efficiency'})
        if a['quality_consistent_with_tie'] and b['quality_consistent_with_tie']:
            if a['depth_ratio']<1 and b['depth_ratio']<1:axes.append({'control':expected,'axis':'depth efficiency'})
            if a.get('clear_wall_gain') and b.get('clear_wall_gain'):axes.append({'control':expected,'axis':'wall efficiency'})
    return {'passed':bool(axes and not veto),'consistent_axes':axes,'matched_comparators':matched,
            'task_regression_veto':veto,'limitation':'Two seeds support directional replication; they do not precisely estimate the distribution over training seeds.'}


def comparison(candidate_name,control_name):
    candidate=registered(candidate_name);control=registered(control_name)
    for key in ('sampling_manifest_sha256','tokenizer_sha256','source_order','token_budget'):
        if candidate[key]!=control[key]:raise ValueError('Unmatched comparison '+key)
    for key in ('seed','batch_size','context_length','optimizer','learning_rate','warmup_fraction','stable_fraction','decay_fraction'):
        if candidate['config'][key]!=control['config'][key] and not (key=='learning_rate' and candidate.get('architecture_specific_tuning')):raise ValueError('Unmatched config '+key)
    cd=RESEARCH/'runs'/candidate_name;rd=RESEARCH/'runs'/control_name
    cache=RESEARCH/'comparisons'/(candidate_name+'__'+control_name+'.json')
    identity={str(directory/k):sha256(directory/k) for directory in (cd,rd) for k in ('train_receipt.json','evaluation_receipt.json','latest_validation.json','wikitext103.json')}
    if cache.exists():
        existing=read(cache)
        if existing.get('input_receipt_hashes')==identity:return existing
    weights=read(__import__('followup_common').ROOT/candidate['manifest_path'])['mixture_definition']
    loss=paired_loss(read(cd/'latest_validation.json'),read(rd/'latest_validation.json'),weights)
    bpb=paired_bpb(read(cd/'wikitext103.json'),read(rd/'wikitext103.json'))
    tasks=paired_tasks(read(cd/'capability_samples.json'),read(rd/'capability_samples.json'))
    cr=read(cd/'train_receipt.json');rr=read(rd/'train_receipt.json')
    quality=loss['candidate_minus_control']<0 and bpb['candidate_minus_control']<0 and (loss['ci95'][1]<0 or bpb['ci95'][1]<0)
    clear_worse=loss['ci95'][0]>0 or bpb['ci95'][0]>0
    consistent_with_tie=loss['ci95'][0]<=0<=loss['ci95'][1] and bpb['ci95'][0]<=0<=bpb['ci95'][1]
    wall_ratio=cr['wall_seconds']/rr['wall_seconds'];depth_ratio=candidate['config']['n_layers']/control['config']['n_layers']
    lower_token_thresholds=[]
    history=cd/'validation_history.jsonl'
    if history.exists():
        for line in history.read_text().splitlines():
            row=json.loads(line)
            if row['logical_tokens']>=candidate['token_budget']:continue
            result=paired_loss(row,read(rd/'latest_validation.json'),weights)
            if result['ci95'][1]<0:lower_token_thresholds.append(row['logical_tokens'])
    wall=wall_uncertainty(candidate,control)
    result={'at':utc(),'candidate':candidate_name,'control':control_name,'input_receipt_hashes':identity,'paired_loss':loss,'paired_bpb':bpb,'capability':tasks,
            'quality_gain':bool(quality and not tasks['any_clear_regression']),'clear_lm_regression':bool(clear_worse),
            'quality_consistent_with_tie':bool(consistent_with_tie),'tie_note':'Failure to detect a difference is not proof of statistical equivalence.',
            'wall_ratio':wall_ratio,'wall_uncertainty':wall,'clear_wall_gain':bool(wall.get('ratio_ci95') and wall['ratio_ci95'][1]<1),
            'depth_ratio':depth_ratio,'parameter_ratio':candidate['parameter_count']/control['parameter_count'],
            'sample_efficiency_strong_threshold_tokens':lower_token_thresholds,
            'efficiency_signal':bool(not clear_worse and not tasks['any_clear_regression'] and
                                    (lower_token_thresholds or (consistent_with_tie and ((wall.get('ratio_ci95') and wall['ratio_ci95'][1]<1) or depth_ratio<1)))),
            'axes_are_separate':True,'final_claim_requires_two_seeds':True}
    atomic_json(cache,result);return result


def wall_uncertainty(candidate,control,draws=1000):
    def blocks(record):
        run=RESEARCH/'runs'/record['experiment_id']
        paths=sorted(run.glob('metrics-*.jsonl'))
        if record.get('historical_experiment'):paths=[ROOT/'post_hackathon_research/runs'/record['historical_experiment']/'metrics.jsonl']
        by_step={}
        for path in paths:
            for line in path.read_text().splitlines():
                row=json.loads(line);by_step[row['step']]=row
        rows=sorted(by_step.values(),key=lambda r:r['step']);values=[]
        for i in range(0,len(rows)-100,100):
            a,b=rows[i],rows[i+100];duration=b['wall_seconds']-a['wall_seconds'];tokens=b['processed_tokens']-a['processed_tokens']
            if duration>0 and tokens>0:values.append(duration/tokens)
        return np.asarray(values)
    a,b=blocks(candidate),blocks(control)
    if min(len(a),len(b))<4:return {'ratio_ci95':None,'reason':'Insufficient wall-time blocks; point ratio remains descriptive.'}
    rng=np.random.default_rng(2026)
    ratios=a[rng.integers(0,len(a),(draws,len(a)))].mean(1)/b[rng.integers(0,len(b),(draws,len(b)))].mean(1)
    return {'ratio_ci95':np.quantile(ratios,[.025,.975]).tolist(),'blocks':[len(a),len(b)],
      'unit':'100 consecutive optimizer steps; independent block bootstrap across runs',
      'limitation':'Within-run timing variability only; historical host conditions, boot/downtime and discarded preemption work are not estimated by this interval. Endpoint wall includes checkpoint overhead.'}
