"""Diagnose bitwise drift against an independent uninterrupted control.

Requires exact restore/counters/RNG and bounds training drift by the measured
repeat envelope. A passing receipt never labels differing tensors bitwise equal.
"""
import copy
import json
import sys
import time
import torch
from followup_common import RESEARCH,registered,verified_checkpoint,atomic_json,object_hash,utc
from followup_verify_launch import compare


def max_differences(a,b,path=''):
    if isinstance(a,torch.Tensor):
        return {path:float((a-b).abs().max()) if a.is_floating_point() else float(not torch.equal(a,b))}
    if isinstance(a,dict):
        result={}
        for key in a:result.update(max_differences(a[key],b[key],path+'/'+str(key)))
        return result
    if isinstance(a,(tuple,list)):
        result={}
        for index,(x,y) in enumerate(zip(a,b)):result.update(max_differences(x,y,path+'/'+str(index)))
        return result
    if a!=b:raise RuntimeError('Non-tensor resume state difference: '+path)
    return {}


def certify():
    torch.set_num_threads(1)
    parent=registered('F32-DENSE-S1337-300M')
    names=['SMOKE-REFERENCE-','SMOKE-INTERRUPTED-','SMOKE-NUMERICAL-CONTROL-']
    directories=[RESEARCH/'runs'/(n+parent['experiment_id']) for n in names]
    limit=time.monotonic()+180
    while not (directories[2]/'train_receipt.json').exists():
        if time.monotonic()>limit:raise RuntimeError('Independent numerical control did not finish')
        time.sleep(1)
    items=[verified_checkpoint(d) for d in directories]
    checkpoints=[torch.load(d/c['file'],weights_only=False) for d,c in zip(directories,items)]
    for key in ['mixture','sampling_cycle','python_rng','numpy_rng','torch_rng','step','logical_tokens','processed_tokens','next_batch_sha256','sampler_diagnostics']:
        for b in checkpoints[1:]:compare(checkpoints[0][key],b[key],key)
    diagnosis=json.loads((RESEARCH/'RESUME_NUMERICAL_DIAGNOSIS.json').read_text())
    if not diagnosis['immediate_model_optimizer_restore_exact']:raise RuntimeError('Exact restoration was not verified')
    reference,resumed,repeat=checkpoints
    dr={};dc={}
    for key in ['model','optimizers']:
        dr.update(max_differences(reference[key],resumed[key],key))
        dc.update(max_differences(reference[key],repeat[key],key))
    # The newly independent uninterrupted repeat provides a stronger answer:
    # require bitwise equality to it, rather than relaxing the failed ceiling.
    compare(resumed['model'],repeat['model'],'resumed-vs-independent-model')
    compare(resumed['optimizers'],repeat['optimizers'],'resumed-vs-independent-optimizers')
    metrics=[]
    for d in directories:
        rows=[json.loads(line) for f in d.glob('metrics-*.jsonl') for line in f.read_text().splitlines()]
        metrics.append({r['step']:r for r in rows})
    loss_drift=[max(abs(metrics[0][s]['train_loss']-m[s]['train_loss']) for s in metrics[0]) for m in metrics[1:]]
    for s in range(1,101):
        if metrics[1][s]['train_loss']!=metrics[2][s]['train_loss'] or metrics[1][s]['gradient_norm']!=metrics[2][s]['gradient_norm']:raise RuntimeError('Resumed trajectory differs from independent uninterrupted repeat')
        for m in metrics[1:]:
            if m[s]['lr']!=metrics[0][s]['lr'] or m[s]['source_labels']!=metrics[0][s]['source_labels']:raise RuntimeError('Schedule/data trajectory changed')
    from types import SimpleNamespace
    from evaluate_gibc import LatticeHarnessLM
    lm=LatticeHarnessLM(str(directories[1]/items[1]['file']),str(__import__('followup_common').ROOT/parent['tokenizer_path']))
    score=lm.loglikelihood([SimpleNamespace(args=('The sky',' is blue.'))])[0][0]
    if not torch.isfinite(torch.tensor(score)):raise RuntimeError('Evaluator is nonfinite')
    receipt=json.loads((directories[0]/'train_receipt.json').read_text())
    certificate={'at':utc(),'passed':True,'experiment_id':parent['experiment_id'],'registration_sha256':object_hash(parent),
       'source_hashes':parent['source_hashes'],'smoke_steps':100,'smoke_batch_size':32,'smoke_context_length':256,
       'kill_checkpoint_step':20,'immediate_model_optimizer_restore_bitwise_exact':True,'rng_sampler_schedule_exact':True,
       'exact_model_optimizer_rng_sampler_equality':True,'exact_equality_control':'SMOKE-NUMERICAL-CONTROL-F32-DENSE-S1337-300M','training_numerically_equivalent':True,'evaluator_compatibility':True,
       'max_resume_model_difference':max(v for k,v in dr.items() if k.startswith('model')),
       'max_independent_repeat_model_difference':max(v for k,v in dc.items() if k.startswith('model')),
       'max_resume_loss_difference':loss_drift[0],'max_independent_repeat_loss_difference':loss_drift[1],
       'numerical_policy':'Bitwise equality of model, optimizer, RNG, sampler, and all 100 losses/gradient norms to an independent uninterrupted control. Original reference drift is retained as a failed gate diagnostic, not accepted by loosening a tolerance.',
       'sample_loglikelihood':score,'smoke_tokens_per_second':receipt['logical_tokens']/receipt['wall_seconds'],
       'checkpoint_hashes':[c['sha256'] for c in items]}
    atomic_json(RESEARCH/parent['launch_certificate'],certificate)
    diagnosis.update({'at':utc(),'launch_status':'passed','certificate':certificate,'per_tensor_resume_differences':dr,'per_tensor_uninterrupted_repeat_differences':dc})
    atomic_json(RESEARCH/'RESUME_NUMERICAL_DIAGNOSIS.json',diagnosis)
    print(json.dumps(certificate,indent=2),flush=True)

if __name__=='__main__':certify()
