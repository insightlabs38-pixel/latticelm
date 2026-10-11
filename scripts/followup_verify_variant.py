"""Controller-owned preflight: exclusive scientific-batch smoke and exact resume."""
from __future__ import annotations
import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import torch
from followup_common import ROOT,RESEARCH,registered,register,verified_checkpoint,atomic_json,object_hash,alive,utc,reconcile_rolling_manifest
from followup_verify_launch import compare

CHILD=None


def launch(record):
    global CHILD
    run=RESEARCH/'runs'/record['experiment_id'];run.mkdir(parents=True,exist_ok=True)
    reconcile_rolling_manifest(run)
    heartbeat=run/'heartbeat.json'
    if heartbeat.exists():
        identity=json.loads(heartbeat.read_text())
        if alive(identity,record['experiment_id']):
            raise RuntimeError('A preflight trainer is still active; controller must adopt it before repeating preflight')
    with (run/'train.log').open('a') as log:
        CHILD=subprocess.Popen([sys.executable,str(ROOT/'scripts'/record.get('trainer_script','followup_train.py')),'--experiment-id',record['experiment_id'],
             '--stop-epoch',str(time.time()+7200)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return CHILD


def endpoint(record):
    run=RESEARCH/'runs'/record['experiment_id'];path=run/'train_receipt.json'
    if not path.exists():return False
    receipt=json.loads(path.read_text())
    if receipt['status']!='TRAINED':return False
    cp=verified_checkpoint(run)
    return cp['logical_tokens']==record['token_budget'] and cp['sha256']==receipt['checkpoint_sha256']


def wait_child(process,record):
    if process.wait()!=0:
        failure=RESEARCH/'runs'/record['experiment_id']/'failure.json'
        if failure.exists():raise RuntimeError(json.loads(failure.read_text())['message'])
        raise RuntimeError('Preflight trainer exited before valid completion')
    if not endpoint(record):raise RuntimeError('Smoke exited without a valid endpoint receipt')


def verify(name):
    parent=registered(name);step_tokens=parent['config']['batch_size']*parent['config']['context_length']
    c0=None
    if parent.get('full_co4'):
        c0=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_full_co4.py'],cwd=ROOT,
             capture_output=True,text=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        atomic_json(RESEARCH/'runs'/name/'causality_reference_receipt.json',{'at':utc(),'passed':c0.returncode==0,
            'source_hashes':parent['source_hashes'],'stdout':c0.stdout,'stderr':c0.stderr})
        if c0.returncode:raise RuntimeError('Full-Co4 reference/causality gates failed: '+c0.stdout+c0.stderr)
    offset=parent.get('schedule_start_step',0);initial_tokens=parent.get('initial_logical_tokens',0)
    kill_offset=90 if parent.get('anneal_rule') else 20
    kill_step=offset+kill_offset
    records=[]
    for kind in ('REFERENCE','INTERRUPTED','REPEAT'):
        r=copy.deepcopy(parent)
        r.update({'experiment_id':'SMOKE-'+kind+'-'+name,'verification_parent':name,'smoke':True,
                  'token_budget':initial_tokens+100*step_tokens,'schedule_token_budget':parent.get('schedule_token_budget',parent['token_budget']),
                  'checkpoint_tokens':20*step_tokens,'milestone_tokens':[initial_tokens+kill_offset*step_tokens,initial_tokens+100*step_tokens],
                  'hypothesis':'Operational validity check; not a quality result','ready':True})
        r.pop('gate_file',None)
        if kind=='INTERRUPTED':r['test_pause_after_checkpoint_step']=kill_step
        register(r);records.append(r)
    reference,interrupted,repeat=records
    if not endpoint(reference):wait_child(launch(reference),reference)
    run=RESEARCH/'runs'/interrupted['experiment_id'];ready=run/'test_checkpoint_ready.json'
    marker=run/'verified_interruption.json'
    if not marker.exists() and not endpoint(interrupted):
        cp=verified_checkpoint(run)
        if not cp or cp['step']<kill_step:
            process=launch(interrupted);limit=time.monotonic()+7200
            while not ready.exists():
                if process.poll() is not None:raise RuntimeError('Interrupted smoke failed before committed checkpoint')
                if time.monotonic()>limit:raise RuntimeError('Smoke checkpoint rendezvous timed out')
                time.sleep(.2)
            item=json.loads(ready.read_text())
            if item['step']!=kill_step:raise RuntimeError('Smoke interruption position mismatch')
            process.kill();process.wait()
        atomic_json(marker,{'at':utc(),'kill_step':kill_step,'classification':'INTERRUPTED_INFRA'})
    if not endpoint(interrupted):wait_child(launch(interrupted),interrupted)
    dirs=[RESEARCH/'runs'/r['experiment_id'] for r in records[:2]]
    items=[verified_checkpoint(d) for d in dirs]
    checkpoints=[torch.load(d/c['file'],weights_only=False) for d,c in zip(dirs,items)]
    keys=('model','optimizers','mixture','sampling_cycle','python_rng','numpy_rng','torch_rng','step','logical_tokens',
          'processed_tokens','next_batch_sha256','sampler_diagnostics')
    equality_control=reference['experiment_id']
    try:
        for key in keys:compare(checkpoints[0][key],checkpoints[1][key],key)
    except RuntimeError:
        if not endpoint(repeat):wait_child(launch(repeat),repeat)
        d=RESEARCH/'runs'/repeat['experiment_id'];cp=verified_checkpoint(d);c=torch.load(d/cp['file'],weights_only=False)
        for key in keys:compare(c[key],checkpoints[1][key],key)
        equality_control=repeat['experiment_id'];items.append(cp)
    torch.set_num_threads(1)
    from evaluate_gibc import LatticeHarnessLM
    lm=LatticeHarnessLM(str(dirs[1]/items[1]['file']),str(ROOT/parent['tokenizer_path']))
    value=lm.loglikelihood([SimpleNamespace(args=('The sky',' is blue.'))])[0][0]
    if not torch.isfinite(torch.tensor(value)):raise RuntimeError('Evaluator produced a nonfinite likelihood')
    if parent.get('full_co4'):
        lm.model.eval()
        tokens=torch.randint(0,parent['config']['vocab_size'],(1,16))
        with torch.no_grad():
            longer=lm.model(tokens)[0];shorter=lm.model(tokens[:,:7])[0]
        torch.testing.assert_close(shorter,longer[:,:7],atol=2e-5,rtol=2e-5)
    receipt=json.loads((dirs[0]/'train_receipt.json').read_text())
    output={'at':utc(),'passed':True,'experiment_id':name,'registration_sha256':object_hash(parent),'source_hashes':parent['source_hashes'],
      'smoke_steps':100,'smoke_batch_size':parent['config']['batch_size'],'smoke_context_length':parent['config']['context_length'],
      'kill_checkpoint_step':kill_step,'exact_model_optimizer_rng_sampler_equality':True,'exact_equality_control':equality_control,
      'evaluator_compatibility':True,'sample_loglikelihood':value,'checkpoint_hashes':[i['sha256'] for i in items],
      'reference_and_causal_gates_passed':None if c0 is None else c0.returncode==0,
      'smoke_tokens_per_second':(receipt['logical_tokens']-initial_tokens)/max(1.,receipt['wall_seconds']-parent.get('initial_wall_seconds',0)),
      'numerical_policy':'Bitwise exact model/optimizer/RNG/sampler comparison; one independent repeat allowed to diagnose CPU floating-point variation; no relaxed tolerance.'}
    atomic_json(RESEARCH/parent['launch_certificate'],output)
    from followup_retention import cleanup_passed_smokes
    cleanup_passed_smokes()
    print(json.dumps(output),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment-id',required=True);args=parser.parse_args()
    run=RESEARCH/'runs'/args.experiment_id;run.mkdir(parents=True,exist_ok=True)
    (run/'failure.json').unlink(missing_ok=True)
    def interrupted(*_):
        if CHILD and CHILD.poll() is None:
            CHILD.terminate()
            try:CHILD.wait(timeout=180)
            except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
        atomic_json(run/'failure.json',{'classification':'INTERRUPTED_INFRA','at':utc(),'message':'Preflight checker received stop signal'})
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:verify(args.experiment_id)
    except Exception as error:
        if CHILD and CHILD.poll() is None:CHILD.terminate();CHILD.wait(timeout=180)
        classification='FAILED_CODE'
        if CHILD:
            child_name=CHILD.args[CHILD.args.index('--experiment-id')+1]
            child_failure=RESEARCH/'runs'/child_name/'failure.json'
            if child_failure.exists():classification=json.loads(child_failure.read_text())['classification']
        atomic_json(run/'failure.json',{'classification':classification,'at':utc(),'message':str(error),'error_type':type(error).__name__})
        raise

if __name__=='__main__':main()
