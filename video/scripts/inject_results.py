#!/usr/bin/env python3
"""Import final salvage evidence without fabricating missing metrics."""
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
KEYS=['final_model_name','final_checkpoint_sha','final_lineage','final_recovery_tokens','final_interpolation_alpha','data_d','wikitext_ppl','wikitext_bpb','hellaswag','arc_easy','piqa','winogrande','reasoning_accuracy','reasoning_margin','reasoning_scope']

def load(p): return json.loads(Path(p).read_text())
def first(d,*keys):
    for k in keys:
        if isinstance(d,dict) and d.get(k) is not None:return d[k]
    return None

def normalize(selection,evaluations):
    sid=first(selection,'selected_candidate_id','candidate_id','selected_candidate','selected','id','name')
    if isinstance(sid,dict): sid=first(sid,'candidate_id','id','name')
    if not sid: raise ValueError('Selection must identify a candidate')
    if isinstance(evaluations,list):
        matches=[e for e in evaluations if first(e,'candidate_id','id','name')==sid]
        e=matches[0] if matches else None
    elif isinstance(evaluations,dict):
        e=evaluations.get(sid) or evaluations.get('candidates',{}).get(sid) or evaluations.get('selected') or evaluations.get('metrics')
    else:e=None
    if not isinstance(e,dict):raise ValueError(f'No evaluation for selected candidate {sid}')
    data=e.get('metrics',e)
    selected_sha=first(selection,'checkpoint_sha256','checkpoint_sha')
    evaluated_sha=first(data,'checkpoint_sha256','checkpoint_sha')
    if selected_sha and evaluated_sha and selected_sha!=evaluated_sha:
        raise ValueError('Selected checkpoint hash differs from evaluated checkpoint hash')
    proxy=data.get('proxy_1024',{})
    official=data.get('official',data.get('full',data))
    out={k:None for k in KEYS}
    out.update(final_model_name=first(selection,'final_model_name','model_name') or sid,
               final_checkpoint_sha=first(selection,'checkpoint_sha256','checkpoint_sha') or first(data,'checkpoint_sha256','checkpoint_sha'),
               final_lineage=first(selection,'lineage') or sid,
               final_recovery_tokens=first(selection,'recovery_tokens') or first(official,'recovery_tokens'),
               final_interpolation_alpha=first(selection,'alpha','interpolation_alpha') or first(official,'interpolation_alpha'),
               data_d=first(official.get('data_d',{}),'loss') if isinstance(official.get('data_d'),dict) else first(official,'data_d','data_d_validation','validation_loss'),
               reasoning_accuracy=first(proxy,'accuracy') or first(official,'reasoning_accuracy','v2_ranking_accuracy'),
               reasoning_margin=first(proxy,'mean_margin') or first(official,'reasoning_margin','v2_mean_margin'),
               reasoning_scope='proxy_1024' if proxy else 'final_evaluation')
    for k in ['wikitext_ppl','wikitext_bpb','hellaswag','arc_easy','piqa','winogrande']:
        out[k]=first(official,k) or first(official.get('gibc_raw',{}),k)
    if out['reasoning_accuracy'] is None and isinstance(official.get('symbolic'),dict):
        out['reasoning_accuracy']=first(official['symbolic'].get('raw',{}),'accuracy')
    # Proxy measurements never silently become official final metrics.
    required=['final_checkpoint_sha','data_d','reasoning_accuracy','reasoning_margin','wikitext_ppl','wikitext_bpb','hellaswag','arc_easy','piqa','winogrande']
    missing=[k for k in required if out[k] is None]
    if missing:raise ValueError('Final evaluation lacks '+', '.join(missing))
    for k in ['arc_easy','piqa','hellaswag','winogrande','reasoning_accuracy']:
        v=out[k]
        if v is not None and not 0<=v<=1:raise ValueError(f'{k} must be a fraction in [0,1]')
    return out,sid

def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--evaluations',required=True);p.add_argument('--output',default=str(ROOT/'config/metrics.json'));a=p.parse_args()
    selection,evaluations=load(a.selection),load(a.evaluations)
    result,sid=normalize(selection,evaluations)
    payload={'status':'final','final':result,'source':{'selection':str(Path(a.selection).resolve()),'evaluations':str(Path(a.evaluations).resolve())}}
    Path(a.output).write_text(json.dumps(payload,indent=2)+'\n')
    # Include only measured proxy recovery points. Never interpolate missing values.
    candidates=REPO/'artifacts/posttraining_salvage/candidates.json'
    if candidates.exists() and Path(a.output).resolve()==(ROOT/'config/metrics.json').resolve():
        cs=load(candidates);points=[]
        for name,c in cs.items():
            if not name.startswith('joint-recovery-'):continue
            pp=REPO/f'artifacts/posttraining_salvage/candidates/{name}/proxy-256.json'
            if not pp.exists():continue
            pr=load(pp);acc=first(pr,'v2_ranking_accuracy')
            loss=first(pr,'data_d_validation')
            if acc is not None and loss is not None: points.append({'tokens':int(c.get('recovery_tokens',name.rsplit('-',1)[-1])),'reasoning_accuracy':round(acc*100,6),'data_d':loss,'kind':'proxy_256'})
        interpolations=[]
        for name,c in cs.items():
            if c.get('method')!='interpolation':continue
            pr=c.get('proxy_256')
            pp=REPO/f'artifacts/posttraining_salvage/candidates/{name}/proxy-256.json'
            if pr is None and pp.exists():pr=load(pp)
            if not isinstance(pr,dict):continue
            accuracy=first(pr,'v2_ranking_accuracy');loss=first(pr,'data_d_validation')
            if accuracy is not None and loss is not None and c.get('alpha') is not None:
                interpolations.append({'alpha':c['alpha'],'reasoning_accuracy':accuracy*100,'data_d':loss,'kind':'proxy_256'})
        data_path=ROOT/'data/recovery.json';d=load(data_path)
        if points:d['measured']=sorted(points,key=lambda x:x['tokens'])
        d['interpolation']=sorted(interpolations,key=lambda x:x['alpha'])
        d['show_interpolation']=bool(interpolations and result['final_interpolation_alpha'] is not None)
        data_path.write_text(json.dumps(d,indent=2)+'\n')
    print(f'Injected final result: {sid}')
if __name__=='__main__':main()
