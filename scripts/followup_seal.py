"""Pin production registrations to a commit containing their scientific code.

Preflight registrations remain immutable. Only identity/provenance metadata is
superseded; a certificate is transferred only after scientific equality checks.
"""
import copy
import json
import subprocess
from followup_common import ROOT,RESEARCH,STATE,register,registered,object_hash,sha256,atomic_json,decision,utc


def seal():
    commit=subprocess.check_output(['git','rev-parse','c9ed214'],cwd=ROOT,text=True).strip()
    replacements={'F32-DENSE-S1337-300M':'F32-DENSE-S1337-FRESH300M',
                  'F16-D4-S2026-150M':'F16-D4-S2026-FRESH150M',
                  'F8-MOD-S7331-100M':'F8-MOD-S7331-FRESH100M',
                  'F8-DIFF-S7331-100M':'F8-DIFF-S7331-FRESH100M'}
    state=json.loads(STATE.read_text())
    if state.get('active'):raise RuntimeError('Cannot seal registrations while preflight master is active')
    for old_name,name in replacements.items():
        original=registered(old_name);r=copy.deepcopy(original)
        for rel,digest in r['source_hashes'].items():
            data=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)
            import hashlib
            if hashlib.sha256(data).hexdigest()!=digest or sha256(ROOT/rel)!=digest:raise RuntimeError('Code commit/source hash mismatch: '+rel)
        r.update({'experiment_id':name,'code_commit':commit,'launch_certificate':'certificates/'+name+'.json',
                  'supersedes_preflight_registration':old_name})
        r['parent_control']=replacements.get(r['parent_control'],r['parent_control'])
        register(r);atomic_json(RESEARCH/'configs'/(name+'.json'),r['config'])
        cert=RESEARCH/original['launch_certificate']
        if cert.exists():
            c=json.loads(cert.read_text())
            if c['registration_sha256']!=object_hash(original) or c['source_hashes']!=r['source_hashes']:raise RuntimeError('Preflight identity mismatch')
            for key in ['config','parameter_count','optimizer_partition','token_budget','sampling_manifest_sha256','tokenizer_sha256','source_order','source_hashes','schedule','precision','variant']:
                if original[key]!=r[key]:raise RuntimeError('Scientific change during code pin: '+key)
            c.update({'experiment_id':name,'registration_sha256':object_hash(r),'verified_preflight_registration':old_name,
                      'metadata_only_supersession':True,'code_commit_containing_verified_scientific_sources':commit})
            atomic_json(RESEARCH/r['launch_certificate'],c)
        state['runs'].setdefault(old_name,{})['status']='REJECTED'
        state['runs'][old_name]['reason']='Superseded before production launch to pin the verified source hashes to a commit containing those exact files.'
    atomic_json(STATE,state)
    decision('PRODUCTION_SOURCE_COMMIT_PINNED',code_commit=commit,replacements=replacements)
    subprocess.run(['git','archive','--format=zip','--output='+str(RESEARCH/('source-snapshot-'+commit+'.zip')),commit],cwd=ROOT,check=True)
    print('Sealed production registrations to '+commit)

if __name__=='__main__':seal()
