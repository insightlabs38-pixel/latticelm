"""User-approved disposal of reproducible passed smoke weights; evidence stays."""
import json
from followup_common import RESEARCH,atomic_json,alive,decision,sha256,utc


def cleanup_passed_smokes():
    policy=RESEARCH/'SMOKE_RETENTION_POLICY.json'
    if not policy.exists() or not json.loads(policy.read_text()).get('delete_passed_weights'):return 0
    hashes=set()
    for path in (RESEARCH/'certificates').glob('*.json'):
        certificate=json.loads(path.read_text())
        if certificate.get('passed') and certificate.get('exact_model_optimizer_rng_sampler_equality'):
            hashes.update(certificate.get('checkpoint_hashes',[]))
    reclaimed=0
    for run in (RESEARCH/'runs').glob('SMOKE-*'):
        receipt=run/'train_receipt.json';manifest=run/'checkpoint_manifest.json'
        if not receipt.exists() or not manifest.exists() or (run/'smoke_weights_disposed.json').exists():continue
        r=json.loads(receipt.read_text())
        if r.get('status')!='TRAINED' or r.get('checkpoint_sha256') not in hashes:continue
        heartbeat=run/'heartbeat.json'
        if heartbeat.exists() and alive(json.loads(heartbeat.read_text()),run.name):continue
        items={i['file']:i for i in json.loads(manifest.read_text())['rolling']+json.loads(manifest.read_text())['milestones']}
        evidence=[]
        for name,item in items.items():
            path=run/name
            if not path.exists():continue
            digest=sha256(path)
            if digest!=item['sha256']:raise RuntimeError('Smoke disposal hash mismatch: '+str(path))
            evidence.append({'file':name,'sha256':digest,'size_bytes':path.stat().st_size,'step':item['step']})
        atomic_json(run/'smoke_weights_disposed.json',{'at':utc(),'user_approved':True,'completed_endpoint_sha256':r['checkpoint_sha256'],
           'files':evidence,'comparison_proof':'Passed certificate records exact equality of model, optimizer, RNG and sampler; retain all receipts, hashes, registration, logs and metric histories.'})
        for item in evidence:
            (run/item['file']).unlink();reclaimed+=item['size_bytes']
    if reclaimed:decision('PASSED_SMOKE_WEIGHTS_DISPOSED',bytes_reclaimed=reclaimed)
    return reclaimed

if __name__=='__main__':print(cleanup_passed_smokes())
