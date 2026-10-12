"""Publish a small, explicitly audited critical set using metadata-only staging."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, sync_bucket
from inventory_scan import scan_file

ROOT=Path('/home/insightlabs38/latticelm')
E=ROOT/'post_hackathon_followup/evacuation'
STAGE=Path('/home/insightlabs38/latticelm-evacuation-staging')
REPO='insightlabs38-pixel/LatticeLM-public-archive'
BUCKET='insightlabs38-pixel/latticelm-vm-archive'
os.environ['HF_TOKEN']=Path('/run/user/1001/.evacuation-hf-token').read_text()
os.environ['HF_XET_HIGH_PERFORMANCE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
api=HfApi()

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

mapping={
 'artifacts/final_production_run/final.pt':'canonical/checkpoint/final.pt',
 'artifacts/tokenizers/final_corpus_4k.json':'tokenizer/final_corpus_4k.json',
 'artifacts/tokenizers/final_corpus_4k.report.json':'tokenizer/final_corpus_4k.report.json',
 'artifacts/final_model_config.json':'config/final_model_config.json',
 'artifacts/data/data_d_v4/canonical-2250m/manifest.json':'canonical/manifests/DATA-D-v4.json',
 'artifacts/final_model_facts.json':'canonical/manifests/final_model_facts.json',
 'artifacts/final_model_verification.json':'canonical/manifests/original_verification.json',
 'artifacts/final_production_recipe.json':'config/final_production_recipe.json',
 'artifacts/final_production_report.md':'reports/final_production_report.md',
 'artifacts/final_production_evidence_report.md':'reports/final_production_evidence_report.md',
 'post_hackathon_followup/HISTORICAL_8M_PROVENANCE_RECOVERY.md':'development/provenance/HISTORICAL_8M_PROVENANCE_RECOVERY.md',
 'post_hackathon_followup/FULL_CO4_SOURCE_AUDIT.md':'development/provenance/FULL_CO4_SOURCE_AUDIT.md',
 'post_hackathon_followup/FOLLOWUP_RUN_REGISTRY.jsonl':'development/experiments/FOLLOWUP_RUN_REGISTRY.jsonl',
 'post_hackathon_followup/FOLLOWUP_CONTROLLER_STATE.json':'development/experiments/FOLLOWUP_CONTROLLER_STATE.json',
 'post_hackathon_followup/evacuation/PARTIAL_RUN_RECEIPT.json':'development/checkpoints/truncated/evacuation_receipt.json',
}
run=ROOT/'post_hackathon_followup/runs/F8-CO4-FULLCONTROL-50M-S1337'
for item in json.loads((run/'checkpoint_manifest.json').read_text())['rolling']:
 mapping[str((run/item['file']).relative_to(ROOT))]='development/checkpoints/truncated/'+item['file']
mapping[str((run/'checkpoint_manifest.json').relative_to(ROOT))]='development/checkpoints/truncated/checkpoint_manifest.json'
dense=ROOT/'post_hackathon_followup/runs/F32-DENSE-S1337-FRESH300M'
item=json.loads((dense/'checkpoint_manifest.json').read_text())['rolling'][0]
mapping[str((dense/item['file']).relative_to(ROOT))]='development/checkpoints/selected/F32-DENSE-S1337-FRESH300M/'+item['file']
for name in ['checkpoint_manifest.json','train_receipt.json','REPORT.md']:
 mapping[str((dense/name).relative_to(ROOT))]='development/checkpoints/selected/F32-DENSE-S1337-FRESH300M/'+name
for parent in ['post_hackathon_followup/provenance_recovery','post_hackathon_followup/certificates','src/latticelm','configs']:
 for p in (ROOT/parent).rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.md','.sha256'} and p.stat().st_size<10_000_000:
   mapping[str(p.relative_to(ROOT))]=('development/provenance/' if 'post_hackathon_followup' in p.parts else 'source-snapshots/')+str(p.relative_to(ROOT))
expected={'artifacts/final_production_run/final.pt':'94b91a2d39c2889209cb47b2e04b2c0d307a25610c40c5a124b595020eb762b0','artifacts/tokenizers/final_corpus_4k.json':'294925d61198dfa0863f91193349f6ebead420d06a3fb650a5122d2264a4ebf7','artifacts/data/data_d_v4/canonical-2250m/manifest.json':'9d2811cfb2824b0dd541af89ebfb64403b3ec7143db97d52c4046aafb8eeae7f'}
manifest=[]
for source,target in mapping.items():
 p=ROOT/source
 hits=scan_file(p)
 if hits:raise RuntimeError('Critical file requires secret review: '+source)
 h=digest(p)
 if source in expected:assert h==expected[source],source
 dest=STAGE/'model'/target;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():os.link(p,dest)
 assert dest.stat().st_ino==p.stat().st_ino
 manifest.append({'local_path':str(p),'relative_path':source,'remote_path':target,'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':h,'classification':'BOTH','destination':REPO,'verification_state':'pending'})
(STAGE/'model/development/README.md').write_text('Development artifacts may be incomplete, negative, exploratory, or superseded. The active F8 Co4 control was truncated at 4,268,032 / 50,000,000 tokens due to GCP credit exhaustion. The fresh 32M dense run completed 300M on October 11. Checkpoint manifests preserve original filenames and hashes.\n')
(STAGE/'model/README.md').write_text('---\nlanguage: en\ntags:\n- research\n- causal-lm\nbuckets:\n- insightlabs38-pixel/latticelm-vm-archive\n---\n# LatticeLM public research archive\n\nCanonical model: 48,636,168 parameters, Co4 causal, trained for 1,900,000,000 tokens. Exact native checkpoint, tokenizer, config, provenance and development artifacts are preserved. The final model SHA-256 is `94b91a2d39c2889209cb47b2e04b2c0d307a25610c40c5a124b595020eb762b0`.\n\nDevelopment checkpoints include completed and truncated experiments. See development/README.md and run receipts. Research use; no model license has been invented. License selection remains for author review.\n\nSource: https://github.com/insightlabs38-pixel/latticelm/tree/posthack-followup-2026-10\n\nPublic archive: https://huggingface.co/buckets/insightlabs38-pixel/latticelm-vm-archive\n')
(E/'MODEL_REPO_MANIFEST.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in manifest))
print('Audited Tier 1:',len(manifest),'files,',sum(x['size'] for x in manifest),'bytes; hardlink staging',flush=True)
api.upload_large_folder(repo_id=REPO,repo_type='model',folder_path=STAGE/'model',num_workers=4,print_report=False)
print('Model Tier 1 upload returned successfully',flush=True)
sync_bucket(str(STAGE/'model'),'hf://buckets/'+BUCKET+'/01-canonical-and-selected',delete=False,exclude=['.cache/**'],quiet=True)
print('Bucket Tier 1 upload returned successfully',flush=True)
