"""Clean restore and deterministic inference; no training."""
import hashlib
import json
import os
from pathlib import Path
import torch
from tokenizers import Tokenizer
from huggingface_hub import HfApi, hf_hub_download
from latticelm.config import LatticeConfig
from latticelm.model import build_model

ROOT=Path('/home/insightlabs38/latticelm')
E=ROOT/'post_hackathon_followup/evacuation'
RESTORE=Path('/home/insightlabs38/latticelm-evacuation-restore')
REPO='insightlabs38-pixel/LatticeLM-public-archive'
BUCKET='insightlabs38-pixel/latticelm-vm-archive'
os.environ['HF_TOKEN']=Path('/run/user/1001/.evacuation-hf-token').read_text()
api=HfApi()
manifest=[json.loads(x) for x in (E/'MODEL_REPO_MANIFEST.jsonl').read_text().splitlines()]
info=api.model_info(REPO,files_metadata=True,token=False)
assert not info.private
remote={x.rfilename:x for x in info.siblings}
for row in manifest:assert remote[row['remote_path']].size==row['size'],row['remote_path']
chosen=['canonical/checkpoint/final.pt','tokenizer/final_corpus_4k.json','config/final_model_config.json',
 'canonical/manifests/DATA-D-v4.json','reports/final_production_report.md','development/checkpoints/truncated/evacuation_receipt.json']
chosen += [r['remote_path'] for r in manifest if r['remote_path'].startswith('development/checkpoints/truncated/checkpoint-00000521')]
results=[]
for target in chosen:
 row=next(r for r in manifest if r['remote_path']==target)
 p=Path(hf_hub_download(REPO,target,revision=info.sha,local_dir=RESTORE/'model',token=False))
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==row['sha256'],target
 results.append({'destination':'model','path':target,'sha256':h,'restore':'PASS'})
torch.set_num_threads(2)
ck=torch.load(RESTORE/'model/canonical/checkpoint/final.pt',map_location='cpu',weights_only=False)
assert ck['tokens']==1900000000
restored_config=json.loads((RESTORE/'model/config/final_model_config.json').read_text())
assert restored_config==json.loads(json.dumps(ck['config']))
cfg=LatticeConfig(**ck['config']);model=build_model(cfg);model.load_state_dict(ck['model'],strict=True);model.eval()
assert sum(p.numel() for p in model.parameters())==48636168
tok=Tokenizer.from_file(str(RESTORE/'model/tokenizer/final_corpus_4k.json'))
inputs=torch.tensor([tok.encode('LatticeLM research archive.').ids],dtype=torch.long)
with torch.no_grad():
 a,_=model(inputs);b,_=model(inputs)
assert torch.isfinite(a).all() and torch.equal(a,b)
bucket=api.bucket_info(BUCKET,token=False);assert not bucket.private
tree=list(api.list_bucket_tree(BUCKET,prefix='01-canonical-and-selected',recursive=True,token=False))
files={f.path:f for f in tree if hasattr(f,'size')}
for row in manifest:assert files['01-canonical-and-selected/'+row['remote_path']].size==row['size'],row['remote_path']
for target in chosen:
 row=next(r for r in manifest if r['remote_path']==target);p=RESTORE/'bucket'/target;p.parent.mkdir(parents=True,exist_ok=True)
 api.download_bucket_files(BUCKET,[('01-canonical-and-selected/'+target,p)],raise_on_missing_files=True,token=False)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==row['sha256'],target
 results.append({'destination':'bucket','path':target,'sha256':h,'restore':'PASS'})
report={'model_repo':REPO,'model_public':True,'model_revision':info.sha,'model_files':len(remote),'model_bytes':sum(x.size or 0 for x in remote.values()),'tier1_model_listing':'PASS','canonical_model_strict_load':'PASS','canonical_config_checkpoint_pairing':'PASS','canonical_parameter_count':48636168,'canonical_token_count':ck['tokens'],'canonical_tokenizer_load':'PASS','deterministic_inference':'PASS','bucket':BUCKET,'bucket_public':True,'bucket_tier1_files':len(files),'bucket_tier1_bytes':sum(x.size for x in files.values()),'tier1_bucket_listing':'PASS','restored_files':results,'development_checkpoint_restored_and_hash_verified':True,'broader_verification_pending':True}
p=E/'REMOTE_VERIFICATION.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');os.replace(temp,p)
print(json.dumps(report,indent=2))
