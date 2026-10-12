"""Public unauthenticated representative restores from verified archive prefixes."""
import hashlib
import json
from pathlib import Path
import torch
from huggingface_hub import HfApi

E=Path('/home/insightlabs38/latticelm/post_hackathon_followup/evacuation')
RESTORE=Path('/home/insightlabs38/latticelm-evacuation-restore/broad')
BUCKET='insightlabs38-pixel/latticelm-vm-archive'
manifest=[json.loads(x) for x in (E/'BUCKET_ARCHIVE_MANIFEST.jsonl').read_text().splitlines()]
selections=[
 'research/working-tree/post_hackathon_research/runs/R32-CO4-CONTROL/latest.pt',
 'research/working-tree/post_hackathon_research/TRAINING_SUPERVISOR_FINAL_REPORT.md',
 'research/working-tree/post_hackathon_followup/FOLLOWUP_CONTROLLER_STATE.json',
 'research/working-tree/post_hackathon_followup/configs/F8-CO4-FULLCONTROL-50M-S1337.json',
 'research/working-tree/post_hackathon_research/research_d/documents.jsonl.gz',
 'research/home-instructions/LatticeLM_Emergency_VM_Evacuation_Prompt.md',
]
api=HfApi();results=[]
for remote in selections:
 row=next(r for r in manifest if r['remote_path']==remote)
 assert row['sha256'] and row['verification_state']=='remote_size_verified',remote
 p=RESTORE/remote;p.parent.mkdir(parents=True,exist_ok=True)
 api.download_bucket_files(BUCKET,[(remote,p)],raise_on_missing_files=True,token=False)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 assert h==row['sha256'],remote
 if p.suffix=='.json':json.loads(p.read_text())
 if p.suffix=='.pt':
  ck=torch.load(p,map_location='cpu',weights_only=False)
  assert isinstance(ck,dict) and ('model' in ck or 'model_state' in ck or 'model_state_dict' in ck)
 results.append({'path':remote,'sha256':h,'size':p.stat().st_size,'restore':'PASS','unauthenticated_public_download':True})
(E/'BROAD_RESTORE_VERIFICATION.json').write_text(json.dumps({'status':'PASS','restores':results},indent=2)+'\n')
print('Broad representative public restores PASS:',len(results))
