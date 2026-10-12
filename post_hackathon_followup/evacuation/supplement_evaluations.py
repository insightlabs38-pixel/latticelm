"""Recover aggregate-only evaluation JSONs initially excluded conservatively."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, sync_bucket
from inventory_scan import scan_file

ROOT=Path('/home/insightlabs38/latticelm')
E=ROOT/'post_hackathon_followup/evacuation'
STAGE=Path('/home/insightlabs38/latticelm-evacuation-staging/supplement')
BUCKET='insightlabs38-pixel/latticelm-vm-archive'
os.environ['HF_TOKEN']=Path('/run/user/1001/.evacuation-hf-token').read_text()
rows=json.loads((E/'CANDIDATE_INVENTORY.json').read_text())

def aggregate(value):
 if isinstance(value,dict):
  for key,child in value.items():
   if key.lower() in {'text','prompt','question','answer','choices','completion','continuation','examples','samples','documents','texts','target','context_text'}:return False
   if not aggregate(child):return False
 elif isinstance(value,list):
  if len(value)>100:return False
  return all(aggregate(x) for x in value)
 elif isinstance(value,str):return len(value)<1024
 return True

manifest=[]
for row in rows:
 p=Path(row['local_path'])
 if row['classification']!='SKIP_LICENSE' or p.suffix!='.json' or row['size']>100_000:continue
 if 'sources' in p.parts or 'data' in p.parts or 'natural_bank' in p.parts:continue
 if not any(x in p.name.lower() for x in ['wikitext','gibc','capability','tasks','evaluation']):continue
 data=json.loads(p.read_text())
 if not aggregate(data) or scan_file(p):continue
 row['classification']='BUCKET_ARCHIVE';row['reason']='inspected aggregate-only evaluation JSON; no raw benchmark text'
 dest=STAGE/row['remote_path'];dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():
  try:os.link(p,dest)
  except PermissionError:
   import subprocess
   subprocess.run(['sudo','-n','ln','--',str(p),str(dest)],check=True)
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 manifest.append({**row,'sha256':h,'destination':BUCKET,'verification_state':'pending'})
(E/'CANDIDATE_INVENTORY.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Aggregate evaluation supplement:',len(manifest),'files',sum(r['size'] for r in manifest),'bytes',flush=True)
plan_path=E/'SYNC_PLAN_evaluation_supplement.jsonl';plan=sync_bucket(str(STAGE),'hf://buckets/'+BUCKET,delete=False,plan=str(plan_path),quiet=True)
assert set(op.path for op in plan.operations)=={r['remote_path'] for r in manifest}
assert all(op.action in {'upload','skip'} for op in plan.operations)
sync_bucket(apply=str(plan_path),delete=False,quiet=True)
remote={f.path:f for f in HfApi().list_bucket_tree(BUCKET,prefix='research',recursive=True) if hasattr(f,'size')}
for row in manifest:assert remote[row['remote_path']].size==row['size'];row['verification_state']='remote_size_verified'
dry=sync_bucket(str(STAGE),'hf://buckets/'+BUCKET,delete=False,dry_run=True,quiet=True);assert all(op.action=='skip' for op in dry.operations)
(E/'SUPPLEMENTAL_ARCHIVE_MANIFEST.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
print('Aggregate evaluations verified',len(manifest),flush=True)
