"""Add-only public archive of a reviewed allowlist, with checked sync plans."""
import hashlib
import json
import os
import shutil
import subprocess
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, sync_bucket
from inventory_scan import scan_file

ROOT=Path('/home/insightlabs38/latticelm')
E=ROOT/'post_hackathon_followup/evacuation'
STAGE=Path('/home/insightlabs38/latticelm-evacuation-staging/bucket')
BUCKET='insightlabs38-pixel/latticelm-vm-archive'
os.environ['HF_TOKEN']=Path('/run/user/1001/.evacuation-hf-token').read_text()
os.environ['HF_XET_HIGH_PERFORMANCE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
api=HfApi()

def atom(path, value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def hash_row(row):
 p=Path(row['local_path'])
 with p.open('rb') as f:row['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
 return row

def save_manifest(rows):
 tmp=E/'BUCKET_ARCHIVE_MANIFEST.jsonl.tmp'
 tmp.write_text(''.join(json.dumps(x)+'\n' for x in rows));os.replace(tmp,E/'BUCKET_ARCHIVE_MANIFEST.jsonl')

def stage_row(row):
 source=Path(row['local_path']);dest=STAGE/row['remote_path'];dest.parent.mkdir(parents=True,exist_ok=True)
 assert source.is_file() and not source.is_symlink()
 assert source.stat().st_size==row['size'] and source.stat().st_mtime_ns==row['mtime_ns'],str(source)
 if dest.exists():
  assert dest.stat().st_ino==source.stat().st_ino or hash_row(dict(row))['sha256']==row.get('sha256'),str(dest)
 else:
  try:os.link(source,dest)
  except PermissionError:
   # Linux protects hardlinks to root-owned historical artifacts. Preserve
   # ownership/content and avoid duplicating their large files.
   subprocess.run(['sudo','-n','ln','--',str(source),str(dest)],check=True)
  assert dest.stat().st_ino==source.stat().st_ino

def main():
 all_rows=json.loads((E/'CANDIDATE_INVENTORY.json').read_text())
 approved=[r for r in all_rows if r['classification'] in {'BUCKET_ARCHIVE','BOTH','MODEL_CURATED'}]
 assert not any(r['classification']=='REVIEW' for r in approved)
 report=json.loads((E/'SECRET_SCAN_REPORT.json').read_text());assert report['status']=='PASS_WITH_EXCLUSIONS' or report['status']=='PASS'
 for row in approved:stage_row(row)
 manifest=[{**r,'destination':BUCKET,'verification_state':'pending'} for r in approved]
 save_manifest(manifest)
 print('Approved broad set:',len(manifest),'files,',sum(r['size'] for r in manifest),'bytes',flush=True)
 p=E/'EVACUATION_STATUS.json';status=json.loads(p.read_text());status['status']='BUCKET_BULK_UPLOADING';atom(p,status)
 groups=[('research/working-tree/post_hackathon_followup','followup'),('research/working-tree/post_hackathon_research','posthack'),('research/frozen-tree','frozen'),('research/home-instructions','home-prompts')]
 artifact_dirs=sorted(set(r['remote_path'].split('/')[3] for r in manifest if r['remote_path'].startswith('research/working-tree/artifacts/') and len(r['remote_path'].split('/'))>4))
 priority=['final_production_run','checkpoints','production_milestone_eval_1500m','final_recipe_experiments','posttraining_salvage_followup','posttraining_salvage','posttraining','lattice_reason']
 for directory in priority+[d for d in artifact_dirs if d not in priority]:
  if directory in artifact_dirs:groups.append(('research/working-tree/artifacts/'+directory,'artifacts-'+directory))
 covered=set()
 for prefix,label in groups:
  rows=[r for r in manifest if r['remote_path'].startswith(prefix+'/')]
  if not rows:continue
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(hash_row,rows))
  save_manifest(manifest)
  source=STAGE/prefix;target='hf://buckets/'+BUCKET+'/'+prefix;plan_path=E/('SYNC_PLAN_'+label+'.jsonl')
  plan=sync_bucket(str(source),target,delete=False,plan=str(plan_path),quiet=True)
  expected={r['remote_path'][len(prefix)+1:] for r in rows}
  assert set(op.path for op in plan.operations)==expected,(label,'plan allowlist differs')
  assert all(op.action in {'upload','skip'} for op in plan.operations),'destructive plan refused'
  print('Uploading',label,plan.summary(),flush=True)
  sync_bucket(apply=str(plan_path),delete=False,quiet=True)
  remote={f.path:f for f in api.list_bucket_tree(BUCKET,prefix=prefix,recursive=True) if hasattr(f,'size')}
  for row in rows:assert remote[row['remote_path']].size==row['size'],row['remote_path'];row['verification_state']='remote_size_verified';covered.add(row['remote_path'])
  dry=sync_bucket(str(source),target,delete=False,dry_run=True,quiet=True)
  assert all(op.action=='skip' for op in dry.operations),(label,'sync verification differs')
  save_manifest(manifest)
  with (E/'EVACUATION_EVENTS.jsonl').open('a') as f:f.write(json.dumps({'event':'PREFIX_VERIFIED','prefix':prefix,'files':len(rows),'bytes':sum(x['size'] for x in rows),'at_epoch':time.time()})+'\n')
  print('Verified',label,len(rows),'files',flush=True)
 remaining=[r for r in manifest if r['remote_path'] not in covered]
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(hash_row,remaining))
 save_manifest(manifest)
 # A final whole-stage add-only pass handles root files/source/config/scripts.
 plan_path=E/'SYNC_PLAN_remaining.jsonl';plan=sync_bucket(str(STAGE),'hf://buckets/'+BUCKET,delete=False,plan=str(plan_path),quiet=True)
 expected={r['remote_path'] for r in manifest};assert set(op.path for op in plan.operations)==expected
 assert all(op.action in {'upload','skip'} for op in plan.operations)
 print('Uploading remaining',plan.summary(),flush=True);sync_bucket(apply=str(plan_path),delete=False,quiet=True)
 remote={f.path:f for f in api.list_bucket_tree(BUCKET,prefix='research',recursive=True) if hasattr(f,'size')}
 for row in manifest:assert remote[row['remote_path']].size==row['size'],row['remote_path'];row['verification_state']='remote_size_verified'
 dry=sync_bucket(str(STAGE),'hf://buckets/'+BUCKET,delete=False,dry_run=True,quiet=True);assert all(op.action=='skip' for op in dry.operations)
 save_manifest(manifest);status=json.loads(p.read_text());status['status']='BUCKET_BULK_VERIFIED';status['bulk_files']=len(manifest);status['bulk_bytes']=sum(r['size'] for r in manifest);atom(p,status)
 print('Broad archive verified:',len(manifest),'files',sum(r['size'] for r in manifest),'bytes',flush=True)

if __name__=='__main__':main()
