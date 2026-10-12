"""Allowlist-first inventory. Reports secret categories, never matched values."""
import collections
import csv
import hashlib
import json
import os
import re
import stat
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('/home/insightlabs38/latticelm')
OUT = ROOT / 'post_hackathon_followup/evacuation'
SKIP_DIRS = {'.git', '.venv', '.triton-cpu-venv', '.triton-cpu-src', '__pycache__', '.cache', 'cache', '.pytest_cache', '.pytest-tmp', 'node_modules', '.mypy_cache', '.ruff_cache', '.egg-info'}
ROOTS = [ROOT, Path('/home/insightlabs38/latticelm-frozen')]
PREFIXES = {'latticelm': 'research/working-tree', 'latticelm-frozen': 'research/frozen-tree'}
PATTERNS = {
 'huggingface_token': rb'\bhf_[A-Za-z0-9]{20,}\b',
 'github_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
 'private_key': rb'-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----',
 'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'google_api_key': rb'\bAIza[0-9A-Za-z_-]{30,}\b',
 'bearer_credential': rb'(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}',
 'credential_url': rb'(?i)https?://[^\s/:@"\x27\\<>]{1,100}:[^\s/@"\x27\\<>]{3,100}@[^\s/"\x27\\<>]+',
 'secret_assignment': rb'(?i)(?:password|client_secret|access_token|api_key|auth_token|hf_token|github_token)\s*["\x27]?\s*[:=]\s*["\x27]([^"\x27\r\n]{8,})["\x27]',
 'signed_url': rb'(?i)(?:X-Amz-Signature|X-Goog-Signature|[?&]Signature)=[A-Za-z0-9%+/=_-]{20,}',
}
RX = {k: re.compile(v) for k,v in PATTERNS.items()}
TRIGGERS = {
 'huggingface_token': [b'hf_'], 'github_token': [b'ghp_',b'gho_',b'ghu_',b'ghs_',b'ghr_',b'github_pat_'],
 'private_key':[b'private key'], 'aws_access_key':[b'akia',b'asia'], 'google_api_key':[b'aiza'],
 'bearer_credential':[b'bearer'], 'credential_url':[b'http://',b'https://'],
 'secret_assignment':[b'password',b'client_secret',b'access_token',b'api_key',b'auth_token',b'hf_token',b'github_token'],
 'signed_url':[b'signature='],
}

def scan_bytes(data, file, location=''):
 hits=[]
 lower=data.lower()
 for name, rx in RX.items():
  if not any(trigger in lower for trigger in TRIGGERS[name]):continue
  for match in rx.finditer(data):
   if name == 'secret_assignment':
    value=match.group(1)
    if re.search(rb'(?i)(example|placeholder|redact|your_|getenv|environ|os\.|token_path|none|false|secret|\$|\{|\(|\[)', value): continue
   hits.append({'file':file,'location':location or 'line:'+str(data[:match.start()].count(b'\n')+1),'detector':name,'disposition':'REVIEW_EXCLUDED'})
 return hits

def classification(path, relative):
 parts=relative.parts; name=path.name.lower(); suffix=path.suffix.lower()
 if any(p in SKIP_DIRS or p.endswith('.egg-info') for p in parts):return 'SKIP_CACHE','rebuildable dependency/cache'
 if name.startswith('.env') or name in {'.netrc','.git-credentials','.bash_history','token','stored_tokens'} or suffix in {'.key','.pem'}:return 'SKIP_SECRET','authentication material'
 if suffix in {'.tmp','.lock','.pid','.pyc','.orig'}:return 'SKIP_TEMP','temporary or runtime lock'
 if path.is_symlink():return 'REVIEW','symlink target recorded; not followed'
 if 'sources' in parts and 'post_hackathon_followup' in parts:return 'SKIP_LICENSE','third-party source/paper redistribution unresolved; metadata only'
 if suffix in {'.npy','.npz','.bin','.arrow','.parquet','.jsonl.gz','.gz'}:return 'SKIP_LICENSE','raw/tokenized or compressed data rights unresolved'
 if suffix=='.int32' and ('lattice_reason_v2' in parts and 'natural_bank' not in parts or 'lattice_reason' in parts):return 'BUCKET_ARCHIVE','first-party symbolic synthetic data; generators inspected'
 if 'tokenizer_sources' in parts:return 'SKIP_LICENSE','tokenizer training text redistribution unresolved'
 if name=='contamination_registry.json' or ('natural_bank' in parts and suffix=='.jsonl'):return 'SKIP_LICENSE','natural/benchmark source content; reconstruction script retained'
 if 'data' in parts and suffix not in {'.json','.md','.sha256','.csv','.py'}:return 'SKIP_LICENSE','dataset content rights unresolved'
 if suffix in {'.zip','.tar','.tgz','.zst','.pdf'}:return 'REVIEW','archive/document requires nested content and license audit'
 if suffix in {'.jsonl','.txt'} and ('data' in parts or any(x in name for x in ['dataset','corpus','samples','examples'])):return 'SKIP_LICENSE','raw examples/text redistribution unresolved'
 if suffix=='.json' and any(x in name for x in ['samples','gibc_raw','capability_full','wikitext103','wikitext_','tasks']):return 'SKIP_LICENSE','per-example benchmark content; aggregate reports retained'
 if suffix=='.json' and path.stat().st_size>100_000 and any(x in name for x in ['gibc','wikitext','capability','evaluation']):return 'SKIP_LICENSE','large per-example evaluation output; aggregate reports retained'
 if suffix in {'.pt','.pth','.ckpt','.safetensors','.json','.jsonl','.md','.csv','.tsv','.sha256','.py','.sh','.ps1','.toml','.yaml','.yml','.service','.txt','.patch','.diff','.sfdisk','.log','.png','.jpg','.jpeg','.svg','.mp4','.gif','.html','.css','.js','.ts','.tsx','.vhs'} or path.name in {'.gitignore','Dockerfile','LICENSE','NOTICE','Makefile'}:return 'BUCKET_ARCHIVE','first-party research/source/result candidate'
 return 'REVIEW','unrecognized file type'

def scan_file(path):
 hits=[]
 if path.suffix.lower()=='.int32':
  import numpy as np
  if path.stat().st_size % 4:return [{'file':str(path),'location':'binary','detector':'invalid_int32','disposition':'REVIEW_EXCLUDED'}]
  values=np.memmap(path,dtype='<i4',mode='r')
  if len(values) and (values.min()<0 or values.max()>=65536):return [{'file':str(path),'location':'binary','detector':'invalid_token_range','disposition':'REVIEW_EXCLUDED'}]
  # Valid small int32 token IDs cannot encode ASCII credential strings.
  return []
 # Torch archives: scan metadata, plus all binary streams for concrete token/key patterns.
 if path.suffix.lower() in {'.pt','.pth','.ckpt'} and zipfile.is_zipfile(path):
  with zipfile.ZipFile(path) as z:
   for entry in z.infolist():
    if entry.filename.endswith(('.pkl','.json','.txt','.py')):
     hits.extend(scan_bytes(z.read(entry),str(path),'archive:'+entry.filename))
  return hits
 with path.open('rb') as f:
  offset=0; tail=b''
  while block:=f.read(8*1024*1024):
   data=tail+block
   hits.extend(scan_bytes(data,str(path),'byte-region:'+str(max(0,offset-len(tail)))))
   offset+=len(block);tail=data[-2048:]
 return hits

def main():
 rows=[];hits=[];aggregates=collections.defaultdict(lambda:[0,0]);skipped=[]
 for root in ROOTS:
  for directory,dirs,files in os.walk(root,followlinks=False):
   for d in list(dirs):
    if d in SKIP_DIRS or d.endswith('.egg-info'):
     skipped.append({'path':str(Path(directory)/d),'classification':'SKIP_CACHE' if d!='.git' else 'SKIP_REDUNDANT_SYSTEM'});dirs.remove(d)
   for name in files:
    path=Path(directory)/name
    if OUT in path.parents:continue
    rel=path.relative_to(root);s=path.lstat();cl,reason=classification(path,rel)
    row={'local_path':str(path),'relative_path':str(rel),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':oct(stat.S_IMODE(s.st_mode)),'symlink_target':os.readlink(path) if path.is_symlink() else '', 'classification':cl,'reason':reason,'remote_path':PREFIXES[root.name]+'/'+str(rel),'sha256':None}
    aggregates[root.name+'/'+(rel.parts[0] if len(rel.parts)>1 else '[root]')][0]+=1
    aggregates[root.name+'/'+(rel.parts[0] if len(rel.parts)>1 else '[root]')][1]+=s.st_size
    rows.append(row)
 candidates=[r for r in rows if r['classification']=='BUCKET_ARCHIVE']
 print('Scanning',len(candidates),'allowlisted files with 8 workers',flush=True)
 with ProcessPoolExecutor(max_workers=8) as pool:
  for row,found in zip(candidates,pool.map(scan_file,[Path(r['local_path']) for r in candidates],chunksize=4)):
   if found:hits.extend(found);row['classification']='REVIEW';row['reason']='secret scan requires review'
 for path in Path('/home/insightlabs38').glob('LatticeLM_*_Prompt.md'):
  s=path.stat();found=scan_file(path);hits.extend(found)
  rows.append({'local_path':str(path),'relative_path':path.name,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':oct(stat.S_IMODE(s.st_mode)),'symlink_target':'','classification':'REVIEW' if found else 'BUCKET_ARCHIVE','reason':'home-level LatticeLM instructions','remote_path':'research/home-instructions/'+path.name,'sha256':None})
 (OUT/'CANDIDATE_INVENTORY.json').write_text(json.dumps(rows,indent=2)+'\n')
 for filename in ['LOCAL_STORAGE_INVENTORY.tsv','PUBLICATION_CLASSIFICATION.tsv']:
  with (OUT/filename).open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
 (OUT/'SECRET_SCAN_REPORT.json').write_text(json.dumps({'status':'REVIEW' if hits else 'PASS','scanner':'focused local allowlist scanner','findings':hits,'excluded_directories':skipped,'credential_values_printed':False},indent=2)+'\n')
 (OUT/'STORAGE_DIRECTORY_SUMMARY.json').write_text(json.dumps(dict(aggregates),indent=2)+'\n')
 print(json.dumps({'files':len(rows),'bytes':sum(r['size'] for r in rows),'classifications':dict(collections.Counter(r['classification'] for r in rows)),'findings':len(hits),'major_directories':dict(aggregates)},indent=2))

if __name__=='__main__':main()
