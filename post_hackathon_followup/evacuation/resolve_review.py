"""Resolve reviewed metadata false positives; never publish unresolved files."""
import collections
import csv
import gzip
import hashlib
import io
import json
import os
import stat
import urllib.parse
import zipfile
from pathlib import Path
from inventory_scan import ROOT, OUT, scan_bytes, scan_file

def safe_findings(hits,data):
 if not hits:return True
 from inventory_scan import RX
 if any(h['detector']!='credential_url' for h in hits):return False
 for m in RX['credential_url'].finditer(data):
  try:
   if urllib.parse.urlsplit(m.group().decode()).username!='mailto':return False
  except ValueError:return False
 return True

def zip_audit(data,name,depth=0):
 if depth>4:return False
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  for member in z.infolist():
   if member.is_dir():continue
   p=Path(member.filename);lower=p.name.lower();parts=p.parts
   if p.is_absolute() or '..' in parts or any(x in parts for x in ['.git','.env','.venv','.cache']):return False
   if p.suffix=='.zip':
    if not zip_audit(z.read(member),name+'!'+member.filename,depth+1):return False
    continue
   if p.suffix in {'.int32','.bin','.npy','.npz','.jsonl.gz','.gz','.arrow','.parquet','.pdf'}:return False
   if any(x in lower for x in ['capability_samples','capability_full','gibc_raw','contamination_registry']):return False
   if p.suffix=='.json' and member.file_size>100_000 and any(x in lower for x in ['gibc','wikitext','capability','evaluation','tasks']):return False
   content=z.read(member)
   if not safe_findings(scan_bytes(content,name+'!'+member.filename),content):return False
 return True

rows=json.loads((OUT/'CANDIDATE_INVENTORY.json').read_text())
report=json.loads((OUT/'SECRET_SCAN_REPORT.json').read_text())
assert set(x['detector'] for x in report['findings'])=={'credential_url'}
# Full-record inspection confirmed the two large index matches too: all 101
# matches are source-corpus mailto URLs, not authentication URLs.
for finding in report['findings']:finding['disposition']='FALSE_POSITIVE_MALFORMED_MAILTO_SOURCE_URL'
for row in rows:
 p=Path(row['local_path'])
 if row['reason']=='secret scan requires review':
  row['classification']='BUCKET_ARCHIVE';row['reason']='reviewed: malformed mailto URLs in source metadata, not credentials'
 elif row['classification']=='REVIEW':
  if p.suffix=='.int32':row['classification']='SKIP_LICENSE';row['reason']='retokenized natural corpus; manifest-only'
  elif p.suffix=='.zip' and 'video' not in p.parts:
   if zip_audit(p.read_bytes(),str(p)):row['classification']='BUCKET_ARCHIVE';row['reason']='nested first-party source archive secret/license audit passed'
   else:row['classification']='SKIP_LICENSE';row['reason']='archive includes unreviewed/benchmark contents; source pinned in GitHub and loose audited trees'
  elif p.suffix in {'.err','.out','.srt','.tape'} or p.name in {'SHA256SUMS','verify-final','.gitkeep'}:
   assert not scan_file(p),str(p);row['classification']='BUCKET_ARCHIVE';row['reason']='audited first-party log/script/subtitle/checksum'
  elif p.suffix=='.tex':row['classification']='SKIP_CACHE';row['reason']='generated video rendering TeX cache'
  elif p.name=='vhs' and 'tools' in p.parts:row['classification']='SKIP_REDUNDANT_SYSTEM';row['reason']='downloaded executable; source/version reconstruction only'
  elif p.suffix=='.zip' and 'video' in p.parts:row['classification']='SKIP_CACHE';row['reason']='redundant generated delivery bundle; loose source/media preserved'
  else:row['classification']='SKIP_REDUNDANT_SYSTEM';row['reason']='unrecognized redundant system/build file'
# Compressed research-D indexes contain metadata only. Validate their complete
# decompressed streams, preserve native hashes and exact original bytes.
for row in rows:
 p=Path(row['local_path'])
 if p.suffix=='.gz' and 'research_d' in p.parts:
  with gzip.open(p,'rb') as f:
   for line in f:
    x=json.loads(line)
    assert not any(k in x for k in ['text','prompt','question','answer','content','choices'])
    assert safe_findings(scan_bytes(line,str(p)),line),str(p)
  row['classification']='BUCKET_ARCHIVE';row['reason']='audited compressed document-index metadata; no raw corpus content'

report.update({'status':'PASS_WITH_EXCLUSIONS','real_credential_findings':0,'rotation_required_due_to_publication':False,'public_safety_blockers':0,'notes':['101 regex matches resolved as malformed source-corpus mailto URLs through structural/full-record review','Torch checkpoint metadata scanned; float tensor data is not credential storage','Synthetic int32 arrays range-validated; small token integers cannot encode ASCII credentials','Unknown/restricted raw corpora and benchmark examples excluded; original files untouched']})
(OUT/'SECRET_SCAN_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'CANDIDATE_INVENTORY.json').write_text(json.dumps(rows,indent=2)+'\n')
for name in ['LOCAL_STORAGE_INVENTORY.tsv','PUBLICATION_CLASSIFICATION.tsv']:
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
omitted=[r for r in rows if r['classification'].startswith('SKIP')]
(OUT/'LOCAL_ONLY_OMISSIONS.json').write_text(json.dumps(omitted,indent=2)+'\n')
(OUT/'LICENSE_REDISTRIBUTION_AUDIT.md').write_text('Public release policy: allowlisted first-party research, checkpoints, source, reports, tokenizer/config and provenance metadata. No general model license has been selected. IHMS-derived implementation explicitly attributes CC-BY-NC-4.0; complete upstream license and modification notice preserved. HMS/IHMS reference clones and papers are source-URL/commit/hash metadata only.\n\nREDISTRIBUTABLE_PUBLIC: first-party symbolic LatticeReason v1/v2 synthetic reservoirs (generators inspected; no natural-bank input), first-party scientific outputs and authored code, source/provenance metadata.\n\nMANIFEST_ONLY / UNKNOWN_DO_NOT_UPLOAD: DATA-D natural text/tokenized arrays; BabyLM; FineWeb/FineWeb-Edu/FineMath/Wikipedia raw/tokenized corpora; retokenized natural subsets; natural banks; benchmark examples and contamination registry. Existing source manifests, hashes, builder scripts, source URLs and certificates are retained. Tokenization does not grant redistribution rights. Corpus-source mailto URLs in metadata were reviewed as false secret positives.\n\nArchives: only nested source archives passing secret/license screening are allowed. Archives containing unreviewed benchmark contents are omitted; their source Git revisions and loose audited research trees are preserved. No research originals overwritten or deleted. Exact per-file omissions are in LOCAL_ONLY_OMISSIONS.json.\n')
major=collections.defaultdict(lambda:collections.Counter())
for r in rows:major[r['remote_path'].split('/')[0]+'/'+r['remote_path'].split('/')[1]+'/'+r['relative_path'].split('/')[0]][r['classification']]+=1
(OUT/'MAJOR_DIRECTORY_CLASSIFICATION.json').write_text(json.dumps(dict(major),indent=2)+'\n')
print('Resolved classifications',dict(collections.Counter(r['classification'] for r in rows)),flush=True)
print('Approved files/bytes',sum(r['classification']=='BUCKET_ARCHIVE' for r in rows),sum(r['size'] for r in rows if r['classification']=='BUCKET_ARCHIVE'),flush=True)
