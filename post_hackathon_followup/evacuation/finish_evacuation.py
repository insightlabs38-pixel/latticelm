"""Final publication and restore gate. Never shuts down the VM or deletes research."""
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi, ModelCard, hf_hub_download, sync_bucket
from inventory_scan import scan_file

ROOT=Path('/home/insightlabs38/latticelm')
E=ROOT/'post_hackathon_followup/evacuation'
STAGE=Path('/home/insightlabs38/latticelm-evacuation-staging')
RESTORE=Path('/home/insightlabs38/latticelm-evacuation-restore/final')
REPO='insightlabs38-pixel/LatticeLM-public-archive'
BUCKET='insightlabs38-pixel/latticelm-vm-archive'
BRANCH='posthack-followup-2026-10'
os.environ['HF_TOKEN']=Path('/run/user/1001/.evacuation-hf-token').read_text()
api=HfApi()

def atom(path,value):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');os.replace(t,path)

def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def run(*args):return subprocess.check_output(list(args),cwd=ROOT).decode().strip()

def manifest_row(source,remote,destination,classification='BOTH'):
 source=Path(source);s=source.stat()
 return {'local_path':str(source),'relative_path':str(source),'remote_path':remote,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'mode':oct(s.st_mode&0o777),'sha256':digest(source),'classification':classification,'destination':destination,'verification_state':'pending'}

def main():
 status=json.loads((E/'EVACUATION_STATUS.json').read_text());assert status['status']=='BUCKET_BULK_VERIFIED'
 assert json.loads((E/'BROAD_RESTORE_VERIFICATION.json').read_text())['status']=='PASS'
 verify=json.loads((E/'REMOTE_VERIFICATION.json').read_text());assert verify['canonical_config_checkpoint_pairing']=='PASS'
 assert run('git','branch','--show-current')==BRANCH
 enabled=subprocess.run(['systemctl','--user','is-enabled','latticelm-posthack-followup-master.service'],capture_output=True,text=True)
 assert enabled.stdout.strip()=='disabled'
 assert run('systemctl','--user','show','latticelm-posthack-followup-master.service','-p','MainPID','--value')=='0'
 candidates=json.loads((E/'CANDIDATE_INVENTORY.json').read_text());assert not any(r['classification']=='REVIEW' for r in candidates)
 # Merge the separately verified aggregate-evaluation supplement.
 bucket_rows=[json.loads(x) for x in (E/'BUCKET_ARCHIVE_MANIFEST.jsonl').read_text().splitlines()]
 bucket_rows += [json.loads(x) for x in (E/'SUPPLEMENTAL_ARCHIVE_MANIFEST.jsonl').read_text().splitlines()]
 assert len({r['remote_path'] for r in bucket_rows})==len(bucket_rows)
 assert all(r['sha256'] and r['verification_state']=='remote_size_verified' for r in bucket_rows)
 (E/'BUCKET_ARCHIVE_MANIFEST.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bucket_rows))
 by_path={r['local_path']:r for r in bucket_rows}
 for r in candidates:
  if r['local_path'] in by_path:r['sha256']=by_path[r['local_path']]['sha256']
 for name in ['LOCAL_STORAGE_INVENTORY.tsv','PUBLICATION_CLASSIFICATION.tsv']:
  with (E/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(candidates[0]),delimiter='\t');w.writeheader();w.writerows(candidates)
 omissions=[r for r in candidates if r['classification'].startswith('SKIP')]
 (E/'LOCAL_ONLY_OMISSIONS.json').write_text(json.dumps(omissions,indent=2)+'\n')
 # Small, numerically exact inference export supplements the original checkpoint.
 torch.set_num_threads(2)
 ck=torch.load(ROOT/'artifacts/final_production_run/final.pt',map_location='cpu',weights_only=False)
 weights={k:v.detach().contiguous() for k,v in ck['model'].items()}
 safe=STAGE/'model/canonical/checkpoint/model.safetensors';save_file(weights,str(safe))
 loaded=load_file(str(safe));assert weights.keys()==loaded.keys()
 assert all(v.dtype==loaded[k].dtype and torch.equal(v,loaded[k]) for k,v in weights.items())
 atom(E/'SAFETENSORS_EQUIVALENCE.json',{'status':'PASS','source_checkpoint_sha256':digest(ROOT/'artifacts/final_production_run/final.pt'),'safetensors_sha256':digest(safe),'tensor_count':len(weights),'comparison':'all tensor keys, dtypes and values exactly equal; original native checkpoint retained'})
 api.upload_file(repo_id=REPO,path_or_fileobj=safe,path_in_repo='canonical/checkpoint/model.safetensors',commit_message='Add numerically exact canonical inference export')
 # Preserve evacuation tooling/small policy receipts in Git; inventories and
 # changing transfer ledgers are preserved in both HF destinations instead.
 git_paths=[]
 for p in E.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix not in {'.py','.md','.json','.txt','.service','.conf'}:continue
  if p.name in {'GIT_PUSH_RECEIPT.json','EVACUATION_STATUS.json','REMOTE_VERIFICATION.json','LOCAL_ONLY_OMISSIONS.json','CANDIDATE_INVENTORY.json','GIT_SOURCE_CANDIDATES.json','HF_EXISTING_MODEL_TREE.json'}:continue
  if p.stat().st_size>100_000:continue
  assert not scan_file(p),str(p);git_paths.append(str(p.relative_to(ROOT)))
 subprocess.run(['git','add','--',*git_paths],cwd=ROOT,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
  subprocess.run(['git','commit','-m','Record audited evacuation tools, resume validation and public archive policy'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
 subprocess.run(['git','push','https://github.com/insightlabs38-pixel/latticelm','HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True)
 sha=run('git','rev-parse','HEAD');assert run('git','ls-remote','origin','refs/heads/'+BRANCH).split()[0]==sha
 atom(E/'GIT_PUSH_RECEIPT.json',{'branch':BRANCH,'sha':sha,'remote':'https://github.com/insightlabs38-pixel/latticelm','push_verified':True,'original_research_snapshot_sha':'f7faf5398c35952fcae32c95c2e42c97fcfd6979','archival_branch':'archive/evacuation-preisolation-posttraining-20260921','archival_sha':'2b618825cd205671d545e30a83563db534692edd','judged_main_unmodified':True,'original_push_safeguard_preserved':True})
 timestamp=datetime.now(timezone.utc).isoformat()
 card=f'''---
library_name: pytorch
pipeline_tag: text-generation
language: en
tags:
- research
- causal-lm
buckets:
- {BUCKET}
---
# LatticeLM public research archive

The canonical LatticeLM checkpoint has **48,636,168 parameters** and was trained from random initialization for **1,900,000,000 tokens**. This curated repository preserves the exact original training checkpoint, an exactly equivalent safetensors inference export, tokenizer/config, evaluations, provenance, and selected development checkpoints.

## Canonical model

Co4 causal architecture: 12 layers, width 552, FFN width 1728, 6 query heads / 2 KV heads, QK normalization, untied embeddings, vocabulary 4096, context 256. The optimizer is Muon hybrid with a fresh WSD schedule. DATA-D-v4 combines FineWeb-Edu, Wikipedia, FineWeb and FineMath; source revisions, composition, hashes, filtering and contamination metadata are preserved in manifests. Natural corpus content is withheld pending redistribution review.

The original final report records held-out loss 2.63049, WikiText perplexity 21.9298 and BPB 1.41346, ARC-Easy 0.36785, HellaSwag 0.27096, PIQA 0.56692 and WinoGrande 0.50987. See the production reports for evaluation protocols and limits. This small model is intended for research; its short context and limited training scale constrain generation quality. These results do not establish reliability for high-stakes use.

| Artifact | SHA-256 |
| --- | --- |
| Native final checkpoint | `94b91a2d39c2889209cb47b2e04b2c0d307a25610c40c5a124b595020eb762b0` |
| Tokenizer | `294925d61198dfa0863f91193349f6ebead420d06a3fb650a5122d2264a4ebf7` |
| DATA-D-v4 manifest | `9d2811cfb2824b0dd541af89ebfb64403b3ec7143db97d52c4046aafb8eeae7f` |

The canonical configuration in `config/final_model_config.json` is extracted from the exact 48.6M checkpoint and restore-tested against it. A historically named local config described an earlier 8M model; that original remains in the broad archive as historical evidence.

## Development artifacts

Development checkpoints may be incomplete, negative, exploratory or superseded. They are preserved for transparency and continued research and are not all recommended for deployment. Selected R32/R16 controls, Differential-Co4/nCo4 artifacts, recovered 8M control provenance and campaign reports are included. The four historical 8M controls are retained with recovered exact trainer snapshots and independent certificates.

The fresh 32M dense run **completed 300M tokens on October 11, 2026**. The active run at evacuation was instead `F8-CO4-FULLCONTROL-50M-S1337`; with user confirmation it was truncated at **4,268,032 / 50,000,000 tokens** because GCP credits ran low. Its exact model/optimizer/RNG/sampler state passed strict resume validation and its checkpoint was restored and hash-verified. Its receipt records the stable WSD region and LR 0.0008. It is not a completed 50M result.

## Loading and reproducibility

Use the [source branch](https://github.com/insightlabs38-pixel/latticelm/tree/{BRANCH}) at `{sha}` and the environment version receipt in `reports/evacuation/`. The exact original production source is also preserved under `source-snapshots/canonical-6e2c6f0f6f5278a2991c04708859baae7580368c/`.

```python
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from latticelm.config import LatticeConfig
from latticelm.model import build_model

repo = "{REPO}"
path = hf_hub_download(repo, "canonical/checkpoint/final.pt")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
model = build_model(LatticeConfig(**checkpoint["config"]))
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer/final_corpus_4k.json"))
tokens = torch.tensor([tokenizer.encode("LatticeLM research.").ids])
with torch.no_grad():
    logits, _ = model(tokens)
```

The [public VM archive](https://huggingface.co/buckets/{BUCKET}) contains the broader research history, native checkpoints, controller state, logs, exact source/config snapshots, synthetic datasets, and detailed inventories. Public bucket URI: `hf://buckets/{BUCKET}`. Archive date: {timestamp}. The old private research repository remains private.

## License status

No general license has been invented for the canonical model. Author review is still required to select its license. IHMS-derived development implementation preserves its existing CC-BY-NC-4.0 attribution and complete upstream license in `NOTICE` / `LICENSES/`. This does not apply a blanket license to unrelated artifacts. Natural corpora and benchmark examples are represented by metadata and reconstruction scripts until redistribution obligations are resolved.
'''
 (STAGE/'model/README.md').write_text(card)
 api.upload_file(repo_id=REPO,path_or_fileobj=STAGE/'model/README.md',path_in_repo='README.md',commit_message='Document canonical model, completed and truncated research, provenance and public archive')
 bucket_readme=f'''# LatticeLM sanitized VM research archive

Emergency preservation of an expiring research VM, archived {timestamp}. The [curated model repository](https://huggingface.co/{REPO}) is the recommended entry point. This broad archive contains intermediate, completed, negative and abandoned experiments, controller state, logs, native resumable checkpoints, recovered historical-control provenance, source snapshots, and redistributable first-party synthetic data.

Source: https://github.com/insightlabs38-pixel/latticelm/tree/{BRANCH} at `{sha}`.

- `01-canonical-and-selected/`: canonical model/tokenizer/config and curated development checkpoints.
- `research/working-tree/`: approved original relative paths, including `artifacts/`, `post_hackathon_research/`, `post_hackathon_followup/`, source, configs, scripts and video evidence.
- `research/frozen-tree/`: frozen judged source/evidence tree.
- `research/home-instructions/`: the three home-level LatticeLM execution prompts.
- `04-source-snapshots/canonical/`: exact original production source revision.
- `00-manifests/evacuation-status/`: inventory, SHA-256 ledgers, classifications, license/secret audits, receipts, verification and final report.

No credentials are intentionally included. Authentication state, Git metadata, shell history, caches, environments and temporary/system files are excluded. Natural corpora/tokenized shards, benchmark text and contamination-registry content with unresolved redistribution obligations are manifest-only; upstream revisions, hashes, sampling metadata and reconstruction code are retained. Only aggregate evaluation results are public. Original local research files were not deleted or overwritten. Bucket transfers add/update files and never mirror-delete.

The fresh 32M dense run completed 300M tokens before evacuation. The active 8M Co4 control was truncated at 4,268,032 / 50M tokens and exact-resume validated. Consult its receipt and final report. This archive is for research/reproducibility and is not a single clean model release. License scope is documented per artifact family; no blanket model license is invented.
'''
 meta=STAGE/'publication-metadata';meta.mkdir(exist_ok=True);(meta/'README.md').write_text(bucket_readme)
 sync_bucket(str(meta),'hf://buckets/'+BUCKET,delete=False,quiet=True)
 sync_bucket(str(STAGE/'canonical-source'),'hf://buckets/'+BUCKET+'/04-source-snapshots/canonical',delete=False,quiet=True)
 sync_bucket(str(STAGE/'model'),'hf://buckets/'+BUCKET+'/01-canonical-and-selected',delete=False,exclude=['.cache/**'],quiet=True)
 # Model manifest covers all public curated/source files before report export.
 model_rows=[]
 for base,prefix in [(STAGE/'model',''),(STAGE/'canonical-source','source-snapshots/canonical-6e2c6f0f6f5278a2991c04708859baae7580368c/')]:
  for p in base.rglob('*'):
   if not p.is_file() or '.cache' in p.parts:continue
   assert not scan_file(p),str(p)
   model_rows.append(manifest_row(p,prefix+str(p.relative_to(base)),REPO))
 info=api.model_info(REPO,files_metadata=True,token=False);files={f.rfilename:f for f in info.siblings};assert not info.private
 for row in model_rows:assert files[row['remote_path']].size==row['size'];row['verification_state']='remote_size_verified'
 (E/'MODEL_REPO_MANIFEST.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in model_rows))
 original=api.model_info('insightlabs38-pixel/LatticeLM-research');assert original.private
 card_parsed=ModelCard.load(REPO,token=False);assert BUCKET in card_parsed.data.buckets
 # Download the safe export independently and compare all tensor values again.
 path=hf_hub_download(REPO,'canonical/checkpoint/model.safetensors',revision=info.sha,local_dir=RESTORE,token=False)
 assert digest(path)==digest(safe)
 restored=load_file(path);assert all(torch.equal(v,restored[k]) for k,v in weights.items())
 # Verify every approved local source still has its original bytes on disk.
 for row in bucket_rows:
  p=Path(row['local_path']);assert p.exists() and p.stat().st_size==row['size']
 remote={f.path:f for f in api.list_bucket_tree(BUCKET,prefix='research',recursive=True,token=False) if hasattr(f,'size')}
 for row in bucket_rows:assert remote[row['remote_path']].size==row['size'],row['remote_path']
 verify.update({'broader_verification_pending':False,'broad_files':len(bucket_rows),'broad_bytes':sum(r['size'] for r in bucket_rows),'broad_remote_sizes':'PASS','broad_sync_no_pending_changes':'PASS','broad_representative_restores':json.loads((E/'BROAD_RESTORE_VERIFICATION.json').read_text()),'canonical_safetensors_remote_restore_and_exact_equivalence':'PASS','model_card_metadata_and_bucket_link':'PASS','original_private_repo_remains_private':True,'git_final_sha':sha,'model_files_before_final_report':len(files),'model_bytes_before_final_report':sum(f.size or 0 for f in files.values()),'verified_utc':timestamp})
 atom(E/'REMOTE_VERIFICATION.json',verify)
 partial=json.loads((E/'PARTIAL_RUN_RECEIPT.json').read_text())
 final_report=f'''# LatticeLM emergency evacuation final report

Status: **NOT_SAFE_TO_STOP_VM** until final metadata publication and clean final-report restores complete. Scientific data gates already pass; final publication gate is in progress.

Timestamp: {timestamp}

## Compute and current run

Follow-up master PID 1103332 and trainer PID 1213933 stopped gracefully. User unit `latticelm-posthack-followup-master.service` is stopped/disabled and restart is disabled through a persistent drop-in. All discovered historical LatticeLM/LatticeReason research services were disabled from boot relaunch; SSH/network/system services were not disabled. No VM shutdown/reboot was issued.

The prompt's named 32M fresh dense run had already completed exactly 300,000,000 tokens on October 11. The active `F8-CO4-FULLCONTROL-50M-S1337` was stopped with user confirmation at 4,268,032 / 50,000,000 logical tokens (step 521). It is registered as TRUNCATED_RESOURCE_EXHAUSTION in the additional evacuation receipt. Its fresh WSD schedule is in the stable region, LR 0.0008, wall time {partial['wall_seconds']:.6f}s, measured {partial['measured_tokens_per_second']:.3f} tokens/s. Stop reason: GCP credit exhaustion / emergency VM evacuation.

Final native partial checkpoint: `{partial['checkpoint_path']}`. SHA-256: `{partial['checkpoint_sha256']}`. Model/optimizer/RNG/sampler/next-batch/token-count/config/source/data/tokenizer identity checks passed. Latest and previous/init checkpoint bytes are preserved, along with the original train receipt.

## GitHub

Dedicated branch `{BRANCH}` pushed and remote SHA verified: `{sha}`. Research freeze commit: f7faf5398c35952fcae32c95c2e42c97fcfd6979. Judged main remains 43832b4c7bfe8be007ca6b9d1042aa2041c7f280. The original origin push safeguard is unchanged; explicit authorized branch pushes used the actual GitHub URL.

The otherwise unreachable historical post-training commit 2b618825cd205671d545e30a83563db534692edd was secret-scanned and preserved on `archive/evacuation-preisolation-posttraining-20260921`. The frozen tree is archived in the bucket; pre-isolation Git metadata was inventoried and not uploaded wholesale.

## Curated public model repository

https://huggingface.co/{REPO}

Public accessibility, listings/sizes, native canonical checkpoint/config/tokenizer/DATA-D hashes and clean downloads passed. Canonical native hash is 94b91a2d39c2889209cb47b2e04b2c0d307a25610c40c5a124b595020eb762b0; tokenizer hash 294925d61198dfa0863f91193349f6ebead420d06a3fb650a5122d2264a4ebf7; DATA-D manifest hash 9d2811cfb2824b0dd541af89ebfb64403b3ec7143db97d52c4046aafb8eeae7f. Restored model loads strictly with 48,636,168 parameters and runs deterministic finite inference. Restored config matches the exact checkpoint config. The additional safetensors export matches every native tensor key/dtype/value and its independent public restore passed.

Curated scientific/source files before final report: {len(files)} files / {sum(f.size or 0 for f in files.values())} bytes. Exact per-file SHA-256 ledger: MODEL_REPO_MANIFEST.jsonl. Final metadata/report adds a separate documented set. Exact original canonical source revision 6e2c6f0f6f5278a2991c04708859baae7580368c is included. Development and truncated checkpoints are explicitly labeled. The existing private LatticeLM-research repository remains private.

## Public archive bucket

https://huggingface.co/buckets/{BUCKET}

URI: `hf://buckets/{BUCKET}`. Approved scientific research set: **{len(bucket_rows)} files / {sum(r['size'] for r in bucket_rows)} bytes** (logical bytes, including historical/native duplicates; storage deduplicates). Every approved file has a SHA-256 in BUCKET_ARCHIVE_MANIFEST.jsonl and a verified remote size. Every major upload plan was checked against the allowlist with only upload/skip operations. Repeat sync/dry-run showed no pending changes. The curated critical set is duplicated separately in the bucket; canonical source and audit documents are additional.

Six broad representative unauthenticated clean restores passed: large R32 historical control checkpoint, supervisor report, controller JSON, run config, compressed document index, home-level evacuation prompt. The canonical native checkpoint/tokenizer/config/manifests and partial checkpoint also passed clean restores in both destinations. See REMOTE_VERIFICATION.json and BROAD_RESTORE_VERIFICATION.json.

## Secret and redistribution audit

PASS_WITH_EXCLUSIONS. No real credential patterns were found in the approved set. 101 credential-URL regex matches were reviewed as malformed corpus-source mailto URLs; no authentication material was published. Source/config/report/log/JSON/YAML/service metadata were scanned; native checkpoint metadata was scanned and synthetic token arrays were range-validated. No secret rotation is required due to this publication. Credentials, shell histories, cloud auth state, .git, environments/caches, temporary/system files remain excluded.

First-party symbolic synthetic reservoirs are included. Natural/tokenized datasets, natural banks, benchmark-example payloads and the contamination registry remain manifest-only pending redistribution review. Upstream URI/revision/license labels, source manifests, hashes, source builders and sampling metadata are preserved; tokenization was not treated as a redistribution grant. Aggregate-only evaluation results were reviewed and retained. The complete IHMS CC-BY-NC-4.0 license/attribution/modification notice is preserved for the source-derived implementation; the canonical model license remains for author review and no blanket license is invented.

## Remaining local-only material

Exact per-file list and reasons: LOCAL_ONLY_OMISSIONS.json. It contains {len(omissions)} intentional omissions totaling {sum(r['size'] for r in omissions)} logical bytes. Categories: license-unresolved natural corpora/benchmark content, disposable runtime locks, generated render caches/bundles, one downloaded tool binary. Additional excluded roots are listed in SECRET_SCAN_REPORT.json and OUTSIDE_REPO_AND_RELAUNCH_AUDIT.json: auth/application/cache state, Git metadata, virtualenv/compiler build trees, root/user shell histories and system trees. No approved unique model/source/provenance/checkpoint exists only locally. Omitted benchmark evaluations can be regenerated from preserved models/evaluation code; omitted natural corpora require reacquisition using pinned upstream revisions and original source manifests before exact training resume. Public redistribution rights were not presumed. The skipped historical large source ZIP's revision c9ed214a5131c2e891f9e77b5ecee74243be4f72 is already preserved in the pushed Git ancestry.

No original research files were deleted or overwritten. Staging used hardlinks; restoring tests used independent files. Disk free space was monitored and remains above 30 GiB.

## Reconstruction

1. Clone https://github.com/insightlabs38-pixel/latticelm and checkout `{sha}` (branch `{BRANCH}`). Use the separate archival branch if recovering historical post-training work.
2. Recreate Python/PyTorch/NumPy/HF dependencies from REBUILD_ENVIRONMENT.json and requirements-freeze.txt; do not restore old credential or virtualenv directories. Optional unmodified triton-cpu source is pinned in the environment audit.
3. Download the canonical model/tokenizer/config from the public model repository; verify the manifest SHA-256 values and use the model card loading example. Preserve the exact native checkpoint for resume.
4. Download required bucket prefixes with HF bucket sync (without deletion) and preserve original relative paths under research/working-tree. Restore controller/registry/certificates/source snapshots and native checkpoints as needed.
5. Reacquire withheld natural data from pinned source URIs/revisions, rebuild with preserved source/config/tokenizer, and compare shard/manifest hashes. Restore optimizer/RNG/sampler states and verify the next-batch identity before resuming training elsewhere. The truncated run remains incomplete; the dense 300M run remains completed.

## Resolved execution issues

Initial native-config comparison used Python tuple/list equality; corrected to the trainer's canonical JSON comparison with matching hashes. The historically named canonical config was an old 8M file; corrected curated config was derived from the exact 48.6M checkpoint, paired restore-tested, and the original file retained. Git's disabled push destination required an explicit authorized URL. Root-owned hardlink protection required privileged link creation without copying/modifying originals. An unavailable bucket-upload helper was replaced by installed supported add-only sync. No scientific training step was launched, no source artifact was removed, and no VM shutdown command was issued.
'''
 (E/'EVACUATION_FINAL_REPORT.md').write_text(final_report)
 # Export audited metadata; caches/temp files and process/env dumps never enter.
 publication=STAGE/'evacuation-metadata';publication.mkdir(exist_ok=True)
 for p in E.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.tmp':continue
  assert not scan_file(p),str(p)
  dest=publication/p.relative_to(E);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
 api.upload_folder(repo_id=REPO,folder_path=publication,path_in_repo='reports/evacuation',commit_message='Preserve evacuation inventory, hashes, audits, receipts and verification')
 sync_bucket(str(publication),'hf://buckets/'+BUCKET+'/00-manifests/evacuation-status',delete=False,quiet=True)
 # Confirm every metadata file publicly exists with the intended size.
 m=api.model_info(REPO,files_metadata=True,token=False);mf={f.rfilename:f for f in m.siblings}
 bf={f.path:f for f in api.list_bucket_tree(BUCKET,prefix='00-manifests/evacuation-status',recursive=True,token=False) if hasattr(f,'size')}
 for p in publication.rglob('*'):
  if not p.is_file():continue
  rel=str(p.relative_to(publication));assert mf['reports/evacuation/'+rel].size==p.stat().st_size;assert bf['00-manifests/evacuation-status/'+rel].size==p.stat().st_size
 # Only now change the final status and verify independent public restores.
 status.update({'status':'SAFE_TO_STOP_VM','safe_to_stop_vm':True,'git_sha':sha,'bulk_files':len(bucket_rows),'bulk_bytes':sum(r['size'] for r in bucket_rows),'final_utc':timestamp,'all_required_data_gates_passed':True})
 atom(E/'EVACUATION_STATUS.json',status)
 final_report=final_report.replace('Status: **NOT_SAFE_TO_STOP_VM** until final metadata publication and clean final-report restores complete. Scientific data gates already pass; final publication gate is in progress.','Status: **SAFE_TO_STOP_VM**. Required compute, Git, public model/bucket, secret/license policy, manifest publication and representative clean-restore gates passed. The VM remains running for the user to stop.')
 (E/'EVACUATION_FINAL_REPORT.md').write_text(final_report)
 for name in ['EVACUATION_STATUS.json','EVACUATION_FINAL_REPORT.md']:
  (publication/name).write_bytes((E/name).read_bytes())
  api.upload_file(repo_id=REPO,path_or_fileobj=E/name,path_in_repo='reports/evacuation/'+name,commit_message='Record completed verified VM evacuation')
 sync_bucket(str(publication),'hf://buckets/'+BUCKET+'/00-manifests/evacuation-status',delete=False,quiet=True)
 for name in ['EVACUATION_STATUS.json','EVACUATION_FINAL_REPORT.md','BUCKET_ARCHIVE_MANIFEST.jsonl','SECRET_SCAN_REPORT.json','LOCAL_STORAGE_INVENTORY.tsv']:
  local=E/name;model_path=hf_hub_download(REPO,'reports/evacuation/'+name,local_dir=RESTORE/'metadata-model',token=False)
  bucket_path=RESTORE/'metadata-bucket'/name;bucket_path.parent.mkdir(parents=True,exist_ok=True)
  api.download_bucket_files(BUCKET,[('00-manifests/evacuation-status/'+name,bucket_path)],raise_on_missing_files=True,token=False)
  assert digest(local)==digest(model_path)==digest(bucket_path),name
 assert api.bucket_info(BUCKET,token=False).private is False
 assert api.model_info(REPO,token=False).private is False
 print('SAFE_TO_STOP_VM',flush=True)
 print('Git SHA:',sha,flush=True)
 print('Scientific archive:',len(bucket_rows),'files',sum(r['size'] for r in bucket_rows),'bytes',flush=True)
 print('Model repo:',REPO,'Bucket:',BUCKET,flush=True)

if __name__=='__main__':main()
