"""Moderate, verified delta uploads for the separate final-week namespace."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import fcntl
import subprocess
import tempfile
import shutil
import errno
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from followup_common import ROOT, RESEARCH, atomic_json, decision, sha256, utc
from followup_common import atomic_bytes,verified_checkpoint,registrations,complete_evidence

REPO = "insightlabs38-pixel/LatticeLM-research"
PREFIX = "post-hackathon-followup/2026-10-11"
SECRET = Path("/home/insightlabs38/.config/latticelm-followup/huggingface.token")
_HASH_CACHE_PATH=RESEARCH/'.archive-hash-cache.json'
_HASHES=json.loads(_HASH_CACHE_PATH.read_text()) if _HASH_CACHE_PATH.exists() else {}


def cached_hash(path):
    s=path.stat();key=json.dumps((str(path),s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
    if key not in _HASHES:
        _HASHES[key]=sha256(path);atomic_json(_HASH_CACHE_PATH,_HASHES)
    return _HASHES[key]


def historical_files():
    """Frozen historical evidence and actual deterministic input shards, once."""
    old=ROOT/'post_hackathon_research';files=[]
    for p in old.rglob('*'):
        if not p.is_file() or p.is_symlink() or p.name=='previous.pt':continue
        if p.name in ('documents.jsonl','capability_documents.jsonl'):continue # verified gzip + source SHA restores these
        if p.suffix in ('.json','.jsonl','.md','.csv','.png','.sha256','.gz','.int32','.zip','.pt','.log','.service'):files.append(p)
    manifest=ROOT/'artifacts/data/data_d_v4/canonical-2250m/manifest.json'
    top=json.loads(manifest.read_text());files.append(manifest)
    for item in top['shards']:
        child=manifest.parent/item['manifest_path'];row=json.loads(child.read_text())
        if cached_hash(child)!=item['manifest_sha256']:raise ValueError('Canonical child manifest changed')
        shard=manifest.parent/row['path']
        if cached_hash(shard)!=row['sha256']:raise ValueError('Canonical data shard changed')
        files.extend((child,shard))
    files.append(ROOT/'artifacts/final_production_run/final.pt')
    return files


def source_snapshot():
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    destination=RESEARCH/f'source-snapshot-{revision}.zip'
    if not destination.exists():
        content=subprocess.check_output(['git','archive','--format=zip',revision,'scripts','src','tests','pyproject.toml','README.md','post_hackathon_followup'],cwd=ROOT)
        atomic_bytes(destination,content)
    return destination


def git_blob_hash(path):
    h = hashlib.sha1(); h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def remote_matches(path, sibling, digest):
    if not sibling or sibling.size != path.stat().st_size:return False
    if sibling.lfs:return sibling.lfs.sha256 == digest
    return sibling.blob_id == git_blob_hash(path)


def selected_files():
    files=[]
    for path in RESEARCH.iterdir():
        if path.is_file() and not path.name.startswith('.') and path.suffix in (".json", ".jsonl", ".md") and path.name != "FOLLOWUP_ARCHIVE_MANIFEST.json":files.append(path)
    for directory in ("configs", "certificates"):
        base=RESEARCH/directory
        if base.exists():files.extend(p for p in base.rglob("*") if p.is_file() and p.suffix in (".json", ".md"))
    for directory in ('decisions','comparisons','plots'):
        base=RESEARCH/directory
        if base.exists():files.extend(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.png','.svg','.csv'))
    runs=RESEARCH/"runs"
    if runs.exists():
        for run in runs.iterdir():
            if not run.is_dir():continue
            files.extend(p for p in run.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl",'.log'))
            if run.name.startswith('SMOKE-'):continue
            mp=run/'checkpoint_manifest.json'
            if mp.exists():
                m=json.loads(mp.read_text());files.extend(run/i['file'] for i in m['milestones'])
            receipt=run/"train_receipt.json"
            if receipt.exists() and json.loads(receipt.read_text())["status"] == "TRAINED":
                manifest=json.loads((run/"checkpoint_manifest.json").read_text())
                files.extend(run/i["file"] for i in manifest["rolling"][:1]+manifest["milestones"])
            elif (RESEARCH/'FINAL_ARCHIVE_REQUEST.json').exists() and mp.exists():
                files.extend(run/i['file'] for i in json.loads(mp.read_text())['rolling'])
    # Exact tokenizer/config pairing is shared by every run and the frozen model.
    files += [ROOT/"artifacts/tokenizers/final_corpus_4k.json",ROOT/"artifacts/data/data_d_v4/canonical-2250m/manifest.json"]
    files.extend(RESEARCH.glob("source-snapshot-*.zip"))
    return sorted(set(files))


def archive(final=False):
    os.nice(10)
    source_snapshot()
    api=HfApi(token=SECRET.read_text().strip())
    info=api.repo_info(REPO,files_metadata=True)
    if not info.private:raise RuntimeError("archive destination must be private")
    remote={s.rfilename:s for s in info.siblings}
    by_digest={s.lfs.sha256:s for s in info.siblings if s.lfs}
    manifest_path=RESEARCH/"FOLLOWUP_ARCHIVE_MANIFEST.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema":"followup-archive-v1","files":{}}
    selected=selected_files()+historical_files()
    # Scientific outputs and their code precede historical/input delta uploads.
    for local in dict.fromkeys(selected):
        if local.is_symlink():raise RuntimeError("symlinks refused")
        rel=local.relative_to(ROOT).as_posix()
        destination=('post-hackathon/2026-10-01/'+local.relative_to(ROOT/'post_hackathon_research').as_posix()
                     if local.is_relative_to(ROOT/'post_hackathon_research') else
                     PREFIX+"/"+(local.relative_to(RESEARCH).as_posix() if local.is_relative_to(RESEARCH) else "shared/"+rel))
        # Snapshot mutable reports/receipts so hash, upload, and remote verification refer to identical bytes.
        with tempfile.TemporaryDirectory(prefix="followup-upload-") as tmp:
            snapshot=Path(tmp)/local.name
            if local.suffix in ('.pt','.int32','.gz','.zip'):
                try:
                    os.link(local,snapshot)  # immutable checkpoint; zero-copy stable inode
                except OSError as error:
                    if error.errno != errno.EXDEV: raise
                    shutil.copyfile(local,snapshot)
            else:
                snapshot.write_bytes(local.read_bytes())
            digest=cached_hash(local) if local.suffix in ('.pt','.int32','.gz','.zip') else sha256(snapshot)
            sibling=remote.get(destination)
            if not remote_matches(snapshot,sibling,digest) and digest in by_digest:
                sibling=by_digest[digest];destination=sibling.rfilename
            changed=not remote_matches(snapshot,sibling,digest)
            revision=info.sha
            if changed:
                if snapshot.stat().st_size < 2_000_000:
                    data=snapshot.read_bytes()
                    if re.search(rb"github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}",data):raise RuntimeError("credential-pattern content refused: "+rel)
                print("Delta upload: "+rel,flush=True)
                result=api.upload_file(path_or_fileobj=str(snapshot),path_in_repo=destination,repo_id=REPO,
                                      commit_message="Verified final-week research delta",repo_type="model")
                revision=result.oid
                current=api.get_paths_info(REPO,[destination],revision=revision)
                sibling=next((s for s in current if s.rfilename==destination),None)
                remote[destination]=sibling
                if sibling and sibling.lfs:by_digest[sibling.lfs.sha256]=sibling
            if not remote_matches(snapshot,sibling,digest):raise RuntimeError("remote hash/size verification failed: "+destination)
            manifest["files"][rel]={"local_sha256":digest,"size_bytes":snapshot.stat().st_size,"remote":destination,
                "repo":REPO,"revision":revision,"verified":True,"at":utc(),
                "verification":"LFS SHA256" if sibling.lfs else "Git blob SHA1 and size","uploaded_this_pass":changed}
        manifest["updated_utc"]=utc();manifest["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        manifest["hard_deadline_utc"]="2026-10-17T20:10:00+00:00"
        atomic_json(manifest_path,manifest)
    # Publish the manifest after its entries have all been verified; no recursive self hash.
    result=api.upload_file(path_or_fileobj=str(manifest_path),path_in_repo=PREFIX+"/FOLLOWUP_ARCHIVE_MANIFEST.json",
                          repo_id=REPO,commit_message="Final-week verified delta manifest")
    atomic_json(RESEARCH/"last_archive_receipt.json",{"at":utc(),"revision":result.oid,"file_count":len(manifest["files"]),"manifest_sha256":sha256(manifest_path)})
    decision("HF_DELTA_VERIFIED",revision=result.oid,file_count=len(manifest["files"]))
    return manifest


def representative_loads(require_new=False):
    """Download real remote bytes and load paired artifacts, at one CPU thread."""
    import torch
    from types import SimpleNamespace
    from evaluate_gibc import LatticeHarnessLM
    torch.set_num_threads(1)
    api=HfApi(token=SECRET.read_text().strip());info=api.repo_info(REPO,files_metadata=True)
    remote={s.rfilename:s for s in info.siblings};results=[]
    manifest=json.loads((RESEARCH/'FOLLOWUP_ARCHIVE_MANIFEST.json').read_text())
    tokrel='artifacts/tokenizers/final_corpus_4k.json';entry=manifest['files'][tokrel]
    with tempfile.TemporaryDirectory(prefix='followup-remote-load-') as directory:
        tokenizer=Path(hf_hub_download(REPO,entry['remote'],revision=entry['revision'],token=SECRET.read_text().strip(),local_dir=directory))
        if sha256(tokenizer)!=entry['local_sha256']:raise RuntimeError('Downloaded tokenizer hash mismatch')
        results.append({'kind':'non-model tokenizer','remote':entry['remote'],'sha256':entry['local_sha256'],'passed':True})
        frozen='final-production/final.pt';expected=json.loads((RESEARCH/'FOLLOWUP_INPUT_AUDIT.json').read_text())
        expected=next(r['sha256'] for r in expected['trusted_files'] if r['path']=='artifacts/final_production_run/final.pt')
        selections=[('frozen submission model',frozen,expected,info.sha)]
        candidates=[r for r in registrations() if not r.get('smoke') and not r.get('historical_experiment') and complete_evidence(RESEARCH/'runs'/r['experiment_id'],r)]
        candidates.sort(key=lambda r:(not bool(r.get('full_co4')),r['experiment_id']!='F32-DENSE-S1337-FRESH300M'))
        if candidates:
            r=candidates[0];cp=verified_checkpoint(RESEARCH/'runs'/r['experiment_id'])
            local='post_hackathon_followup/runs/'+r['experiment_id']+'/'+cp['file'];m=manifest['files'][local]
            selections.append(('new completed research model '+r['experiment_id'],m['remote'],cp['sha256'],m['revision']))
        for kind,path,digest,revision in selections:
            modelpath=Path(hf_hub_download(REPO,path,revision=revision,token=SECRET.read_text().strip(),local_dir=directory))
            if sha256(modelpath)!=digest:raise RuntimeError('Downloaded checkpoint hash mismatch')
            lm=LatticeHarnessLM(str(modelpath),str(tokenizer));score=lm.loglikelihood([SimpleNamespace(args=('The sky',' is blue.'))])[0][0]
            if not torch.isfinite(torch.tensor(score)):raise RuntimeError('Remote model evaluator is not finite')
            results.append({'kind':kind,'remote':path,'revision':revision,'sha256':digest,'sample_loglikelihood':score,'passed':True})
            del lm
    output={'at':utc(),'results':results,'new_research_model_available':bool(candidates),'passed':all(r['passed'] for r in results) and bool(candidates),
            'status':'PASSED' if candidates else 'FROZEN_AND_TOKENIZER_PASSED_NEW_ENDPOINT_PENDING'}
    atomic_json(RESEARCH/'HF_REPRESENTATIVE_LOAD_RECEIPT.json',output)
    if require_new and not candidates:raise RuntimeError('No completed new research model for mandatory remote load verification')
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--experiment-id",default="HF-DELTA");p.add_argument('--loads-only',action='store_true');args=p.parse_args()
    with (RESEARCH/".archive.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("archive already active")
        if args.loads_only:representative_loads()
        else:archive()
