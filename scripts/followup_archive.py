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

REPO = "insightlabs38-pixel/LatticeLM-research"
PREFIX = "post-hackathon-followup/2026-10-11"
SECRET = Path("/home/insightlabs38/.config/latticelm-followup/huggingface.token")


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
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md") and path.name != "FOLLOWUP_ARCHIVE_MANIFEST.json":files.append(path)
    for directory in ("configs", "certificates"):
        base=RESEARCH/directory
        if base.exists():files.extend(p for p in base.rglob("*") if p.is_file() and p.suffix in (".json", ".md"))
    runs=RESEARCH/"runs"
    if runs.exists():
        for run in runs.iterdir():
            if not run.is_dir() or run.name.startswith("SMOKE-"):continue
            files.extend(p for p in run.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl"))
            receipt=run/"train_receipt.json"
            if receipt.exists() and json.loads(receipt.read_text())["status"] == "TRAINED":
                manifest=json.loads((run/"checkpoint_manifest.json").read_text())
                files.extend(run/i["file"] for i in manifest["rolling"][:1]+manifest["milestones"])
    # Exact tokenizer/config pairing is shared by every run and the frozen model.
    files += [ROOT/"artifacts/tokenizers/final_corpus_4k.json",ROOT/"artifacts/data/data_d_v4/canonical-2250m/manifest.json"]
    files.extend(RESEARCH.glob("source-snapshot-*.zip"))
    return sorted(set(files))


def archive():
    api=HfApi(token=SECRET.read_text().strip())
    info=api.repo_info(REPO,files_metadata=True)
    if not info.private:raise RuntimeError("archive destination must be private")
    remote={s.rfilename:s for s in info.siblings}
    manifest_path=RESEARCH/"FOLLOWUP_ARCHIVE_MANIFEST.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema":"followup-archive-v1","files":{}}
    for local in selected_files():
        if local.is_symlink():raise RuntimeError("symlinks refused")
        rel=local.relative_to(ROOT).as_posix()
        destination=PREFIX+"/"+(local.relative_to(RESEARCH).as_posix() if local.is_relative_to(RESEARCH) else "shared/"+rel)
        # Snapshot mutable reports/receipts so hash, upload, and remote verification refer to identical bytes.
        with tempfile.TemporaryDirectory(prefix="followup-upload-") as tmp:
            snapshot=Path(tmp)/local.name
            if local.suffix == ".pt":
                try:
                    os.link(local,snapshot)  # immutable checkpoint; zero-copy stable inode
                except OSError as error:
                    if error.errno != errno.EXDEV: raise
                    shutil.copyfile(local,snapshot)
            else:
                snapshot.write_bytes(local.read_bytes())
            digest=sha256(snapshot); sibling=remote.get(destination)
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
                current=api.repo_info(REPO,revision=revision,files_metadata=True)
                sibling=next((s for s in current.siblings if s.rfilename==destination),None)
                remote[destination]=sibling
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


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--experiment-id",default="HF-DELTA");p.parse_args()
    with (RESEARCH/".archive.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("archive already active")
        archive()
