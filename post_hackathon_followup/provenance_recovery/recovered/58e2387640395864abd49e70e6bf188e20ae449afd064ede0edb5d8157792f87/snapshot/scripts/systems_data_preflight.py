"""Canonical repository, manifest, checkpoint, and credential preflight."""
from __future__ import annotations
import hashlib,json,re,subprocess,sys
from pathlib import Path
import torch
from latticelm.data_d import verify_top_manifest
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts";TOK=ART/"tokenizers/babylm_2026_4k.json"
def main():
 subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,check=True)
 subprocess.run([sys.executable,"-m","compileall","-q","src","scripts"],cwd=ROOT,check=True)
 subprocess.run(["git","diff","--check"],cwd=ROOT,check=True)
 verify_top_manifest(ART/"data/phase7f_v2r1/canonical-100m/manifest.json",TOK)
 p=ART/"checkpoints/co4-l-data-d-seed2026-100m/latest.pt";expected=p.with_suffix(".sha256").read_text().strip();h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 if h.hexdigest()!=expected:raise RuntimeError("checkpoint hash mismatch")
 x=torch.load(p,map_location="cpu",weights_only=False);required={"model","optimizer","scheduler","python_rng_state","torch_rng_state","next_batch_sha256","data_manifest_sha256"}
 if required-set(x) or x.get("tokens_seen")!=100_000_000:raise RuntimeError("checkpoint verifier failed")
 pattern=re.compile(rb"(?:hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")
 for name in subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).split(b"\0"):
  if not name:continue
  data=(ROOT/name.decode()).read_bytes()
  if pattern.search(data):raise RuntimeError(f"credential pattern in tracked file: {name.decode()}")
 print(json.dumps({"tests":"PASS","compile":"PASS","diff":"PASS","manifest":"PASS","checkpoint":"PASS","credential_scan":"PASS"}))
if __name__=="__main__":main()
