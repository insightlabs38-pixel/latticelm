"""Verify local/remote recovery state and resume the frozen GCP 10M lineage."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / "artifacts/checkpoints/final-gcp-10m"


def digest(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="explicitly authorize random initialization")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    latest = RECOVERY / "latest.pt"; checksum = RECOVERY / "latest.sha256"
    if latest.exists() and checksum.exists() and digest(latest) == checksum.read_text().strip():
        mode = "--resume"
    else:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        repo = os.environ.get("LATTICELM_HF_REPO", "insightlabs38-pixel/LatticeLM-research")
        try:
            if not token: raise RuntimeError("no process-local HF token")
            RECOVERY.mkdir(parents=True, exist_ok=True)
            remote_pt = Path(hf_hub_download(repo, "recovery/final-gcp-10m/latest.pt", repo_type="model", token=token))
            remote_sum = Path(hf_hub_download(repo, "recovery/final-gcp-10m/latest.sha256", repo_type="model", token=token)).read_text().strip()
            if digest(remote_pt) != remote_sum: raise RuntimeError("remote checkpoint checksum mismatch")
            latest.write_bytes(remote_pt.read_bytes()); checksum.write_text(remote_sum + "\n"); mode = "--resume"
        except Exception as error:
            if not args.fresh:
                raise RuntimeError("no valid recovery checkpoint; refusing silent restart") from error
            mode = "--fresh"
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_phase7c_final.py"), mode]
    if args.no_upload: command.append("--no-upload")
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
