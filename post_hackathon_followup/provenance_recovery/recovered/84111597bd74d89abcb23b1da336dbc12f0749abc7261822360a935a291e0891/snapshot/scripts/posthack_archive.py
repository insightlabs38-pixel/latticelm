"""Upload an explicit allowlist of research evidence to the private HF model repo."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token

from posthack_campaign import RESEARCH, event, sha256, utc

DEFAULT_REPO = "insightlabs38-pixel/LatticeLM-research"
REMOTE_PREFIX = "post-hackathon/2026-10-01"


def upload(paths: list[Path], experiment_id: str) -> list[dict]:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face credentials unavailable")
    repo = os.environ.get("LATTICELM_HF_REPO", DEFAULT_REPO)
    api = HfApi(token=token)
    info = api.repo_info(repo, repo_type="model")
    if not info.private:
        raise RuntimeError("research upload requires a private Hugging Face repository")
    outcomes = []
    for path in paths:
        path = path.resolve()
        relative = path.relative_to(RESEARCH.resolve())
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"not a regular research file: {path}")
        if path.name.endswith((".tmp", ".lock", ".pid")) or path.name.startswith(".env"):
            raise ValueError(f"transient or secret file refused: {path}")
        remote = f"{REMOTE_PREFIX}/{relative.as_posix()}"
        digest = sha256(path)
        commit = api.upload_file(path_or_fileobj=str(path), path_in_repo=remote, repo_id=repo,
                                 repo_type="model", commit_message=f"Archive post-hackathon {experiment_id}")
        sibling = next(item for item in api.repo_info(repo, repo_type="model", files_metadata=True).siblings
                       if item.rfilename == remote)
        lfs = getattr(sibling, "lfs", None)
        remote_digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if sibling.size != path.stat().st_size or (remote_digest and remote_digest != digest):
            raise RuntimeError(f"remote verification failed for {relative}")
        outcome = {"at": utc(), "path": str(relative), "remote": remote, "repo": repo,
                   "sha256": digest, "size_bytes": path.stat().st_size,
                   "remote_lfs_sha256": remote_digest, "revision": commit.oid, "verified": True}
        outcomes.append(outcome)
        event(experiment_id, "HF_ARCHIVE_VERIFIED", **outcome)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(upload(args.paths, args.experiment_id), indent=2), flush=True)


if __name__ == "__main__":
    main()
