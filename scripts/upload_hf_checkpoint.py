"""Upload and remotely verify one exported checkpoint bundle."""
from __future__ import annotations

import argparse
import json
import os

from latticelm.hf_storage import upload_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--repo", default=os.environ.get("LATTICELM_HF_REPO"))
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not args.repo:
        parser.error("--repo or LATTICELM_HF_REPO is required")
    if not token:
        parser.error("HF_TOKEN or HUGGING_FACE_HUB_TOKEN with write permission is required")
    revision = upload_checkpoint(args.directory, args.repo, args.path, token)
    print(json.dumps({"repo": args.repo, "path": args.path, "verified": True, "revision": revision}, indent=2))


if __name__ == "__main__":
    main()
