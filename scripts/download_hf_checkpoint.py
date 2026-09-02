"""Download, reconstruct, and smoke-test a private LatticeLM model snapshot."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_model

from latticelm.config import LatticeConfig
from latticelm.hf_storage import sha256
from latticelm.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("LATTICELM_HF_REPO"))
    parser.add_argument("--path", default="tests/smoke")
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.repo:
        parser.error("--repo or LATTICELM_HF_REPO is required")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    root = Path(snapshot_download(args.repo, repo_type="model", revision=args.revision, token=token,
                                  allow_patterns=[f"{args.path}/**"], local_dir=args.output)) / args.path
    large_manifest = root / "large_files.json"
    if large_manifest.exists():
        for name, metadata in json.loads(large_manifest.read_text()).items():
            destination = root / name
            with destination.open("wb") as output:
                for part in metadata["parts"]:
                    output.write(base64.b64decode((root / part).read_bytes()))
            if destination.stat().st_size != metadata["size"] or sha256(destination) != metadata["sha256"]:
                raise RuntimeError(f"chunked artifact reconstruction failed: {name}")
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha256(root / name) != expected:
            raise RuntimeError(f"downloaded checksum mismatch: {name}")
    config_data = json.loads((root / "config.json").read_text())
    if config_data.get("memory_token_map_path"):
        config_data["memory_token_map_path"] = str(root / config_data["memory_token_map_path"])
    model = build_model(LatticeConfig(**config_data))
    load_model(model, root / "model.safetensors", strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.arange(8).view(1, -1).remainder(model.config.vocab_size))[0]
    manifest = json.loads((root / "training_manifest.json").read_text())
    print(json.dumps({"loaded": True, "shape": list(logits.shape), "experiment": manifest["experiment_id"],
                      "revision": args.revision}, indent=2))


if __name__ == "__main__":
    main()
