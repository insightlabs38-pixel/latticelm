"""Export a LatticeLM checkpoint to a verified safetensors model bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from latticelm.hf_storage import export_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--role", choices=("best", "latest", "named"), required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-report", required=True)
    parser.add_argument("--dataset-manifest", default="artifacts/dataset_manifest.json")
    parser.add_argument("--metrics", required=True, help="JSON file or inline JSON object")
    args = parser.parse_args()
    metrics = json.loads(Path(args.metrics).read_text() if Path(args.metrics).exists() else args.metrics)
    manifest = export_checkpoint(args.checkpoint, args.output, args.experiment, args.role,
                                 args.tokenizer, args.tokenizer_report, args.dataset_manifest, metrics)
    print(json.dumps({"output": args.output, "verified": True,
                      "weights_sha256": next(line.split()[0] for line in Path(args.output, "SHA256SUMS").read_text().splitlines() if line.endswith("model.safetensors")),
                      "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
