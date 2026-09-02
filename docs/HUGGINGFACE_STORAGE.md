# Hugging Face checkpoint storage

LatticeLM uses two repositories with deliberately different responsibilities.

## Source Git repository

This repository contains source code, configs, tests, reports, metrics, and
architecture-tournament results. Generated model weights and transient local
checkpoints remain ignored.

## Private Hugging Face model repository

The Hub model repository contains selected safetensors weights, tokenizer and
memory-identity metadata, model/training manifests, and limited private resume
state. It must not contain access tokens, `.env` files, datasets, caches,
virtual environments, build trees, or redundant failed-trial checkpoints.

Set secrets and repository identity only in the process environment:

```bash
export HF_TOKEN=...                         # write-scoped secret; never commit
export LATTICELM_HF_REPO=user/LatticeLM-research
```

Export locally, which performs a strict fresh-model tensor and logits round trip:

```bash
PYTHONPATH=src python scripts/export_release_model.py \
  --checkpoint artifacts/checkpoints/EXPERIMENT_stepN.pt \
  --output artifacts/hf_staging/latest \
  --experiment EXPERIMENT --role latest \
  --tokenizer artifacts/tokenizers/babylm_2026_4k.json \
  --tokenizer-report artifacts/tokenizers/babylm_2026_4k.report.json \
  --metrics artifacts/hf_staging/metrics.json
```

Upload and verify remote file sizes:

```bash
PYTHONPATH=src python scripts/upload_hf_checkpoint.py \
  --directory artifacts/hf_staging/latest --path latest
```

Download, reconstruct, load safetensors, and run deterministic inference:

```bash
PYTHONPATH=src python scripts/download_hf_checkpoint.py \
  --path latest --output artifacts/hf_download
```

`best` and `latest` are replaceable semantic slots. Only important controls and
finalists belong under `experiments/`; ordinary losing trials remain represented
by the source repository's append-only logs and metrics.
