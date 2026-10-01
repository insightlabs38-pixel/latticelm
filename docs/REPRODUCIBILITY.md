# Reproducibility

## Environment

The package requires Python 3.10 or newer. The final run used PyTorch on CPU, 16 threads, seed 1337, and a fixed tokenizer and DATA-D-v4 manifest. Exact environment and data identity are not implied by the package version alone; use the recorded SHA-256 values in [`../artifacts/manifests/final_production_provenance.json`](../artifacts/manifests/final_production_provenance.json) and the evaluation report.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[evaluation]' pytest
```

## Inputs and commands

The DATA-D-v4 builder creates the source manifest and token shards locally. The certified data and tokenizer used by the final run are not included in Git. The builder, trainer, and evaluator command interfaces were checked with `--help` in this checkout:

```bash
python scripts/build_data_d_v4.py --help
python scripts/train_final_production.py --help
python scripts/evaluate_wikitext103.py --help
python scripts/evaluate_gibc.py --help
python scripts/run_posttraining_master.py --help
```

With the matching input manifest and tokenizer available, run the final recipe using the command template in the README. Evaluation scripts accept the trained checkpoint and tokenizer and write JSON outputs to a caller-selected path. The documented interfaces were verified; full training, dataset construction, and full GIBC evaluation were not repeated during repository reconciliation.

Post-training proxy analysis requires a candidate checkpoint, parent checkpoint, tokenizer, and fixed example manifest. It is used as a screening measurement. Full GIBC evaluation is required before interpreting a candidate as a broad capability improvement. The selection evidence documents which candidates received that full evaluation.

## Determinism and integrity

Training seeds and data sampling policy are stored in the final recipe and run metadata. The final artifact manifest records the source code commit used for production, selected recipe identity, final checkpoint SHA-256, tokenizer SHA-256, and DATA-D-v4 manifest SHA-256. Use these hashes to detect a different input or model. The official metrics refer to the corrected joined causal-boundary evaluator described in the production results.

## Expected outputs and versioning

Training and evaluation create checkpoints, optimizer state, run-state JSON, lock/PID files, logs, token shards, and temporary results under `artifacts/`. They are intentionally excluded from Git. Compact results, reports, manifests, and the validation curve are retained under `artifacts/`; [`../artifacts/README.md`](../artifacts/README.md) explains that policy. Do not infer that an omitted raw checkpoint is reproducible from source alone: exact replay requires access to the data and tokenizer identified by the manifests.

To run the focused regression and systems tests:

```bash
python -m pytest -q tests/test_systems_data_master.py tests/test_posttraining_followup.py tests/test_production_eval_companion.py tests/test_rlvr_watchdog_regressions.py
```

The video source and build workflow are documented in [`../video/README.md`](../video/README.md). Generated renders and QA material are kept out of Git; the seven final submission stills are retained.
