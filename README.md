# LatticeLM

LatticeLM is a reproducible CPU-first language-model experiment that compares a
compact causal Transformer with an identical backbone augmented by trainable,
causal hashed n-gram memory. It is trained from random initialization and keeps
all trainable parameters below the 50M Track 01 limit.

## Quick start

```powershell
$env:PYTHONPATH = 'src'
python -m latticelm.train --config configs/dense_smoke.json --experiment dense_smoke
python -m latticelm.train --config configs/lattice_smoke.json --experiment lattice_64k_128
python -m pytest -q
```

The training command creates a deterministic byte-level BPE tokenizer from its permitted training text. To keep laptop setup bounded, it learns up to 128 merges from a fixed corpus sample and reserves the remaining IDs in the configured 4K vocabulary,
saves resumable checkpoints, appends machine-readable metrics to
`artifacts/results.jsonl` and `artifacts/results.csv`, and records the resolved
configuration beside each checkpoint. `scripts/run_overnight.ps1` runs matched
dense and Lattice experiments sequentially. Resume with `--resume PATH`.

## Design and scientific controls

The two primary configurations use the same tokenizer, data split, seed,
context, optimizer, and Transformer backbone. Lattice only adds memory:

`E2[h2] + E3[h3] + E4[h4] -> linear projection -> sigmoid gate -> residual`

The n-gram at input position `t` contains only tokens at or before `t`; it
cannot inspect the target token at `t + 1`. Hashes use an explicit 64-bit
integer mixing function, never Python's randomized `hash()`.

`kernels/` contains PyTorch reference implementations and a guarded Triton-CPU
probe. Triton is optional: the reference path remains the only training path
unless a future custom backward implementation is validated.

## Reproducibility

Use Python 3.12 and PyTorch CPU. The default seed is 1337. See
`EXPERIMENT_LATTICELM.md` for the experiment protocol and
`artifacts/overnight_report.md` for locally measured results and limitations.

## Research status through Phase 7E

Early same-token, low-data experiments favored smaller Co4 models on efficiency; that result is preserved rather than retroactively reinterpreted. In the later DATA-C regime, controlled scaling showed that Co4-L (15.95M parameters) became clearly superior to Co4-S in common-validation and WikiText language modeling by 25M tokens. Its 25M reasoning gains were mixed despite the strong WikiText improvement.

Phase 7E continued the valid Co4-L lineage to 50M and 100M tokens. Common-validation loss improved from 3.328604 at 25M to 3.208180 and 3.134722; WikiText-103 perplexity improved from 102.551 to 79.679 and 72.582. Reasoning improved on all four tracked tasks at 50M, then became mixed from 50M to 100M (small HellaSwag/ARC-Easy gains, PIQA/WinoGrande regressions). The FineWeb-Edu pool was expanded without repeating its old prefix, but the planned DATA-D-BROAD-v1 control was not implemented or trained. See `artifacts/phase7e_overnight_report.md` and `artifacts/phase7e_decision.md`.

Future capacity decisions therefore follow the measured data-rich regime, not the old 3M comparison. A ~24M model remains deferred until broader-data and bounded continued-pretraining tests distinguish data/optimization limits from capacity limits. Negative and invalid runs remain excluded explicitly, including the archived unpaired Phase 7D Co4-L attempt.

## Persistent model storage

Source code, configs, tests, reports, and metrics remain in this Git repository.
Selected safetensors weights, tokenizer/memory metadata, manifests, and private
resume state belong in a separate private Hugging Face **model** repository.
See `docs/HUGGINGFACE_STORAGE.md` for authenticated export, upload, download,
integrity verification, and deterministic inference commands. Credentials are
accepted only through the process environment and must never be committed.
