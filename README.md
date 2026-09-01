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
