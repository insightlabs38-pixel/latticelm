# LatticeLM experiment protocol

The required comparison is a dense causal Transformer versus the same model
with one pre-backbone conditional-memory injection. Both runs must share the
tokenizer, corpus split, token budget, context length, data order and seed.

Run the safety checks first:

```powershell
$env:PYTHONPATH = 'src'
python -m pytest -q
python -m latticelm.train --config configs/dense_smoke.json --experiment dense_smoke
python -m latticelm.train --config configs/lattice_smoke.json --experiment lattice_smoke
```

Then run the matched benchmark:

```powershell
$env:PYTHONPATH = 'src'
./scripts/run_overnight.ps1
```

Results are appended, never silently overwritten. Each record includes loss,
perplexity, throughput, step time, parameter counts, memory counts, thread
settings, RSS, wall time and checkpoint location. Training aborts an individual
experiment on non-finite loss and leaves the latest checkpoint resumable.

The optional Triton-CPU path is deliberately a probe only until its numerical
correctness and backward path are independently validated. A forward benchmark
must not be interpreted as a training speedup.
