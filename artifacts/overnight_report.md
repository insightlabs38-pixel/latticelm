# LatticeLM overnight report

## Environment

| field | observed value |
| --- | --- |
| OS | Windows 10 10.0.19045 |
| Python / PyTorch | 3.12.10 / 2.13.0+cpu |
| logical CPU count / PyTorch capability | 8 / AVX512 |
| RAM | 12,614,754,304 bytes |
| oneDNN/MKLDNN | available |
| Triton | not importable |

The training commands used four PyTorch threads for the public-corpus runs.
`environment.json` is the machine-readable source of truth.

## Implemented reference

- Dense CPU Transformer: tied token/output embeddings, RMSNorm, SwiGLU, RoPE,
  causal GQA with one KV head.
- Lattice conditional memory: three trainable suffix n-gram tables, explicit
  stable 64-bit hashing, 64-to-model projection and a learned sigmoid gate
  before layer 0.
- Safety net: causality, hash determinism, nonzero memory gradients, tied
  parameter accounting, residual-RMSNorm correctness, checkpoint round trip.
- Reproducibility: JSON configurations, seed 1337, append-only results/logs,
  checkpoint configs, resume script and environment capture.

## Results

The public comparison uses the same 100K-character Tiny Shakespeare subset,
tokenizer, seed (1337), batch size (8), context (128), optimizer and 256x8
Transformer backbone. It has only 12,288 training tokens, so it is a pipeline
validation and not evidence of a capability gain.

| experiment | params | memory params | tokens | wall s | tok/s | val loss | val ppl | RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_128_public_12_matched | 7,082,240 | 0 | 12,288 | 17.9497 | 684.581 | 6.153730 | 470.46885 | 468,594,688 |
| lattice_64k_128_public_12_fair | 19,747,584 | 12,665,344 | 12,288 | 21.9032 | 561.013 | 6.152784 | 470.02428 | 711,884,800 |

The model construction order was explicitly arranged so the common backbone
starts with identical seeded weights. At equal tokens, Lattice's validation
loss was lower by 0.000946 (not meaningful at 12,288 tokens), and it was 22.0%
slower. There is no exact equal-wall quality measurement yet, so these
preliminary results do **not** support a conditional-memory or CPU-hour
advantage. The 1,536-token synthetic smoke rows are retained in `results.csv`
strictly for correctness smoke tests.

Earlier public rows without the `_fair` suffix are retained in the append-only
ledger for audit. They predate the initialization-order fairness test and are
not used for the comparison table or conclusion above.

The best current public checkpoint by the recorded validation loss is
`artifacts/checkpoints/lattice_64k_128_public_12_fair_step12.pt`; its `.json`
config is versioned, while the binary checkpoint remains locally available and
is ignored to avoid committing large generated binaries.

## Triton-CPU

The environment probe and PyTorch references for residual-add/RMSNorm and
three-table conditional-memory gather are included. Triton is not installed in
the inspected environment, so no Triton kernel or training acceleration is
claimed. `kernel_benchmarks.jsonl` records the exact blocker and reference
latencies across 1, 2, 4 and 8 PyTorch threads. The reference memory gather was
fastest at 4 threads in this sample (0.45000 ms median); the reference
residual/RMSNorm was fastest at 8 (1.04815 ms median). These are forward-only
reference figures, not a custom-kernel training claim.

## Negative / deferred results

- Initial naive BPE construction with a larger merge budget was too slow for
  this CPU run. The implementation therefore uses a deterministic 20K-byte
  corpus sample and 128 learned merges while reserving the configured 4K model
  vocabulary IDs; this tradeoff is documented in code and README.
- No recurrent/local hybrid, Co4 reproduction, or INT8/VNNI claim is included:
  these are intentionally deferred until the main dense-versus-memory evidence
  is healthy.
- No custom Triton backward exists; forward-only work would not constitute a
  training speedup.

## Next experiment

Run `scripts/run_overnight.ps1` to extend the matched runs to 600 steps, add
intermediate equal-wall validation checkpoints, then optimize n-gram ID
precomputation before considering a fused CPU kernel.
