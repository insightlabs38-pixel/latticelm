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

The primary comparison uses the same 100K-character Tiny Shakespeare subset,
tokenizer, seed (1337), batch size (8), context (128), optimizer and 256x8
Transformer backbone. The full paired run trained 614,400 tokens each.

| experiment | params | memory params | tokens | wall s | tok/s | val loss | val ppl | RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_128_public_12_matched | 7,082,240 | 0 | 12,288 | 17.9497 | 684.581 | 6.153730 | 470.46885 | 468,594,688 |
| lattice_64k_128_public_12_fair | 19,747,584 | 12,665,344 | 12,288 | 21.9032 | 561.013 | 6.152784 | 470.02428 | 711,884,800 |
| dense_128 | 7,082,240 | 0 | 614,400 | 530.3726 | 1,158.431 | 2.251499 | 9.50197 | 514,613,248 |
| lattice_64k_128 | 19,747,584 | 12,665,344 | 614,400 | 612.8580 | 1,002.516 | 2.749724 | 15.63832 | 558,505,984 |

The model construction order was explicitly arranged so the common backbone
starts with identical seeded weights. The full equal-token result rejects the
current hypothesis: Lattice was 13.5% slower and its validation loss was 0.498225
higher (perplexity 15.64 vs 9.50), even though its final training loss was lower
(0.864 vs 1.756). Thus it learned the training stream more aggressively but
generalized worse on this validation split. At roughly the dense run's wall
time, Lattice had not finished, so there is no plausible equal-wall advantage.
The 1,536-token synthetic smoke rows are retained in `results.csv` strictly for
correctness smoke tests.

Earlier public rows without the `_fair` suffix are retained in the append-only
ledger for audit. They predate the initialization-order fairness test and are
not used for the comparison table or conclusion above.

The best current public checkpoint by recorded validation loss is
`artifacts/checkpoints/dense_128_step600.pt`; its `.json` config is versioned,
while the binary checkpoint remains locally available and is ignored to avoid
committing large generated binaries.

## Triton-CPU

The environment probe and PyTorch references for residual-add/RMSNorm and
three-table conditional-memory gather are included. Triton is not installed in
the inspected environment, so no Triton kernel or training acceleration is
claimed. `kernel_benchmarks.jsonl` records the exact blocker and reference
latencies across 1, 2, 4 and 8 PyTorch threads. In the final sweep, reference
memory gather was fastest at 4 threads (0.22710 ms median); residual/RMSNorm
was fastest at 2 (0.32610 ms median). These are forward-only
reference figures, not a custom-kernel training claim.

## Follow-up branches attempted

- Vectorized causal n-gram addressing: `lattice_64k_128_vectorized_12` reached
  808.802 tokens/sec and 6.150152 validation loss, a major throughput recovery
  from the original Python-loop reference (561 tokens/sec); the complete run
  sustained 1,002.516 tokens/sec.
- 128K memory: `lattice_128k_128_12` has 32,330,496 parameters (25,248,256
  memory parameters), remained within cap, and reached 6.146013 validation
  loss at 504.393 tokens/sec.
- Hybrid: a causal PyTorch mLSTM/mLSTM/local-attention/mLSTM topology was
  implemented and tested. It trained stably, but was slower (281.637 tokens/sec)
  and worse (6.199381 validation loss). Its recurrence follows the matrix-memory
  update/retrieval equations in the [BLaLM paper](https://aclanthology.org/2025.babylm-main.14.pdf).
- Co4: the actual [BabyLM Co4 paper](https://aclanthology.org/2025.babylm-main.24.pdf)
  was read. It does not publish the cited MOD transfer-function equations needed
  for a faithful reproduction, so `Co4InspiredExperimental` was implemented and
  explicitly named as non-faithful. Its causal O(T×24) run was stable (803.157
  tokens/sec) but had much worse validation loss (6.548297).
- Collision-aware memory: exact-heavy-hitter plus hashed-tail preprocessing was
  implemented and measured. The 65,536-slot tails had 48 bigram, 1,763 trigram,
  and 7,170 fourgram unique-key collisions; see `collision_analysis.json`.
- INT8: rowwise symmetric memory quantization and oneDNN dynamic-INT8 projection
  were numerically accurate but slower (2.14175 vs 0.72530 ms gather; 0.75030 vs
  0.43900 ms projection). VNNI is not claimed: no installed compiler or
  disassembler could inspect emitted instructions. See `int8_benchmark.json`.
- Triton-CPU: the official source repository at `b124250` was cloned; runtime
  dependencies were installed in an isolated environment and the source build
  reached CMake. It is blocked by missing C/C++ compiler tooling after a Windows
  directory-symlink privilege failure; `triton_attempt.json` contains the exact
  attempt. Guarded forward K1/K2 kernel source and fallback tests are present,
  but no unexecuted kernel speed claim is made. [Official README](https://github.com/triton-lang/triton-cpu/blob/main/README.md)

## Negative / deferred results

- Initial naive BPE construction with a larger merge budget was too slow for
  this CPU run. The implementation therefore uses a deterministic 20K-byte
  corpus sample and 128 learned merges while reserving the configured 4K model
  vocabulary IDs; this tradeoff is documented in code and README.
- No faithful Co4 reproduction is claimed: its needed modulation equations are
  delegated to external work. The separate inspired experiment above is kept
  distinct from Co4 claims.
- No custom Triton backward exists; forward-only work would not constitute a
  training speedup.

## Next experiment

The concrete next experiment is memory regularization / smaller tables with
intermediate equal-wall validation checkpoints: the full run indicates
conditional memory is overfitting this corpus. Reattempt Triton only after
installing Microsoft C++ Build Tools or clang and enabling a directory-symlink
capability, then re-run K1/K2 numerical and forward benchmarks before any
training-speed claim.
