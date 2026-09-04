# Phase 7C — GCP C4A bring-up and canonical 10M run

## Outcome

The fresh LatticeLM-Co4-S canonical GCP lineage reached exactly **10,000,000 loss-bearing training tokens** and stopped. It consumed exactly 7,500,000 BabyLM tokens and 2,500,000 decontaminated FineWeb-Edu tokens. No 25M/50M continuation was started, no architecture or optimizer setting changed, and no prior trained weights were used.

## Environment and validation

- GCP machine: `c4a-standard-16` Spot, `europe-west4-c`
- Architecture: AArch64 / ARM64, ARM Neoverse-V2
- Visible/usable cores: 16 independent guest-visible cores, one thread per core, one NUMA node; no SMT exposed
- RAM: 62 GiB; swap: none
- Storage: 100 GB Hyperdisk Balanced boot disk mounted at `/`; no separate persistent disk was attached. Checkpoints were therefore also mirrored to Hugging Face. Guest metadata did not expose the boot disk auto-delete policy.
- OS/kernel: Ubuntu 24.04.4 LTS / 7.0.0-1011-gcp
- Python/PyTorch: 3.12.3 / 2.14.0+cpu, native AArch64 build with oneDNN and OpenMP

All 22 tests passed. Source compilation and `git diff --check` passed. The exact Phase 7A binaries, tokenizer, decontamination record, source revisions, validation set, and their hashes were reconstructed and verified. The DATA-C sampler passed independent-process determinism and serialized-resume next-batch hash tests. HF upload/download smoke verification passed.

## Runtime selection

The exact workload sweep used real DATA-C batches and complete forward/loss/backward/clipping/AdamW steps. Twelve threads at batch eight was selected for stability: its longer run measured 7,688.64 tokens/s with 0.133183 s median step time. Sixteen threads had a faster 0.130057 s median but much worse variance (0.02925 s standard deviation and 0.03901 s IQR). Batch 16/32 throughput results were retained but rejected because adopting them would alter the frozen effective batch.

- Selected PyTorch intra-op threads: 12
- Selected inter-op threads: 1
- Selected microbatch: 8 sequences / 1,024 tokens
- Selected gradient accumulation: 1
- Selected effective batch: 8 sequences / 1,024 tokens
- Selected runtime: eager PyTorch

`torch.compile` took 16.554 s to initialize, improved steady-state median throughput by only about 0.45%, and produced a maximum one-step parameter difference of 3.353e-4 in the conservative equivalence check. It was therefore rejected. Profiling showed matrix multiplication dominant (46.96% self CPU), with AdamW 8.04% total CPU and CPU flash-SDPA forward/backward 2.66%/4.89%.

## Performance and estimated cost

Canonical training-only throughput was **7,752.82 tokens/s** over 1289.854 s. That projects to **3.583 hours per 100M tokens** and **$0.3436 per 100M tokens**, using the user's console-derived estimate of $70/730 h = $0.0959/h. This is an approximate normalized compute estimate, not an audited billing rate and does not include setup, evaluation, downtime, storage, networking, or idle VM time. There were zero observed preemptions and zero recorded infrastructure downtime.

## Triton-CPU / AArch64

Status: **WORKING for minimal correctness; partial for production training integration.** Official `triton-lang/triton-cpu` commit `9a3dd8096b3c5b89a6dfeba012221f3fed450eb0` built on AArch64, and the bounded 257-element FP32 vector-add smoke test passed exactly. Missing LLD and a missing SLEEF submodule caused two recorded failed attempts before the successful build. The broad upstream autotuning tutorial was stopped after three minutes. A Co4 MOD kernel was deliberately deferred because a correct custom backward and full-step speedup proof were not immediately available. Best current MOD implementation: eager PyTorch reference.

## Canonical 10M result

| Metric | Result |
|---|---:|
| Parameters | 8,923,392 |
| Common validation loss | 3.626181 |
| Common validation perplexity | 37.56908 |
| BabyLM validation loss | 3.366002 |
| FineWeb-Edu validation loss | 4.204997 |
| WikiText-103 perplexity | 145.97471 |
| HellaSwag accuracy | 0.258016 |
| ARC-Easy accuracy | 0.277778 |
| PIQA accuracy | 0.524483 |
| WinoGrande accuracy | 0.500395 |

The GIBC tasks used lm-eval 0.4.13, zero-shot evaluation, batch eight, and the frozen tokenizer. Normalized scores and standard errors are preserved in `gcp_final_10m_gibc.csv`; raw output is retained separately. WikiText-103 scored 411,412 tokens from `wikitext-103-raw-v1`.

## Learning progression and interpretation

| Tokens | Common loss | Common perplexity |
|---:|---:|---:|
| 1,000,448 | 4.558660 | 95.45546 |
| 3,000,320 | 3.974787 | 53.23875 |
| 5,000,192 | 3.805152 | 44.93208 |
| 7,499,776 | 3.675049 | 39.45058 |
| 10,000,000 | 3.626181 | 37.56908 |

Interval improvements per million tokens were:

- 1.000M→3.000M: loss 0.291955/M, perplexity 21.1097/M
- 3.000M→5.000M: loss 0.084823/M, perplexity 4.1536/M
- 5.000M→7.500M: loss 0.052050/M, perplexity 2.1930/M
- 7.500M→10.000M: loss 0.019545/M, perplexity 0.7525/M


The recent common-loss slope remains negative (improving), but it has slowed materially. BabyLM and FineWeb-Edu validation both continued improving through 10M. Classification: **SLOWING, NOT NEAR PLATEAU**; the model remains data-limited at 10M. The recommended next deliberate target is **25M**, with a new authorization and another slope review; no continuation has been launched.

The Phase 7A DATA-C research model reached 3.957705 common loss / 52.3371 perplexity at 3,145,728 tokens. The new lineage reached 3.974787 / 53.2388 at its nearby 3,000,320-token checkpoint and 3.626181 / 37.5691 at 10M. This is learning/data-scaling progression across different seeds and endpoints, not a controlled same-token claim. The Phase 6 Co4-S research checkpoint's 3.523641 BabyLM-only validation loss used a different validation regime and cannot be ranked directly against the balanced common-validation numbers.

## Persistence and lineage

The public immutable bundle is `experiments/final-gcp-10m/`; private resumable state remains at `recovery/final-gcp-10m/`. The final bundle was remotely size-verified at Hugging Face revision `0ab7f9d033d5b1c50ac0f21d688371951879b5e0`. The final training checkpoint SHA-256 is `5a9f33ff95c0c0ad4f103abfe4de1db13dc52a82cdf8f4b3c8e6f1b746d16fc7` and tokenizer SHA-256 is `4f313ebc481a77e8ad2179cf2d7a3836b28d50773ef9f62ef08831bf076637e5`. The canonical manifest was frozen before training, at source code commit `f74e932`; manifest commit `c55437e` records that freeze. The rolling local latest/previous/fallback recovery states and periodic remote recovery uploads were maintained. The existing global `best/` was not replaced because its recorded validation loss is lower under its own historical validation regime.
