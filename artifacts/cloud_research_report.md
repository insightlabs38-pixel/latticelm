# LatticeLM cloud research report — architecture tournament milestone

## Status and integrity boundary

This report now includes the real-data tokenizer, Round 1 (131,072 tokens),
Round 2 (1,048,576 tokens), and persisted Round 3 (3,145,728 tokens) family
comparison. It does not claim full-corpus training, HPO, official benchmark
results, or Triton training speedups.
All runs were sequential with two PyTorch threads because the cgroup provides a
two-CPU quota. Tiny Shakespeare results remain in the historical ledger but are
not used below.

## 1. Hardware/environment

See `cloud_system_report.md` and `cloud_environment.json`. The effective
allocation is 16 GiB RAM and two CPU-equivalents over a three-vCPU Intel Xeon
8370C cpuset. AVX-512 is exposed; VNNI, BF16 and AMX are not.

## 2. Dataset and tokenizer

- Corpus: official `BabyLM-community/BabyLM-2026-Strict-Small`, pinned to
  Hugging Face revision `c92ab16b4f08858304b0815706065b3354d8fc0a`.
- The six official training files and SHA-256 hashes are recorded in
  `dataset_manifest.json`. No BabyLM/GIBC evaluation data was downloaded.
- A deterministic, source-stratified holdout takes the final 2% of lines from
  each source: 9,807,491 whitespace-delimited training words and 192,509
  validation words.
- The real byte-level BPE has 4,096 populated tokens (3,836 learned merges),
  rather than reserved unused IDs. Training took 25.05 s. On the deterministic
  one-million-character sample it measured 3.433 characters/token and 1.585
  tokens/word.
- Engram identity normalization follows the official demo's intent: NFKC,
  NFD/strip accents, lowercase and whitespace collapse. This compresses 4,096
  model tokens to 3,216 memory identities. The complete map is stored beside
  the tokenizer.

## 3. Matched dense baseline

The dense reference is the requested 256-wide, eight-layer causal Transformer:
four query heads, one KV head, 768-wide SwiGLU, RoPE, RMSNorm, tied 4K
embedding/head, context 128, FP32, batch 8, AdamW at 3e-4, and seed 1337. It has
7,082,240 trainable parameters. At 1,048,576 tokens it reached validation loss
4.106869 (perplexity 60.7562) in 475.33 cumulative training seconds at 2,206
tokens/s.

## 4. Mini-Engram

The implementation was derived from official DeepSeek Engram repository commit
`fb7f84a21f91223715394a33a1dc24bbfb7f788e` and its Apache-2.0 demo. It retains:

- compressed/normalized memory-token identity;
- causal bigrams and trigrams;
- two independently seeded hash heads per order and concatenated head values;
- a memory-key/hidden-query RMS-normalized similarity gate with the reference
  signed-square-root transform and sigmoid;
- projected values plus a causal depthwise short convolution (kernel 4,
  dilation 3), inserted once before backbone layer 1;
- a separate 0.3x memory learning rate and 0.1 memory dropout.

This is explicitly a **tiny adaptation**, not a scale-faithful reproduction of
the large model. Memory budgets tested were 558,592, 1,082,880, 2,131,456 and
4,228,608 parameters. At Round 1 the 2.13M memory was best by validation loss,
but the entire range was effectively tied; larger memory was not automatically
better. The 2.13M candidate advanced. At 1,048,576 tokens it reached 4.105738
(ppl 60.6875) in 498.69 s: only 0.001131 loss better than dense at equal tokens,
while dense remained faster.

## 5. Lattice extensions

Collision-aware heavy hitters were not promoted from the old Shakespeare
analysis. They must be rebuilt from training-only BabyLM n-gram counts after a
memory benefit becomes statistically meaningful. This avoids spending CPU on
an extension before establishing that the base mechanism helps.

## 6. Co4 / IHMS causal adaptation

The authoritative implementation inspected was
`ARIA-Funded-TREND/IHMS` commit
`e25e9119173d59fdb9e4548ef640734dbd19f067` (CC BY-NC 4.0). Its released
operator targets vision/RL and uses non-causal latent/patch aggregation, so it
cannot be copied directly into an autoregressive LM. `CausalCo4Attention`
retains its exact awake MOD law
`ReLU6(R² + 2R + C(1 + |R|))`, learned normal latent receptive streams, and
Q/K/V modulation before readout; it replaces non-causal patch top-k readout
with RoPE causal SDPA. It is therefore named a **causal adaptation**, not a
faithful reproduction of the vision architecture.

The previous guessed `Co4InspiredExperimental` was not run and supplies no
evidence here. The new 7,874,816-parameter adaptation reached loss 4.034405
(ppl 56.5093) at 1,048,576 tokens, a 0.072464 loss improvement over dense at
equal tokens, though it was 13.6% slower.

## 7. Architecture tournament

### Round 1 — equal 131,072 training tokens

| family / memory | total params | memory params | wall s | tok/s | train loss | val loss | val ppl |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 7,082,240 | 0 | 59.62 | 2,198.6 | 4.7671 | 5.223749 | 185.629 |
| Mini-Engram 0.56M | 7,640,832 | 558,592 | 62.72 | 2,089.7 | 4.7672 | 5.223576 | 185.597 |
| Mini-Engram 1.08M | 8,165,120 | 1,082,880 | 63.55 | 2,062.7 | 4.7676 | 5.222782 | 185.449 |
| Mini-Engram 2.13M | 9,213,696 | 2,131,456 | 62.76 | 2,088.6 | 4.7698 | **5.222599** | 185.416 |
| Mini-Engram 4.23M | 11,310,848 | 4,228,608 | 66.09 | 1,983.1 | 4.7674 | 5.223582 | 185.598 |
| Co4 causal adaptation | 7,874,816 | 0 | 73.83 | 1,775.3 | 4.6650 | **5.127342** | **168.568** |

Dense, Co4, and the best early Mini-Engram budget advanced. The other memory
capacities were pruned because their tiny differences did not justify nearly
identical candidates in Round 2.

### Round 2 — equal 1,048,576 training tokens

| family | total params | wall s | tok/s | train loss | val loss | val ppl |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 7,082,240 | **475.33** | **2,206.0** | 4.0765 | 4.106869 | 60.7562 |
| Mini-Engram 2.13M | 9,213,696 | 498.69 | 2,102.7 | 4.0738 | 4.105738 | 60.6875 |
| Co4 causal adaptation | 7,874,816 | 539.96 | 1,942.0 | **3.9641** | **4.034405** | **56.5093** |

The machine-readable append-only rows are in `results.jsonl`; the requested
tournament projection is `architecture_tournament.csv`. Early rows recorded
resident memory at completion rather than OS high-water RSS; this limitation is
preserved rather than relabeled. Future rows use `ru_maxrss`.

### Equal wall-clock comparison

At the shortest finalist completion time (475.33 s), use the nearest recorded
validation point without extrapolation:

| family | recorded wall s | tokens | val loss |
|---|---:|---:|---:|
| Dense | 475.33 | 1,048,576 | 4.106869 |
| Mini-Engram 2.13M | 468.31 | 983,040 | 4.150612 |
| Co4 causal adaptation | 474.72 | 917,504 | **4.103226** |

Co4 therefore leads both equal-token quality and the measured equal-wall point,
but its equal-wall advantage over dense is only 0.003643 loss and needs another
seed/Round 3 confirmation. Mini-Engram's equal-token gain disappears under the
equal-wall control at this stage.

### Round 3 — equal 3,145,728 training tokens

Round 3 used the same data, tokenizer, seed, optimizer settings, two-thread
allocation, and token count. Both runs enabled the same one-million-token remote
`latest`/`best` persistence cadence and final named checkpoint upload, so upload
overhead is included in both wall-clock totals.

| family | total params | wall s | tok/s | train loss | val loss | val ppl |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 7,082,240 | 1,539.03 | 2,044.0 | 3.6266 | 3.609044 | 36.9308 |
| Co4 causal adaptation | 7,874,816 | **1,512.97** | **2,079.2** | **3.5901** | **3.545640** | **34.6619** |

Co4 improves validation loss by 0.063404 at equal tokens and is also 26.06 s
faster in the complete persisted-run wall measurement. It therefore remains the
leading architecture family after successive halving. The generated tournament
rows accidentally retained the generic text “Round 2” in their `notes` field;
the experiment IDs, token counts, configs, and this report correctly identify
them as Round 3, and the writer is corrected for subsequent runs.

Both final checkpoints were exported with exact safetensors tensor/logit
round-trip verification and persisted remotely as named scientific checkpoints.
The repository-global `best/` slot was restored to Co4 after detecting and
fixing an initial per-experiment-best policy error; `latest/` contains the final
Dense recovery checkpoint, while `experiments/co4-round3/` and
`experiments/dense-round3/` preserve both controls.

## 8. HPO

Deferred as requested. All families used the same sensible optimizer settings;
only reference-motivated memory regularization differed.

## 9. Triton-CPU

Deferred until a winner is confirmed. No Triton speedup is claimed.

## 10. Ranking and next successive-halving decision

1. **Best validation quality:** Co4 causal adaptation (Round 3 loss 3.545640).
2. **Best measured quality per wall-clock time:** Co4 causal adaptation; it is
   both better and faster in the complete persisted Round 3 runs.
3. **Most promising for scaling:** Co4 causal adaptation, because its
   equal-token gap widened to 0.072464 by 1M tokens rather than vanishing.
4. **Highest-risk architecture worth investigating:** Co4 plus 0.5–2M Engram
   memory. This is now justified by Co4 competitiveness, but should follow a
   second-seed Co4 confirmation rather than be assumed beneficial.

The next decision is an independent-seed confirmation, followed by the
high-risk Co4-plus-small-Engram branch now that plain Co4 is competitive.
Profiling and Triton work remain downstream of that comparison.

## 11. Official benchmark readiness/results

No official benchmark task was evaluated, preventing test-set-driven pruning.

## 12. Negative results

- Mini-Engram capacities from 0.56M through 4.23M were indistinguishable at
  131K tokens; “more memory” had no monotonic benefit.
- At 1M equal tokens, 2.13M Mini-Engram improved loss by only 0.001131 and lost
  at equal wall time.
- Co4 is slower and its causal readout necessarily deviates from the released
  non-causal vision implementation.

## 13. Recommended i5-1035G1 follow-up

Do not optimize memory gather first. If Co4 survives Round 3/multiple seeds,
profile its MOD elementwise path, QKV projection and SDPA on the laptop. Compare
the same dense checkpoint path, verify the laptop's actual ISA, and only then
choose a Triton fusion target.
