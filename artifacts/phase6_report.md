# Phase 6 — Co4-L confirmation, competition snapshot, and architecture freeze

## Executive decision

**SELECTED FINAL ARCHITECTURE: untied Co4-S causal adaptation (8,923,392 parameters), fixed context 128, pinned BabyLM 4K tokenizer, AdamW (learning rate 3e-4, weight decay 0.1, betas 0.9/0.999, gradient clipping 1.0), constant learning rate.**

**RUNNER-UP: untied Co4-L causal adaptation (15,949,760 parameters).** It won validation and WikiText-103, reproduced at 1M tokens, and beat the matched Dense model, but its 0.026276 validation-loss / 2.59% BabyLM perplexity improvement over Co4-S cost 7.026M parameters and 871.650 s (57.0%) more 3M-token training wall time. Its reasoning gains were tiny and mixed under normalized scoring. Co4-L therefore did not earn its capacity under the Phase 6 competition threshold.

If submission were tomorrow, **submit Co4-S**. This is classification **B: Co4-L is slightly better but a poor efficiency tradeoff**. Architecture is now frozen; changing it requires compelling new evidence.

## Integrity and execution

The repository, all Phase 5 artifacts/configs/logs, dataset/tokenizer manifests, persistence tooling, and Git state were inspected before edits. CPU-heavy runs were sequential under the effective two-CPU quota with two PyTorch threads. Training used the canonical token stream reconstructed from the checksum-pinned BabyLM corpus and pinned tokenizer. No pretrained weights, teachers, benchmark data in training, repeated benchmark HPO, or fabricated values were used.

The missing historical Co4-S checkpoint was scientifically necessary and was reconstructed with the pinned architecture, seed 1337, optimizer, dataset, tokenizer, ordering, context, and 3,145,728-token budget. Its 3.523641 validation loss differs from the historical 3.516811 by 0.006830; the new reproducible canonical-stream value is used for direct Phase 6 comparisons, while the prior result remains preserved.

## HPO validation

The short-run HPO winner **did not survive** at 1M tokens. Both trials used identical Co4-L architecture, seed, data ordering, batch 8, context 128, tokenizer, and evaluation points.

| tokens | default val loss | HPO val loss | default − HPO |
|---:|---:|---:|---:|
| 262,144 | 4.693476 | **4.626122** | +0.067354 |
| 524,288 | 4.359533 | **4.335184** | +0.024349 |
| 786,432 | **4.127844** | 4.183130 | −0.055286 |
| 1,048,576 | **3.990510** | 4.148820 | −0.158310 |

At 1M, default perplexity was 54.08245 versus 63.35921 for HPO. Wall time was 792.999 s versus 795.139 s and throughput 1,322.29 versus 1,318.73 tokens/s. Cosine decay closed too aggressively for this horizon. The default constant-3e-4 setup was retained; no narrow HPO follow-up was justified.

## Co4-L scaling result

The default 1M checkpoint was resumed to 3,145,728 tokens. The resume restored model, optimizer, PyTorch RNG, Python RNG, data-generator state, and cumulative wall time. Because the winning schedule is constant, extending `max_steps` introduced no scheduler-horizon discontinuity.

Final exact metrics:

- parameters: 15,949,760
- train tokens: 3,145,728
- train loss (last sampled batch): 3.535820
- validation loss: **3.497365**
- validation perplexity: **33.02830**
- cumulative wall time: **2,401.6371 s**
- throughput: **1,309.827 tokens/s**
- peak RSS: **1,912,020,992 bytes**

The raw curve is preserved in `artifacts/logs/phase6_co4_l_final_3m.jsonl`, including evaluations from 1.31M through 3.145M tokens.

## Co4-S vs Co4-L

| model | params | tokens | wall s | tok/s | val loss | val ppl | peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Co4-S | 8,923,392 | 3,145,728 | 1,529.9870 | **2,056.049** | 3.523641 | 33.90766 | **1,525,002,240** |
| Co4-L | 15,949,760 | 3,145,728 | 2,401.6371 | 1,309.827 | **3.497365** | **33.02830** | 1,912,020,992 |

Scaling gains and costs:

- absolute validation-loss gain: **0.026276**
- relative perplexity improvement: **2.5934%**
- additional parameters: **7,026,368** (+78.74% relative to Co4-S)
- additional wall time: **871.6501 s** (+56.97%)
- loss improvement per additional million parameters: **0.003740**
- perplexity improvement per additional million parameters: **0.12516**

Co4-L lies on the strict quality frontier, but not on a compelling competition quality/compute frontier: the quality gain is real yet small for the capacity and CPU cost. This supports classification B rather than automatic selection of the larger model.

## Dense ~16M control

A conventional untied 11-layer Dense Transformer with d_model 320, four query heads / one KV head, SwiGLU 960, RMSNorm, RoPE, context 128, and **15,582,400** trainable parameters was constructed without awkward geometry. At 1,048,576 tokens it reached validation loss **4.123672**, perplexity **61.78572**, 747.2814 s, 1,403.188 tokens/s, and 1,743,306,752-byte peak RSS.

At equal 1M tokens Co4-L reached 3.990510 / 54.08245, an absolute 0.133162 loss and 12.47% perplexity advantage. Co4-L was 5.76% slower by throughput and used 4.79% more parameters, but the quality lead was clear at every recorded equal-token checkpoint. Dense was therefore defensibly pruned at 1M rather than spending another 2M tokens. Equal-wall interpolation was not used to invent an unmeasured score; the raw curves support the same directional conclusion near their overlapping wall range. Co4 is architecturally superior to this sensible parameter-matched Dense control at the tested budget.

## Seed robustness

Co4-L seed 2026 reached validation loss **3.995800**, perplexity **54.36934**, 804.4464 s, and 1,303.475 tokens/s at 1,048,576 tokens. Seed 1337 reached 3.990510 / 54.08245 in 792.9993 s. The 0.005290 loss difference is small and the trajectories agree. Two seeds rule out an obvious lucky initialization; they do not establish universal robustness.

## GIBC snapshot

A single zero-shot snapshot used `lm-evaluation-harness` 0.4.13 for HellaSwag, ARC-Easy, PIQA, and WinoGrande. WikiText-103 used the same adapter semantics with `Salesforce/wikitext` `wikitext-103-raw-v1` test data. Exact hashes and both raw/normalized accuracies are in the CSV.

Co4-L versus Co4-S raw accuracy moved by +0.00070 HellaSwag, +0.00673 ARC-Easy, +0.00762 PIQA, and +0.00474 WinoGrande. HellaSwag and PIQA normalized accuracy moved slightly *against* Co4-L. These changes are small relative to standard errors and do not show a decisive reasoning transfer. WikiText-103 perplexity improved from 263.307 to 245.850 (6.63%), showing transfer in general language modeling. Dense-16M at its disclosed 1M budget was validation-dominated and had WikiText-103 perplexity 594.718; its reasoning values were near chance and are not treated as an equal-token 3M comparison.

## Context pilot

Both branches resumed the identical 1,048,576-token Co4-L checkpoint and trained 262,144 additional tokens. Fixed context 128 (batch 8) reached **3.872539 / 48.06426**. Context 256 (batch 4, equal tokens per step) reached **3.978874 / 53.45683**. Incremental wall time was 200.738 versus 202.420 s; peak RSS was 1.901 versus 1.905 GB. The curriculum was worse by 0.106335 loss with no useful throughput or memory benefit. **Freeze context 128 and prune the 256 curriculum.** No reasoning re-query was justified.

## SLlama

No Phase 6 training was attempted. Phase 5 had already read the primary source and established that the available implementation truthfully covers RRHP, SPMLP, and whole-layer sharing but omits underspecified PWA, so it is not faithful. A bounded challenger was optional and could not change the required Co4-S/Co4-L decision enough to justify delaying persistence and GIBC work. The branch remains implementation-only, not a claimed negative training result.

## Triton target

The best candidate is the **Co4 MOD/context elementwise forward+backward chain**, after explicit module-range attribution. It plausibly combines portions of the measured `mul`, `add`, activation, concatenation, and copy traffic while eliminating intermediate tensors. Residual+RMSNorm and output loss fusion rank second and third. Custom GEMM and replacement SDPA are not recommended. No kernel or speedup is claimed; see `artifacts/triton_target_analysis.md`.

## Persistence

Checksum-verified Safetensors exports, strict fresh-model round trips, private resume state, configs, tokenizer metadata, dataset manifests, metrics, parameter counts, hashes, source Git commit, seeds, tokens, and cumulative wall times were uploaded and remotely size-verified in the private repository:

- `experiments/co4-l-3m/` — remote revision `aa15825c3670de6ee1a95c5ebebdc513ac6432b2`
- `experiments/co4-l-seed2/` — `1ea801515b4bd1fb794ab35f8b307c9dc40785f4`
- `experiments/dense-16m/` — `d23e12a9d6f5dab8b20aacb9b671feb41e524e53`
- `experiments/co4-s-3m/` — `289a49a130858c7e1a0784b398545f3b8f0542a7`
- `best/` — selected Co4-S, `79d82354656f9f0d67c0768833afdcbb2260d921`
- `latest/` — selected Co4-S, `7754b690d92f199738abe8f84fa1f05ee5b7934f`

No authentication material is stored in repository files. Transient binary checkpoints remain git-ignored.

## Final freeze

Frozen model scale and geometry: Co4-S, d_model 256, eight layers, four query heads, one KV head, SwiGLU hidden 768, untied input/output embeddings, 8,923,392 parameters. Frozen training policy: pinned 4K BabyLM tokenizer, AdamW, learning rate 3e-4, weight decay 0.1, betas 0.9/0.999, gradient clipping 1.0, constant schedule, fixed context 128. Co4-L remains the scientifically valuable quality runner-up and scaling reference, but does not receive the final competition training budget.
