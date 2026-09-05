# Phase 7D — Data-rich capacity scaling

Both controlled DATA-C lineages reached exactly 25M tokens with no Spot interruption. Co4-S resumed the canonical 10M state exactly; Co4-L used random initialization with the same predeclared seed and data ordering. No HPO, benchmark tuning, Triton integration, broader-data substitution, or 50M continuation occurred.

| Metric | Co4-S | Co4-L |
|---|---:|---:|
| Parameters | 8,923,392 | 15,949,760 |
| Tokens | 25,000,000 | 25,000,000 |
| Tokens/parameter | 2.8016 | 1.5674 |
| Common validation loss | 3.400900 | 3.328604 |
| Common validation PPL | 29.99108 | 27.89938 |
| Training tok/s | 7552.59 | 4964.97 |
| Cumulative training time | 3275.93s | 5035.28s |
| Peak RSS | 2.672 GiB | 2.918 GiB |
| WikiText-103 PPL | 115.52928 | 102.55101 |
| WikiText BPB | 2.18928 | 2.13435 |

## Restrained GIBC snapshot

| Task accuracy | Co4-S | Co4-L |
|---|---:|---:|
| HellaSwag | 0.258315 | 0.260108 |
| ARC-Easy | 0.293771 | 0.284512 |
| PIQA | 0.519587 | 0.520131 |
| WinoGrande | 0.489345 | 0.481452 |

These zero-shot scores remain near chance and are treated as early scaling measurements. No intermediate GIBC snapshot or benchmark-driven tuning was performed.

## Provenance and persistence

Phase 7D-0 verified the clean Git state at `4dd336e`, all frozen DATA-C/tokenizer/validation hashes, exact optimizer/scheduler/RNG/data-stream recovery fields, post-10M next-batch hashes, local disk capacity, HF access, and the test suite. There were no Spot interruptions. Exact private resume states remain in rolling local and HF `recovery/` paths.

The permanent Co4-S inference bundle remotely verified at revision `22f07105a26e2743b32c1b3aec21ec460242f920`; the corrected Co4-L bundle verified at `4bd98a6f87f5d4594cd7fe0ba3aa28c659e5ae32`. Every remote file matched its local SHA-256.

The 10M S−L gap was 0.051250; the 25M gap is 0.072295. See `data_rich_scaling_analysis.md` for slopes and equal-wall interpolation. The final classification is **A**, with the qualification that the gap did not widen in every interval.

The first completed Co4-L attempt was excluded because it did not reproduce the canonical 640-token masked batch at 10M and advanced its BabyLM stream by two blocks. It is retained under `artifacts/archive/phase7d_invalid_unpaired_l/` for audit. The reported rerun matches Co4-S stream draws at 10M and 25M.

The broader-data design is documented in `future_data_scaling_plan.md`; no broader data entered either controlled lineage and no bulk 1B-token download was started.

Required final fields

Co4-S 25M COMMON VAL LOSS: 3.400900

Co4-L 25M COMMON VAL LOSS: 3.328604

Co4-S 25M TOK/S: 7552.59

Co4-L 25M TOK/S: 4964.97

Co4-S 25M TOKENS/PARAMETER: 2.801625

Co4-L 25M TOKENS/PARAMETER: 1.567422

DATA-RICH CAPACITY RESULT: A

MODEL(S) ADVANCING TO 50M: Co4-L; optionally Co4-S as controlled reference

SHOULD ~24M MODEL BE TESTED YET: NO

BROADER DATA EXPANSION PRIORITY: HIGH
