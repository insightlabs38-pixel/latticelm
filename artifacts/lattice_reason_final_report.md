# DATA-D 50M and LatticeReason final report

## Controlled DATA-D-v2r1 continuation

DATA-D 50M COMPLETED: **YES**

| Metric | 25M | 50M |
|---|---:|---:|
| Common loss | 4.464799 | 4.180764 |
| DATA-D validation loss | 3.695686 | 3.555687 |
| WikiText-103 PPL | 83.40936 | 73.73970 |
| WikiText BPB | 2.03912 | 1.98232 |
| HellaSwag | 0.262896 | 0.264987 |
| ARC-Easy | 0.283249 | 0.291246 |
| PIQA | 0.534276 | 0.536453 |
| WinoGrande | 0.498027 | 0.471981 |

50M training throughput was 4,866.71 tok/s and cumulative lineage training time was 10,129.73 seconds. The native checkpoint SHA-256 is `cc9801ea6b908162e611ce0b8b4dcd02ceba496e812595b3c49789e7adfea8f9`. The immutable HF revision is `1cf7116faf4b2896594239c83008518bcc84c37d`; every remote bundle file was downloaded at that revision and matched its local SHA-256.

DATA-D 25→50M TRAJECTORY: **MIXED**. Language-model validation and three of four GIBC measures improved, but WinoGrande regressed materially.

SHOULD DATA-D CONTINUE TO 100M: **INCONCLUSIVE**. The base-model loss trajectory remains useful, but the controlled benchmark result is mixed. No 100M continuation was launched.

## LatticeReason

GENERATOR IMPLEMENTED: **YES**

GENERATOR COMMIT: `f9515f6bbc9f04862cc9d43c9507de8580e956f5`

ALL GENERATOR TESTS: **PASS** (99 full-repository tests; 56 generator-focused tests)

All requested engines are present: entity/reference, state tracking, spatial/relational, temporal/causal, Boolean/logical, arithmetic/quantity, procedural planning, and physical closed-world. The exact example shares are 15%, 15%, 15%, 15%, 15%, 10%, 10%, and 5%. A deterministic 100,000-example audit estimates token shares of 20.50%, 20.23%, 14.77%, 13.75%, 9.02%, 7.13%, 9.98%, and 4.62%, respectively; these differ because family realizations have different lengths and are reported explicitly rather than concealed.

TEMPLATE-HOLDOUT VALIDATION: **PASS** (1,920 examples; 40 per family/difficulty cell)

DETERMINISTIC REGENERATION: **PASS**

CONTAMINATION CHECK: **PASS** (3,731,125 frozen reference n-grams; zero generated rejections)

## Canonical materialization

LATTICEREASON GENERATION BUDGET: **8h**

100M WALL TIME: **140.108 s**

100M END-TO-END TOK/S: **713,735**

LINEAR SCALING APPROXIMATION: **VALID** (1B actual was 0.65% slower than the 10× T100 estimate)

CANONICAL POOL MATERIALIZED: **YES**

FINAL GENERATED TOKEN COUNT: **1,000,000,000**

FINAL CANONICAL MILESTONE: **1B**

TOTAL GENERATION WALL TIME: **1,410.258 s**

AVERAGE END-TO-END TOK/S: **709,090**

PEAK RAM: **2,443,161,600 bytes (2.28 GiB)**

FINAL DISK USAGE: **4,000,000,000 shard bytes (3.73 GiB)**

FREE DISK REMAINING: **63,384,072,192 bytes (59.03 GiB)**

WHY GENERATION STOPPED: **TARGET_REACHED**

| View | Exact tokens | Examples through prefix | Manifest SHA-256 | Verification |
|---:|---:|---:|---|---|
| 1M | 1,000,000 | 8,459 | `af490e0532a7bab41414e71009424b5e381e6899628578b6b410482a56b55d8f` | PASS |
| 5M | 5,000,000 | 42,268 | `8c8865fce776530f155eeacb8ecc903f189aab9ec3522b71d4984f76a87b2d32` | PASS |
| 10M | 10,000,000 | 84,539 | `1d6f04d4f09e94037dcc038f6912c4525387aac81eef783b4f040dcce0fcccf0` | PASS |
| 25M | 25,000,000 | 211,339 | `b4243fcc515133b8e85616877c2e1109fc069e3abc7449ea439736b5569a4939` | PASS |
| 50M | 50,000,000 | 422,681 | `31755f47aab2495c365e5cb03790e20a4857b172557b094b864bddb1bc6ab857` | PASS |
| 100M | 100,000,000 | 845,359 | `e2c2bd9bbd69678d8340bbe083d494de65e7554a6b20954e4986b0fac5c18496` | PASS |
| 250M | 250,000,000 | 2,113,308 | `b969281819d7d2d0e75ef25eb4760c13fc030bb9c2ec6cc157f863ed064ef90b` | PASS |
| 500M | 500,000,000 | 4,226,699 | `e3c9ae91edc5b6a433f83eb575d9315786a49515a8836e1602d3ac1edae23979` | PASS |
| 1B | 1,000,000,000 | 8,453,439 | `56c7634a0bdc27e21bce454da4a315ecd69c0294474a42d584860223636b4a98` | PASS |

Every view is an exact prefix over the same 1,000 immutable 1M-token shards; no milestone data is duplicated. Family proportions by example are approximately 15/15/15/15/15/10/10/5 in every view and exact counts are stored in each manifest.

## Next experiment recommendation

The safest first experiment is a from-scratch Co4-L base-pretraining control using DATA-D plus a small, predeclared LatticeReason token mixture, compared at the same total token and optimization budget against DATA-D alone. This remains base pretraining and avoids modifying the certified 50M lineage. Continued pretraining or fine-tuning from the 50M checkpoint, SFT, and later contrastive hard-negative training may be useful, but should wait for explicit organizer confirmation that post-training is permitted.

No model was trained on LatticeReason in this phase.
