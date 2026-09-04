# Phase 7A / P0 — final training-data selection

## Executive answer

**SELECTED FINAL DATA REGIME: DATA-C — 75% BabyLM 2026 Strict 100M / 25% practically decontaminated FineWeb-Edu by sampled training tokens.**

**RUNNER-UP: DATA-A — 100% BabyLM 2026 Strict 100M.**

The frozen 8,923,392-parameter Co4-S was never changed. DATA-C won the fixed balanced common validation at every pilot checkpoint, reached loss **3.957705 / perplexity 52.3371** at 3,145,728 tokens, and used 1,473.50 seconds. DATA-A reached **4.144128 / 63.0626** in 1,489.45 seconds. The mix therefore wins at equal tokens and equal CPU wall time. DATA-B was clearly weakest at 1M and was pruned as planned.

## Integrity and frozen controls

All Phase 1–6 artifacts, manifests, Co4-S configuration, source-analysis outputs, persistence tooling, and clean initial Git status were inspected first. Runs started from random initialization with seed 1337 and shared the pinned tokenizer, Co4-S geometry, untied embeddings, AdamW (3e-4, weight decay 0.1, betas 0.9/0.999), clipping 1.0, constant schedule, batch eight, context 128, and two PyTorch threads. Training consumed deterministic, affine-permuted, non-overlapping 128-token blocks within each source; DATA-C used six BabyLM and two FineWeb sequences per batch, realizing exactly 75/25. No pretrained weights, generated teachers, benchmark training examples, Tiny Shakespeare, architecture search, or benchmark-driven mixture tuning was used.

## Acquisition and provenance

The official `BabyLM-community/BabyLM-2026-Strict` repository was pinned at revision `9e57baaaa91ac3c638746be14d1d5fa6c789f4cf` (MIT dataset-card license), downloaded on 2026-09-03, and retained separately from the earlier Strict-Small release. Its six sources are BNC Spoken, CHILDES, Gutenberg, OpenSubtitles, Simple Wikipedia, and Switchboard. Exact file sizes/hashes and the deterministic per-source 1% tail split appear in `babylm_strict_100m_manifest.json`. Raw corpus and token binaries remain git-ignored.

FineWeb-Edu used official `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (ODC-By dataset-card license, underlying pages retain their terms). The reproducible stream used a seed-1337, buffer-1000 shuffle rather than a first-N prefix; candidates needed 50 normalized words and language score >=0.9. Streaming stopped at 20,000,582 accepted training tokens. A SHA-256 document-ID rule reserved 703 documents / 1,065,118 tokens for validation before training could see them.

## Decontamination

Validation/test material for HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText-103 was used only to build contamination fingerprints. Normalized exact reference strings >=80 characters or two matching normalized 13-word n-grams triggered removal. Practical decontamination screening found/removed **9 candidate overlaps** under this heuristic (seven WikiText-103 flags, two HellaSwag flags) from 16,677 streamed candidates. This does not detect paraphrases and is not a guarantee of perfect decontamination. Full method and machine-readable document results are in the dedicated decontamination artifacts.

## Tokenizer and quality

The unchanged BabyLM 4,096-token byte-level BPE hash is `4f313ebc481a77e8ad2179cf2d7a3836b28d50773ef9f62ef08831bf076637e5`. BabyLM measured 2.794 chars/token, 0.542 words/token, 1.846 tokens/word, and zero UNKs. FineWeb-Edu measured 3.207 chars/token, 0.534 words/token, 1.874 tokens/word, and zero UNKs. Compatibility was acceptable on both.

BabyLM contributed 187,825,322 available train tokens across 11,485,880 nonempty line-documents; FineWeb-Edu contributed 20,000,582 tokens across 13,875 long documents. Because token binaries deliberately omit document boundaries, exact unique-document participation in each sampled run is unknown rather than fabricated. No source category loss breakdown is claimed: compatible trustworthy category labels were unavailable. DATA-C source-specific validation is represented by its corrected 75/25 validation (final loss 3.758786); the common suite is independently balanced 50/50.

## Common and own validation

The immutable common suite contains exactly 250,000 held-out tokens from each family. Its 500,000-token binary hash is recorded in `common_validation_manifest.json`. DATA-A and DATA-B own validation use their held-out source. A protocol defect made the initially logged DATA-C `own_validation_loss` column use a larger 50/50 source-balanced set; these raw rows remain preserved. The final DATA-C checkpoint was re-evaluated on the intended 75/25 own-validation composition, yielding **3.758786**, and the corrected value/status appears in `data_regime_comparison.csv`. This defect does not affect any common-validation value, model update, ranking, or checkpoint.

## Matched pilot

| regime | 1M train loss | own validation* | common loss | common ppl | wall s | tok/s | outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| DATA-A BabyLM | 4.179971 | 4.098849 | 4.752617 | 115.887 | 525.48 | 1,995.47 | advance |
| DATA-B FineWeb-Edu | 4.634077 | 4.862342 | 5.483732 | 240.743 | 501.31 | 2,091.67 | prune |
| DATA-C 75/25 | 4.502298 | 4.817604* | **4.552271** | **94.848** | **500.11** | **2,096.69** | advance |

DATA-B's own validation was reasonable relative to its common loss, but it generalized poorly to the balanced suite at this short budget. DATA-C led from 262K onward and combined the source strengths.

## Top-two extension

| tokens | DATA-A common | DATA-C common | DATA-A wall s | DATA-C wall s |
|---:|---:|---:|---:|---:|
| 1,572,864 | 4.510405 | **4.286013** | 772.69 | 747.52 |
| 2,097,152 | 4.325862 | **4.131917** | 1,009.12 | 988.51 |
| 2,621,440 | 4.233290 | **4.044666** | 1,249.28 | 1,231.57 |
| 3,145,728 | 4.144128 | **3.957705** | 1,489.45 | 1,473.50 |

Both remained unsaturated. DATA-C's advantage persisted at approximately equal wall time and ended at 0.186423 loss / 17.0% lower perplexity. Exactly 2,359,296 BabyLM and 786,432 FineWeb tokens participated in its 3M run. Peak RSS was 3.106 GB versus 3.091 GB for DATA-A.

## Restrained GIBC snapshot

Only the two 3.145M finalists were evaluated once. Checkpoint hashes were `172413...e56d32` (DATA-A) and `f8eb48...b1b7a3` (DATA-C); exact full hashes and task metrics are in `gibc_snapshot_phase7a.csv`.

| metric | DATA-A | DATA-C |
|---|---:|---:|
| HellaSwag raw / normalized | 0.25354 / 0.24816 | **0.25851** / 0.24796 |
| ARC-Easy raw / normalized | 0.25000 / 0.25505 | **0.26599 / 0.26852** |
| PIQA raw / normalized | **0.52339** / 0.49674 | 0.52285 / **0.50054** |
| WinoGrande raw | **0.50039** | 0.49487 |
| WikiText-103 perplexity | 262.03 | **211.78** |

Reasoning differences are mixed and mostly within standard errors, but ARC-Easy and WikiText transfer align with the common-validation decision. No mixture was changed afterward.

## Decision and next run

A 10M comparison was not necessary: DATA-C clearly led at 3.145M under the specified early-stop criterion. The next final-training checkpoints should be **10M, 25M, and 50M**. The still-negative common-loss slope justifies 10M and probably 25M. Evaluate the slope at 25M and 50M before authorizing 100M; this phase provides no evidence for blindly maximizing tokens. `configs/latticelm_final_data.yaml` freezes the selected corpus revisions, exact token mixture, loader, model, optimizer, validation, and persistence policy.

## Persistence

Checksum-manifested bundles were uploaded to the verified private Hugging Face repository. Remote file sizes were checked after upload: `experiments/data-babylm-3m/` revision `99a652ab895707d06941e7080e6e49b09e1b15fb`, `experiments/data-finewebedu-1m/` revision `69e91a23f746b501b107d3e4e5bf2f9fd96bcc9d`, and selected `experiments/data-mix-3m/` revision `1e7bd21f21a9ec0366ac56fd03f97adb1b0140a0`. The bundles include Safetensors, private resume state, config, metrics, tokenizer/report, data/validation manifests, curves, checksums, and raw evaluation results for the finalists. No credential was written to disk or Git.
