# Phase 7F — DATA-D-BROAD-v1 controlled comparison

## Outcome

Phase status: STOPPED AT DATA ACCEPTANCE GATE

DATA-D PREPARED: NO

TOTAL CLEAN UNIQUE TOKENS: unavailable

TOTAL RETAINED DOCUMENTS: unavailable

The predeclared mixture cannot be constructed from the allowed sources while satisfying global deduplication and validation separation. The unchanged pinned BabyLM release was wholly consumed by DATA-C except for its frozen 1% validation tail. After excluding DATA-C, eligible BabyLM training content is zero, not the required 15%. Reusing DATA-C documents would violate the global-dedup and distribution-breadth requirements; using the tail would violate training/validation separation. No optional source was substituted.

## Infrastructure

All four public repositories resolved and were pinned in `data_d_broad_manifest.json`. A reusable offline implementation was added for immutable little-endian int32 shards, manifest/hash verification, read-only mmap, affine without-replacement source streams, an exact five-batch 50/20/15/15 cycle, deterministic state serialization, exact/paragraph/near deduplication, and practical decontamination.

The synthetic 16,388-token four-source infrastructure fixture passes same-process and fresh-process reproduction, selector-state restore, shard and child-manifest corruption rejection, and wrong-tokenizer rejection. It is deliberately not represented as the required 1M–5M acquired-source fixture.

FIXTURE TEST: FAIL (required acquired-source fixture not created)

FRESH-PROCESS BATCH HASH: PASS (synthetic fixture)

EXACT RESUME TEST: PASS (data-selector next batch only; model/optimizer interruption not run)

MANIFEST VERIFICATION: PASS (synthetic fixture only)

## Production fields

TOP-LEVEL MANIFEST SHA256: `6d073ff0fa6c6742c3f2acd830a3c422c72bd96948ea70b9c67c8d7f1c02dbf7`

Realized source mixture: unavailable

EXACT DOC REMOVALS: unavailable

PARAGRAPH DEDUP REMOVALS: unavailable

NEAR-DUP REMOVALS: unavailable

HELLASWAG MATCHING DOCS REMOVED: unavailable

ARC-EASY MATCHING DOCS REMOVED: unavailable

PIQA MATCHING DOCS REMOVED: unavailable

WINOGRANDE MATCHING DOCS REMOVED: unavailable

WIKITEXT MATCHING DOCS REMOVED: unavailable

COMMON-VAL MATCHING DOCS REMOVED: unavailable

No model was launched. All requested Co4-L DATA-D metrics, throughput, runtime, peak RSS, and Hugging Face revision are unavailable. The DATA-C control is preserved in `data_c_vs_data_d_25m.csv`; no deltas were fabricated.

BROADER-DATA RESULT: E — INCONCLUSIVE (experiment did not run)

## Verification

`PYTHONPATH=src .venv/bin/pytest -q`: 40 passed.

`.venv/bin/python -m compileall -q src scripts`: passed.

`git diff --check`: passed.
