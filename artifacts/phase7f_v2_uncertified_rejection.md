# DATA-D-BROAD-v2 — UNCERTIFIED audit record

Status: **REJECTED — DO NOT TRAIN**

The preserved top-level manifest is `artifacts/data_d_v2_uncertified_manifest.json`. Its SHA-256 is `087046006b31169e772295c84e2468f02741fab9983b0c48a3121fd47028e54d`, identical to the historical build at `artifacts/data/phase7f_v2/pilot-100m/manifest.json`. The ignored historical directory remains the immutable audit location for its shards, child manifests, SQLite index, and decision logs.

The build materialized 100,045,134 tokens in the intended 50/25/25 mixture, but it is not acceptance-certified and must never be used for training. It was rejected for five independent reasons:

1. The builder used an empty `common_validation_token_proxy`, so it did not decontaminate against the frozen common-validation corpus.
2. Validation token arrays lacked document-level manifests, stable IDs, boundaries, provenance, and recorded hashes; train/validation disjointness was therefore not independently auditable.
3. Resume validation covered only selector state, not the complete model, optimizer, scheduler, Python/PyTorch RNG, stream position, token/step counters, and manifest identity.
4. Child manifests recorded builder commit `556774ea02874934e4d34c9a6c893c3d2c086001`, which does not contain the v2 builder and related implementation changes.
5. The retained records are insufficient to independently recompute normalization, contamination, SimHash, and cross-source deduplication decisions from reacquired or retained source text.

This record preserves the failure rather than relabeling or mutating the historical manifest. The replacement corpus must be freshly rebuilt from source under identity `DATA-D-BROAD-v2r1` after the repaired builder has been committed.
