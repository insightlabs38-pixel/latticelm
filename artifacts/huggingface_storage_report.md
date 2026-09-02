# Hugging Face storage report

## Status

Persistent storage was configured before Round 3. Authentication succeeded with
a write-capable access token supplied for this run; the token value was not
printed, written to an artifact, added to a URL, committed, or stored in the
repository. No token is present in this report or the source tree.

| field | value |
|---|---|
| repository | `insightlabs38-pixel/LatticeLM-research` |
| repository type | Hugging Face **model** repository |
| visibility | private |
| verified smoke revision | `76ad80c97e33f87f317e78ce09fad9056ea09f48` |
| verified path | `tests/api-smoke/` |
| safetensors SHA-256 | `320442f59401a270b4486d27cbba92164325fbcbaba001a6b2ebeea1c4e3f174` |
| shared tokenizer revision | `6c08c3e97c4c2959b6ec9d2cb37c205e16ed9734` |

The private model card is installed at repository root and prominently labels
the contents as research checkpoints rather than a final GIBC release.
The real BabyLM 4K tokenizer, compressed identity map, dataset manifest, and
reference provenance are remotely verified under `shared/` for reuse by Round
3 bundles.

## Repository policy

The source Git repository remains authoritative for code, tests, configs,
reports, metrics, and tournament results. The private Hub model repository is
limited to selected weights, tokenizer/memory metadata, manifests, and useful
resume state. Dataset copies, evaluation sets, caches, environments, build
trees, and credentials are excluded.

Semantic paths are:

- `best/` for the best qualifying validation checkpoint;
- `latest/` for replaceable cloud-failure recovery weights and resume state;
- `experiments/<name>/` for scientifically important immutable checkpoints;
- `tests/` for tiny persistence workflow checks.

## Export integrity

`scripts/export_release_model.py` writes `model.safetensors` using
Safetensors' shared-weight-aware model API. Publication aborts unless all of the
following pass:

1. construct the source model from the checkpoint config;
2. strict-load every checkpoint parameter;
3. export safetensors;
4. construct a second fresh model;
5. strict-load safetensors;
6. compare every state-dict tensor exactly;
7. compare deterministic logits exactly;
8. calculate and record SHA-256 checksums.

The bundle also contains portable config, tokenizer, normalized memory-token
map, metrics, complete training manifest, `SHA256SUMS`, and a distinct private
`resume_state.pt` without duplicate model weights.

## End-to-end smoke test

A 19,616-parameter dense model was initialized deterministically and exported.
The local export passed exact tensor and logits equality. All eight bundle files
were uploaded to the private model repository, remotely verified by filename
and byte size, downloaded at the pinned revision, checked against
`SHA256SUMS`, strict-loaded into a fresh model, and used for deterministic
inference with output shape `[1, 8, 64]`. The downloaded safetensors SHA-256
matched the local export. The optimizer state loaded into a fresh AdamW
instance, and the smoke resume artifact contained step, Torch RNG, and data
generator state.

## Upload transport observation

The current Hub API was attempted first. This cloud's outbound proxy rewrites
signed S3/LFS uploads with `Transfer-Encoding`, which the Hub endpoint rejects
with HTTP 501; Xet also returned HTTP 400. The tooling therefore retains the
official Hub API as the primary path and automatically falls back to an
authenticated Hugging Face Git push for selected artifacts. The fallback uses
an ephemeral `GIT_ASKPASS`, reads the credential only from process memory, does
not embed it in a URL, deletes its temporary directory, and verifies the final
Hub contents through `HfApi`.

The Hub rejects ordinary Git blobs above 10 MiB and binary-looking chunks. The
fallback therefore base64-encodes fixed-size parts below that limit and records
their order, original byte size, and SHA-256 in `large_files.json`. The download
tool reconstructs the exact original `.safetensors`/private resume file and
aborts unless its size and SHA-256 match before normal `SHA256SUMS` validation.

## Long-run integration

Training configuration now controls remote persistence:

- `hf_persistence_enabled` gates all network behavior and requires both a Hub
  token environment variable and `LATTICELM_HF_REPO`;
- `hf_upload_interval_tokens` controls `latest` (default 1,048,576 tokens);
- `hf_best_upload_interval_tokens` rate-limits meaningful `best` replacement
  (default 524,288 tokens);
- `hf_named_checkpoint` creates one final `experiments/<name>/` checkpoint.

Local checkpoints now include optimizer state, current step/tokens, Python and
Torch RNG state, data-generator state, best validation loss, and cumulative
wall time. This makes new Round 3 checkpoints materially more resumable than
the legacy Round 2 files. Upload failure remains fatal when persistence is
enabled, so a run cannot silently claim cloud protection it did not obtain.

## Round 2 limitation

The ignored Round 2 binary `.pt` files were already absent when this persistence
stage began (only committed JSON configs and logs survived). They therefore
could not be retroactively converted or uploaded. No result is marked remotely
persisted without actual weights. Round 3 must start only with the new verified
persistence policy enabled.

## Round 3 persistence outcome

Round 3 subsequently completed with persistence enabled. Remote revision
`859419343b47872c625123b656ce50497ef89b34` was verified through the Hub API.
`best/` contains Co4 at validation loss 3.545640, `latest/` contains the final
Dense recovery state, and immutable `experiments/co4-round3/` and
`experiments/dense-round3/` bundles preserve both scientific finalists. All
large reconstructed artifacts retain the hashes recorded at export.
