# Phase 7B — canonical final pretraining status

## Status and integrity

Phase 7B infrastructure and preflight completed, but the canonical run did **not** reach its first 10M selection gate in this execution window. It was terminated cleanly after the first recovery evaluation at 1,000,448 tokens. This report deliberately does not fabricate milestone, GIBC, release, or final-selection results.

The lineage started from random initialization with seed 314159. Frozen geometry was Co4-S (8,923,392 trainable parameters), untied embeddings, d_model 256, eight layers, four query heads, one KV head, SwiGLU width 768, context 128, no conditional memory. AdamW remained at learning rate 3e-4, weight decay 0.1, betas 0.9/0.999, epsilon 1e-8, clipping 1.0, and a constant schedule.

DATA-C used exactly six BabyLM and two FineWeb-Edu sequences per frozen 1,024-token step. The reconstructed Phase 7A arrays reproduced every pinned SHA-256. The 1,000,448-token point consumed 750,336 BabyLM and 250,112 FineWeb-Edu tokens, exactly 75%/25%; neither stream repeated.

## Preflight

The source commit was `f74b46c89bb7bcc873affe28ddb63d3d74a5a812`. Tokenizer, all source token arrays, common validation, decontamination summaries, parameter cap, private Hub access, smoke inference, short training, exact stream-resume test, CPU/RAM/cgroup metadata, and tracked-file credential scan passed. Authentication was process-local and is absent from tracked files.

The decimal milestones are not divisible by the frozen 1,024-token effective batch. The immutable policy evaluates at the first whole batch at or above each label: 10,000,384; 25,000,960; and 50,000,896 realized tokens. This preserves the frozen batch and exact 75/25 ratio instead of silently using a partial, ratio-breaking batch.

## Partial measured result

| checkpoint | tokens | common val loss | common PPL | WikiText PPL | HellaSwag | ARC-Easy | PIQA | WinoGrande | wall time | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recovery-1m | 1,000,448 | 4.562346 | 95.8080 | unevaluated | unevaluated | unevaluated | unevaluated | unevaluated | 483.704 s | 2,068.31 |
| 10M | 10,000,384 | not reached | not reached | not reached | not reached | not reached | not reached | not reached | not reached | not reached |
| 25M | 25,000,960 | not reached | not reached | explicitly unevaluated | explicitly unevaluated | explicitly unevaluated | explicitly unevaluated | explicitly unevaluated | not reached | not reached |
| 50M | 50,000,896 | not reached | not reached | not reached | not reached | not reached | not reached | not reached | not reached | not reached |

At recovery-1m, BabyLM validation loss was 4.161834 and FineWeb-Edu validation loss was 5.371603; peak RSS was 3,047,112,704 bytes. No correctness failure or data exhaustion was observed. The attempt is not presented as the canonical final model, and its recovery checkpoint is not allowed to replace `LatticeLM-Base`.

## Resume and persistence

The checkpoint includes weights, optimizer and constant-scheduler state, Python and PyTorch RNG, per-source serialized draw positions, cumulative tokens/wall time, and best validation loss. Tests confirm a restored draw count generates the same remaining permutation without an accidental restart. The run stopped before the configured approximately-5M remote-persistence point, so no false remote-persistence claim is made.

## Decision

**BEST BASE CHECKPOINT: NOT YET DETERMINED.**

**FINAL TRAINING TOKENS: 1,000,448 (incomplete attempt).**

**SELECTED DATA REGIME: 75% BabyLM 2026 Strict / 25% decontaminated FineWeb-Edu.**

**SELECTED ARCHITECTURE: untied LatticeLM-Co4-S.**

**100M EXTENSION RECOMMENDATION: NO DECISION; do not launch.**

A subsequent execution must resume this exact serialized stream (or explicitly document a restart if source changes require it), persist at 5M and major gates, run GIBC at 10M and 50M, and only then select a base checkpoint and decide the 100M gate.
