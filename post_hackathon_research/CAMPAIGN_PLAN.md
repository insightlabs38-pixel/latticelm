# Post-hackathon research campaign

Started 2026-10-01 22:24:33 UTC. The working deadline is 2026-10-16 22:24:33 UTC. The final 12 hours are reserved for evaluation and archival. The user has one additional day of VM lifetime outside this working window.

The enabled user systemd `latticelm-posthack-master.service` runs the single
restartable supervisor described in `MASTER_CONTROLLER.md`. It adopts the
already-running Phase 1 service and owns every later phase, timing decision,
final synthesis, and private Hugging Face archive. No Git commit or push is
authorized for this campaign.

The frozen hackathon checkpoint and data identities are in `FROZEN_HACKATHON_STATE.json`. The source revision at campaign start is `43832b4c7bfe8be007ca6b9d1042aa2041c7f280`. All research output belongs under this directory and under `post-hackathon/2026-10-01/` in the private Hugging Face repository. Git commits and Git/GitHub pushes are prohibited for this campaign.

## Scientific sequence

1. Calibrate 8M, 16M, and 32M Co4/dense runs with batch 32, context 256, and 12/16 intra-op threads. Calibrate the 49M regime only if promoted.
2. Run modern matched Co4/dense controls at 8M (two paired seeds), 16M, and 32M. Initial token budgets follow the execution prompt, subject to the live 12-hour archival reserve.
3. Screen post-MOD value residual at 16M, then matched 8K/16K tokenizer models if justified by compression and parameter economics.
4. Build Research-D metadata from existing DATA-D-v4 documents, then test D1 through D4, with D5 conditional on evidence. Do not rebuild DATA-D-v4.
5. Test curriculum-guided depth growth and developmental expert birth under matched FLOPs. Run one PHM-4 FFN screen at 16M.
6. Test replay and reference-constrained post-training, promote at most two independent mechanisms to the 32M tournament, and run a fresh near-49M trajectory only if evidence and at least 30 hours justify it.
7. Apply four-hour engineering gates to nCo4 and Differential-Co4 after mandatory tracks are secure. Use frozen evidence/time decisions for a fresh 8M extension, a matched 16M second seed, a selected-mechanism second seed, and the overflow adaptive-birth, scalar-skip, and tiny Engram screens.

One heavy trainer may run at a time. Promotion is based on source-wise held-out loss, WikiText-103 BPB, broad capability, transfer capability, uncertainty, FLOPs, and wall time. The four competition tasks remain HellaSwag, ARC-Easy, PIQA, and WinoGrande. The frozen transfer suite is ARC-Challenge, OpenBookQA, BoolQ, and LAMBADA; capability-selector training must exclude their evaluation splits as well as competition evaluation splits.

## Control design

The dense control uses the production QK-Norm, GQA, RoPE, SwiGLU, untied embeddings, Muon/AdamW partition, 2/83/15 WSD schedule, 8e-4 peak learning rate, beta2 0.95, batch 32, context 256, tokenizer, corpus, sampler, and seed. The only architectural difference in the first paired study is the Co4 MOD transformation and latent streams. Config pairs are parameter-matched to within 0.1%.

The controller records every registration before training, including hashes, parameter counts, optimizer partitions, evaluation points, thresholds, expected wall time, and estimate provenance. Results and decisions append to JSONL. The local run folder retains two rolling resumable checkpoints; selected model artifacts and compact evidence are uploaded to the private Hugging Face repository.

## Existing evidence reused

The prior tokenizer audit found a weighted 8K compression gain of 11.73% over 4K under its earlier rule. The earlier 4K competition-task prompt audit found the largest fraction exceeding context 256 was 0.299% (PIQA). A new full eight-task audit will determine the context-512 training decision. The common-subset 16K tokenizer is always trained and measured; a parameter-matched 16K model is trained only when its frozen compression/evidence and time gates pass.
