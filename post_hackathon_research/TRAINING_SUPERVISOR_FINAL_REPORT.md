# Training supervisor final report

**Prepared:** 2026-10-10  
**Scope:** Autonomous post-hackathon research campaign managed by `posthack_master.py`, launched 2026-10-01 22:24 UTC. This report covers its training, evaluations, decisions, and archive handoff. It does not reinterpret the frozen hackathon model as a campaign result.

## Executive summary

The supervisor completed the nine research phases and ran a finalizer. The recorded state says phases 1–9 are `COMPLETED`, `finalized` is true, there is no active or queued training job, and the finalizer log ends with exit code 0. The final archival manifest and synthesis were written. The frozen 48.6M hackathon checkpoint was preserved separately.

The strongest result by absolute held-out loss was the fresh 32M Co4 control, `R32-CO4-CONTROL`, after 300M tokens: weighted held-out loss **2.90383**, WikiText BPB **1.52771**, and four-task raw accuracy **0.4300**. This is the campaign's best completed large-scale baseline, not a demonstrated architecture win: the 32M Co4 versus dense matched comparison was inconclusive, and only the Co4 arm entered the fresh 300M-token tournament.

Several smaller findings warrant targeted follow-up, but none supports a large new training allocation yet. Differential-Co4 showed the clearest bounded pilot signal (lower loss and BPB at about 50M tokens, with a measured throughput cost relative to Co4). D4 annealing was the strongest provisional 16M BPB screen. Both need replication/confirmation. Conversely, the 32M promotion gate found no candidate that cleared both the registered quality and broad-capability guard. The campaign's own recommendation is to retain the best control, avoid substantial GPU spend on an unreplicated mechanism, and choose the next experiment to resolve one specific uncertainty.

## 1. What was run and what happened

The supervisor scheduled a sequence of controlled research phases, with per-run train/evaluation receipts, checkpoint hashes, measured throughput calibration, bounded retry behavior, and a final synthesis/archive step. The phase plan covered matched Co4/dense baselines; data, tokenizer, context and curriculum studies; value residuals and model growth; developmental experts and PHM-4; post-training transfer; a fresh 32M tournament; second-seed and operator pilots; and final archival.

The campaign ran from 2026-10-01 to the finalizer on 2026-10-10. It therefore completed well before the controller's 2026-10-16 working deadline. This is a completed snapshot, not proof that the scheduled service was disabled or that the full 15-day window elapsed. The final synthesis table still labels `final` as `RUNNING`, while `master_state.json` says `finalized: true` and the last finalizer log records `exit=0`; the latter two are stronger completion evidence, but the stale status should be reconciled in controller metadata.

The main results include a matched 8M/16M/32M Co4-versus-dense study; DATA-D/tokenizer/context and training-recipe screens; a 2×2 depth-growth/curriculum design; four expert-growth arms; a PHM-4 FFN screen; retention-aware post-training arms; a 300M-token 32M Co4 control; a second 16M seed pair; and 50M-token nCo4, Differential-Co4, and Engram pilots. Run receipts show completed endpoints for all registered training runs, including the small smoke runs. A smoke run is operational evidence only and is not a quality result.

## 2. Best results

| Result | Evidence | Interpretation |
|---|---|---|
| Best absolute campaign held-out loss | `R32-CO4-CONTROL`: 2.90383 at 300M tokens; WikiText BPB 1.52771; four-task raw accuracy 0.4300 | Best completed large-scale control and natural next checkpoint for comparison. It is one 32M architecture/seed trajectory, not proof Co4 beats dense. |
| Best matched-scale Co4/dense outcome | At 32M/100M tokens, dense loss 3.06593 vs Co4 3.06782; BPB 1.6197 vs 1.6177 | Extremely close and directionally mixed across measures; paired-window uncertainty includes zero. No architecture winner established. |
| Second-seed 16M comparison | Co4 minus dense: held-out loss −0.00211, BPB −0.00694, four-task accuracy −0.0065 | Supports an unresolved, small Co4 quality edge on language-model metrics with a capability tradeoff. A single seed pair cannot establish generality. |
| Strongest provisional 16M recipe screen | D4 anneal vs A16 control: loss relative change about +0.018%, BPB −0.474% | BPB is promising; loss barely changes in the unfavorable direction. Treat as a replication candidate, not a selected recipe. |
| Strongest bounded operator pilot | Differential-Co4 at ~50M: loss 3.46590 and BPB 1.87628; Co4 milestone 3.53939 and 1.89704 | Pilot result is favorable on both language-model metrics. Calibrated throughput is 10,163 tokens/s; the synthesis reports −6.17% versus Co4. Replicate at matched seed/data/compute before scaling. |
| Stronger task-tuned diagnostic | P2 reached 0.4243 four-task raw accuracy and BPB 1.61535 vs D4's 0.4118/1.62169 | P2 trained on official task training splits and is explicitly an upper-bound diagnostic. It cannot be compared as a general pretraining improvement. |

The 16M matched Co4/dense seed-1337 pair was similarly close: Co4 loss 3.09548 vs dense 3.09741, BPB 1.6294 vs 1.6418, with Co4 taking 6.01 hours vs 5.72. The window-bootstrap loss interval spans zero, and the single seed does not establish a stable advantage. At 8M, dense had lower loss for both seeds, but BPB and wall-time movements were mixed. Across all sizes, parameter matching was close, but the 6NT proxy omits Co4 operator overhead and is not a complete FLOP comparison.

## 3. Does any result warrant more work?

**Yes, narrowly and in a staged way.** The next spend should buy evidence, not scale. These are the most worthwhile follow-ups:

1. **Replicate Differential-Co4 at a matched 8M or 16M scale.** Reuse the same data order, seed pairing, tokenizer, optimizer where scientifically appropriate, and a common held-out evaluator. Record operator-level compute and full wall time. The present 50M pilot is encouraging but the attention-map design and throughput cost make quality-per-wall the decision metric.
2. **Confirm the D4 annealing signal.** Run a fresh paired seed against its base control with the same token budget and schedule. Require loss and BPB to agree, and include source-wise loss plus a full capability evaluation. The current BPB improvement does not coincide with a clear loss improvement.
3. **Resolve 32M Co4 versus dense only if that comparison matters to the product/research question.** The 32M comparison used one seed per architecture and 100M tokens per arm, while the 300M tournament ran only a Co4 control. A replicated matched 32M pair is needed before claiming an architecture advantage or choosing an architecture for a larger trajectory.

Other mechanisms do not currently warrant more compute absent a changed hypothesis. No 32M candidate cleared the promotion guard. PHM-4 was slower and worse on both loss and BPB at matched parameter count. nCo4 slightly improved quality metrics in a 50M pilot but ran about 9.91% slower, so the bounded quality-per-wall gate did not pass. Developmental experts did not improve the fixed expert baseline: F0 had the best held-out loss, while F1's BPB was marginally lower; cloning/catch-up arms were worse on both. Curriculum/depth growth and value residual screens were ambiguous or unfavorable on the registered joint quality criteria. The tokenizer/context and data results remain useful constraints, but they do not by themselves nominate a higher-quality training recipe.

## 4. What went well

- **The supervisor delivered an auditable campaign.** Run IDs, completion receipts, checkpoint hashes, source commits, calibrated throughput, registered decisions, and per-phase logs support reconstruction of what ran.
- **Experiments were gated by evidence and time.** The controller did not launch a second mechanism seed pair without a promotable 16M candidate, nor a fresh near-49M trajectory without supported 32M improvement. It protected archival time and preserved the frozen hackathon checkpoint.
- **Matched comparisons and multiple evaluation views were used.** Co4/dense pairs tracked held-out loss, BPB, capability, and wall-clock time; the report explicitly calls out where metrics disagree and where bootstrap windows do not represent seed uncertainty.
- **The breadth of research was useful.** The study tested data mixture, context truncation, tokenizer sizes, curriculum, capacity growth, expert initialization, post-training retention, and novel attention/normalization pilots. Several tempting claims were rejected by gates instead of being presented as wins.
- **Finalization and archive evidence exists.** The finalizer exited successfully; the frozen record identifies checkpoint, data manifest, tokenizer, and hashes. The final phase log shows archive uploads and no empty source commit was attempted.

## 5. What went badly or remains weak

- **Many apparent differences are smaller than the evidence can support.** Most comparisons have one seed, fixed-screen capability measurements, or validation-window intervals that do not capture seed variance. Several result classifications are explicitly `AMBIGUOUS`.
- **The promotion tournament was under-informative.** Only the 32M Co4 control appears in its result table. It establishes a useful baseline but cannot identify a promoted mechanism or settle Co4 versus dense at that scale.
- **Some evaluation scopes are narrow.** Capability screens generally use 1,000 examples per task, and point-estimate shifts are not sufficient for broad claims. P2 is task tuned and its advantage does not transfer automatically to general capability.
- **Efficiency proxies omit real costs.** The 6NT proxy omits Co4 operator overhead and optimizer/sampler costs. PHM-4 materially expanded dense-equivalent FFN multiply-adds despite matched parameter count. Throughput and wall-clock measurements are essential to interpret these designs.
- **The campaign had operational friction.** Phase 4 failed 12 times before it completed, according to `master_state.json`. Restart/retry handling ultimately worked, but repeated failures consumed time and indicate that the phase should be inspected before reuse. Low-priority tokenizer training was paused after throughput fell from roughly 10.6k to 6.2k tokens/s.
- **Completion metadata is inconsistent.** The campaign synthesis still says final is running even though finalized state and exit-0 finalizer log say complete. Correct the stale status or preserve an explicit note in the archive so later readers do not mistake this for a partially finished run.
- **No large-scale scientific conclusion is justified.** The near-49M run was correctly gated out. A 300–500M token trajectory would not establish the frozen model's historical 1.9B-token outcome, and the campaign does not claim it does.

## 6. What should be done next

### Immediate closeout

1. Reconcile `master_state.json`, `FINAL_RESEARCH_SYNTHESIS.md`, and any live schedule/status file so all report the same final state and timestamp. Verify the archive manifest covers the expected artifacts and hashes.
2. Save a compact decision table with each candidate's hypothesis, comparison, result, uncertainty, and next action. Keep the existing detailed per-question reports as supporting evidence.
3. Inspect the repeated Phase 4 failures and document the root cause and recovery change before another autonomous campaign uses that path.

### Next research tranche

1. Choose **one** primary branch: Differential-Co4 replication, D4 annealing replication, or a matched 32M Co4/dense replication. Do not run all three in parallel on the same limited budget.
2. Pre-register the practical promotion rule: paired seeds, matched data and token budgets, a minimum meaningful improvement in held-out loss and BPB, non-inferior full capability, and a wall-clock/compute benefit appropriate to the intended use.
3. Add enough seeds to estimate between-seed variation. Keep paired validation-window bootstrap as a secondary check, not a substitute for seed replication.
4. Evaluate the selected candidate and its control on full task suites and source-wise held-out data, then report confidence intervals and failure cases. For post-training, retain clear labels distinguishing general replay/alignment from task-tuned diagnostics.
5. Scale only if the replicated result passes the pre-registered gate. Otherwise retain the 32M Co4 control as the best campaign baseline and stop that branch.

### Suggested priority

If the objective is **research on architecture/efficiency**, Differential-Co4 is the best next bounded experiment because its pilot is favorable across loss and BPB and it has an explicit efficiency question to resolve. If the objective is **improving the current training recipe**, replicate D4 first because it is closer to a conventional, deployable schedule change. If the objective is **selecting the production backbone**, the missing matched 32M dense arm is the priority. The current evidence does not support choosing a large-scale winner among these objectives.

## Evidence and qualifications

Primary campaign evidence is in `FINAL_RESEARCH_SYNTHESIS.md`, `MASTER_CONTROLLER.md`, `master_state.json`, `controller_state.json`, `FINAL_ARCHIVE_MANIFEST.json`, and `master-final.log`. The focused comparisons are documented in `MATCHED_CO4_DENSE.md`, `SECOND_SEED_REPLICATION.md`, `PROMOTION_32M.md`, `STRETCH_DIFFCO4.md`, `STRETCH_NCO4.md`, `PHM4_SCREEN.md`, `DEVELOPMENTAL_EXPERTS.md`, `POSTTRAINING_TRANSFER.md`, `CGLS_CO4.md`, and `VALUE_RESIDUAL.md`.

All capability numbers are screening results unless the source report explicitly says otherwise. Validation-window bootstrap intervals cover sampled windows, not training-seed variation. Raw-byte exposure and 6NT FLOPs are proxies with the limitations stated in the source reports. The report therefore distinguishes observed metric changes from claims that require replication.
