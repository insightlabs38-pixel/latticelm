# LatticeLM audit after the two long sessions

Audit date: 2026-09-07 UTC. This report separates observed evidence (**FACT**), evidence-based interpretation (**INFERENCE**), and untested proposals (**SPECULATION**). No training, data generation, upload, HPO, architecture, post-training, or Triton job was launched during this audit.

## Executive finding

**FACT:** The named tournament did not complete. It failed at 17:02:27 UTC after three worker starts hit `expected target dtype to be Long or Byte, but got Int`; zero training steps and no checkpoint were produced. The overnight successor then trained valid LR-10 and LR-25 models from scratch, evaluated/published both, classified the outcome **E**, and followed its declared E branch to a DATA-D-only 100M fallback.

**INFERENCE:** LatticeReason shows substantial learnability on its own held-out-template distribution but no demonstrated external transfer. At 25M, replacing natural tokens with LR consistently worsened natural validation and WikiText; every reasoning-benchmark delta is compatible with noise in paired analysis. DATA-D 100M is the current best verified competition checkpoint, though its external reasoning gains are mostly still uncertain.

## 1. Process and service integrity

| Unit | ActiveState / SubState | Result / status | Start | Finish | Runtime | Exit reason |
|---|---|---|---|---|---:|---|
| `latticereason-tournament-20260906.service` | inactive / dead; unit file now not found | state JSON says FAILED; systemd historical properties unavailable | 2026-09-06 17:02:06 | 17:02:27 | 20.813 s | three dtype errors; retries exhausted |
| `latticereason-overnight-20260906.service` | inactive / dead | success / 0 | 2026-09-07 02:28:22 | 08:32:10 (inactive 08:32:11) | 6:03:47.935 | normal `status=0/SUCCESS`; later ExecStop also 0 |

**FACT:** No trainer subprocess, dataset worker, Hugging Face upload, or relevant child process remains. The only matching process was the long-lived project tmux server. `artifacts/post_tournament_overnight.lock` remains, contains the exited PID, and is stale; no writer/uploader exists. It was not removed. **SERVICES STOPPED CLEANLY: PARTIAL**: successor yes; tournament no.

## 2. Reconstructed experiment timeline

The canonical lineage is in `experiment_lineage.csv`. Important correction: the successor's reports said “TOURNAMENT COMPLETED: YES”, but the predecessor state/events/log prove otherwise. The successor's `recover_tournament` routine treated an inactive/not-found predecessor as terminal without checking its scientific completion state, then ran LR-10/LR-25 itself. That recovery produced valid experiments, but it did not make the original tournament complete.

- **VALID_COMPLETE:** successor LR-10 25M, LR-25 25M, and DATA-D 100M fallback; prior Co4-S DATA-C 25M, Co4-L DATA-C 100M, DATA-D 50M.
- **SUPERSEDED:** DATA-D 25M native recovery state, because the same lineage is now at 100M; its metrics remain valid and its earlier HF milestone was recorded.
- **METRICS_ONLY:** Co4-L DATA-C 25M and 50M; preserved metrics/raw evaluations but no corresponding preserved weights.
- **FAILED:** original tournament LR-10 attempts.
- **NOT_RUN:** LR refinement fractions, LR continuation to 50M/100M/200M, and 24.4M capacity experiment.

All canonical training used seed 314159. Co4-L has exactly 15,949,760 parameters (320 width, 10 layers, four query heads/one KV head, FFN 960, context 128, untied embeddings, no memory table). Tokens/parameter are recorded in the master table.

## 3. Checkpoint integrity

| Checkpoint | Native SHA-256 | Strict load / deterministic inference | Exact resume | HF revision |
|---|---|---|---|---|
| LR-10 25M | `b46408d2…ad2fad` | YES / YES | YES | `837ca620994d1b332deaf94b81b7dd7d3f8e7082` |
| LR-25 25M | `43ee5899…01a31` | YES / YES | YES | `530ef189103d8959d02188c9487fe376462f8820` |
| DATA-D 100M | `50d8cdf0…fe5e` | YES / YES | YES | `5ae234a6a9c127ede46768baaf7f0d40892e2df7` |
| DATA-D 50M | `cc9801ea…fea8f9` | prior strict/remote verification PASS | YES at preserved HF revision | `1cf7116faf4b2896594239c83008518bcc84c37d` |
| DATA-C 100M | `3bf27b60…03cb4` | prior strict roundtrip YES | YES | `95d27f1c012893d12e34005b067a58a666fd472b` |

**FACT:** For all three new exports, SHA256SUMS, 15,949,760 parameter identity, strict safetensors load, and repeated-logit equality passed. LR native sidecars match; optimizer, scheduler, Python RNG, PyTorch RNG, exact global/source/family token counts, DATA-D manifest hash, tokenizer hash, mixture scheduler state, and independently reconstructed next-batch SHA all passed. DATA-D 100M preserves optimizer/scheduler/RNG/data-selector state and is the result of two successful exact resumes (25→50→100M); source counts are exactly 50M/25M/25M. Export-time remote hashes are recorded for every file. No new remote mutation was performed.

**CHECKPOINT INTEGRITY: PASS for important preserved checkpoints; PARTIAL across historical metrics-only milestones.**

## 4. Repository and provenance health

**FACT:** HEAD is `fc20610` (`Record post-tournament overnight results`), matching `origin/main`. Experiment-driving commits precede their canonical runs: generator `f9515f6`, tournament hardening `07b3356`, successor `185488c`. Export manifests point to commits containing the relevant worker/controller code. Results were committed afterward by the autonomous successor.

The pre-audit tree was not clean: two terminal state files were modified by final shutdown, three native checkpoints and sidecars were intentionally untracked, the event log and stale lock were untracked. The audit adds five report/CSV files. Existing scientific outputs were not altered. `git diff --check` passes.

- `PYTHONPATH=src pytest -q`: **106 passed**.
- `python -m compileall -q src scripts`: **PASS**.
- Credential-pattern scan over source/scripts/tests/configs/artifacts/docs: **0 matches**.
- DATA-D manifest certification/validation is exercised by the successfully verified loaders.
- LatticeReason canonical manifest at `artifacts/data/lattice_reason/canonical/manifest.json`: **PASS**, 1,000,000,000 tokens and all 1,000 shard hashes.
- Copied manifests at `artifacts/lattice_reason/*.manifest.json`: **FAIL as portable manifests**, because relative shard paths resolve in the wrong directory. The underlying canonical shards are not corrupt.

No audit commit or push was made because the pre-existing working tree is intentionally dirty and the report should be reviewed first.

## 5. Canonical results and competition status

See `latticelm_master_results.csv`; invalid historical runs are excluded. The table retains explicit `METRICS_ONLY` and `SUPERSEDED` labels rather than treating them as inference candidates.

**BEST COMPETITION CANDIDATE / BEST WIKITEXT:** Co4-L DATA-D 100M, 15,949,760 parameters, 100M natural tokens, 5,003 tok/s, 20,122.9 s (5.59 h), exact-resume verified, HF `5ae234…92e2df7`. Metrics: WikiText **60.121859 PPL / 1.888214 BPB**, HellaSwag **0.267576**, ARC-Easy **0.292508**, PIQA **0.541349**, WinoGrande **0.505130**.

Across the canonical valid table (without inventing an aggregate score):

- **BEST HELLASWAG:** DATA-D 100M, 0.267576.
- **BEST ARC-EASY:** DATA-C 100M, 0.297559.
- **BEST PIQA:** DATA-C 50M, 0.541893, but **METRICS_ONLY**; best preserved checkpoint is DATA-D 100M, 0.541349.
- **BEST WINOGRANDE:** LR-25 25M, 0.508287, statistically indistinguishable from control; among natural-only models DATA-C 50M has 0.505919 but is metrics-only, while DATA-D 100M has 0.505130.

## 6. Tournament reconstruction

- **LR-0 RESULT:** valid DATA-D control at 25M; PPL 83.4094, scores 0.262896 / 0.283249 / 0.534276 / 0.498027.
- **LR-10 RESULT:** valid successor run; synthetic held-out loss 0.753410, but DATA-D loss +0.04859 and WikiText PPL +10.1%; all four external deltas ambiguous.
- **LR-25 RESULT:** valid successor run; synthetic held-out loss 0.800058 (worse than LR-10), DATA-D loss +0.11571 and WikiText PPL +14.3%; all external deltas ambiguous.
- **Optional refinement:** NOT_RUN.

**TOURNAMENT TRANSFER CLASSIFICATION: E.** This follows the declared controller rule: its heuristic selector picked LR-25 by counting tiny >0.3-point benchmark movements, then the >10% PPL penalty forced E. **BEST 25M LR FRACTION: none supported; LR-25 was only the mechanical selector. TOURNAMENT WINNER: NONE. WINNER CONTINUED TO 50M DURING TOURNAMENT: NO.**

## 7. Successor behavior

**FACT:** LR-10 and LR-25 completed at 25M. Recomputed classification E sent the controller into `TRANSFER_DIAGNOSTIC` and DATA-D fallback, exactly as declared. DATA-D resumed from 50M and completed/evaluated/uploaded 100M. Winner 100M: NO. Winner 200M: NO. Capacity: NO. The continuation gates were never applicable after E.

**Accidental divergence:** predecessor failure was mislabeled as a completed tournament; `classification` was later blanked in the final state/report even though the scaling CSV and code imply E. Selection text promoted LR-25 despite no statistical evidence. The actual branch execution itself was consistent with E.

## 8. Benchmark uncertainty

Raw sample hashes and IDs align exactly, enabling paired analysis. `benchmark_uncertainty_audit.csv` records 10,000-replicate paired bootstrap intervals, disagreement counts, and exact two-sided McNemar p-values.

**FACT:** Every LR-vs-control interval crosses zero. LR-25 HellaSwag trends negative (-0.358 points, p=0.125); its ARC/Wino gains (+0.968/+1.026 points) are not significant. LR-10 movements are smaller. DATA-D 100M HellaSwag is borderline suggestive (+0.468 points; bootstrap lower bound +0.010 point, exact p=0.0512); all other 100M deltas are ambiguous. No 0.1–0.3 point movement is called meaningful.

## 9. LatticeReason learning, shortcuts, and dose response

**LATTICEREASON INTERNAL LEARNING: MODERATE.** Both mixtures reach low held-out-template loss, including strong Boolean, physical, planning, temporal, state-tracking, and arithmetic losses. Entity-reference and spatial-relational remain much harder (~1.92–2.19). Loss by difficulty rises coherently from level 1 to 6. However, LR-0 was not evaluated on LR validation, so the causal gain over natural-only training cannot be quantified; “strong” would overstate the evidence.

**LATTICEREASON EXTERNAL TRANSFER: NONE.** No external paired delta is clear, while natural validation and WikiText clearly worsen.

Dose response is simplest described as **harmful at high LR fraction / otherwise noisy**: 25% uses 2.5× more LR tokens yet has worse overall LR held-out loss than 10%, worse losses at every difficulty, much worse entity/spatial losses, higher DATA-D loss, higher WikiText PPL, and lower HellaSwag. Some easier families improve slightly at 25%, but that is not an external sweet spot. LR-10 is the less harmful tested dose, not a validated optimum.

No LR recipe reached multiple major token milestones, so **SCALING TRAJECTORY: NOT TESTED** and whether transfer emerges/disappears with scale is unresolved.

Shortcut audit using existing metadata finds: template/vocabulary holdouts and balanced 40-example family×difficulty cells; deterministic symbolic verification; zero contamination rejections; exact token-balanced family accounting. There is no direct evidence of answer-position bias or split leakage. **INFERENCE:** low loss on six families without benchmark transfer, paired with persistent weakness on entity/spatial families, makes format/template shortcuts or a domain gap plausible. Existing metadata does not retain training-example text/answers sufficient to quantify answer-position, duplicated-state, or token-frequency effects offline. This is a prominent limitation, not a proven leak.

## 10. Natural data versus LR and capacity

At equal 25M tokens/parameters, DATA-D beats both LR mixtures decisively on WikiText and DATA-D validation. LR scores are statistically tied on all four reasoning benchmarks. Therefore LR currently provides **no meaningful external benefit** and a language-modeling cost. DATA-D 50M and 100M improve WikiText monotonically; DATA-D 100M is better than either LR 25M on three metrics and essentially tied on WinoGrande, though cross-token comparisons do not isolate mixture effects.

The optional 24,402,816-parameter geometry never ran. **CAPACITY SCALING RESULT: INCONCLUSIVE.**

## 11. Dataset health

**FACT:** Canonical corpus: exactly 1B int32 tokens, 8,453,439 examples, 1,000 immutable 1M-token shards, nine exact nested prefix views, generator commit `f9515f6`, tokenizer SHA `4f313e…637e5`, deterministic generation/verification tests passing, held-out-template/vocabulary validation, and previously recorded zero contamination hits. Current canonical verification hashes all shards successfully. Disk currently has sufficient availability for reads; the materialization-time report recorded 59.03 GiB free.

**INFERENCE:** The corpus is suitable for controlled training only through the canonical data-location manifests. The report-copy path defect should be fixed before external reuse, without touching shards or results.

## 12. Promising directions

1. **DATA-D scaling — MODERATE CONFIDENCE.** Evidence: 25→50→100M improves WikiText 83.41→73.74→60.12 and natural validation 3.696→3.556→3.466; 100M HellaSwag is borderline suggestive. Upside: strongest verified checkpoint. Uncertainty: other benchmarks are noisy and WinoGrande was non-monotonic.
2. **Low-dose LR only as a rigorously matched scale test — SPECULATIVE.** Evidence: LR-10 learns held-out LR better and damages natural metrics less than LR-25. Upside: a scale-dependent transfer effect remains logically possible. Uncertainty: no external signal at 25M and substantial PPL cost.
3. **Verifier-backed ranking/post-training research — SPECULATIVE.** Evidence: base mixture teaches synthetic format without transfer; an objective focused on correct-vs-hard-negative discrimination may align better. Uncertainty: no experiment and competition eligibility unresolved.

## 13. Weak / deprioritized directions

- **LR-25 and higher mixture fractions:** more synthetic exposure produced worse LR overall/difficulty losses, worse natural loss/PPL, and no clear external gain.
- **Broad fraction search now:** with only one seed and no signal at 10%/25%, interpolating LR-5/15 is unlikely to answer the transfer question efficiently.
- **Unconditional LR scaling:** synthetic validation alone is not a success criterion; no evidence supports spending 100M–200M tokens on LR-25.
- **Immediate 24M capacity run:** the prerequisite evidence never materialized, and capacity is not the bottleneck shown by these sessions.
- **Claiming benchmark wins from point estimates:** all LR comparisons are within paired uncertainty.

## 14. Still unresolved

- **Seed variance:** all key runs use seed 314159. Minimum resolution: one predeclared DATA-D 100M replication.
- **Scale-dependent LR transfer:** no LR run exceeded 25M. Minimum resolution, only if baseline stability is established: one LR-10 100M arm plus a matched fresh natural control.
- **LR causal internal gain:** LR-0 lacks synthetic validation. Minimum resolution: offline evaluation of the preserved LR-0 checkpoint if accessible; no training required.
- **Shortcut mechanisms:** retained aggregate metadata cannot quantify answer-position/token-frequency/duplicate-state artifacts. Minimum resolution would be a bounded audit of existing source records if those records are preserved; do not regenerate a corpus merely for this.
- **Capacity:** no 24M result. It should not be scheduled until transfer/evaluation uncertainty is reduced.

## 15. Bottleneck and post-training reassessment

**PRIMARY BOTTLENECK: SYNTHETIC-TO-NATURAL TRANSFER and EVALUATION NOISE.** Capacity and systems are not implicated: throughput stayed near 4.9–5.0k tok/s, checkpoints are sound, and more natural tokens clearly lower PPL. The research obstacle is turning procedural learnability into external behavior and distinguishing small gains from sampling/seed noise.

**POST-TRAINING RESEARCH PRIORITY: MEDIUM. ORGANIZER CLARIFICATION NEEDED BEFORE SUBMITTED CHECKPOINT: YES.** Ranking/hard-negative objectives deserve design work, not execution, because mixture pretraining failed to transfer. Rule ambiguity prevents treating any resulting checkpoint as submission-safe.

## 16. Exactly the justified next experiments

Only two are justified; details and gates are in `post_long_runs_decision.md`:

1. DATA-D 100M replication at seed 2026 (~5.6 h).
2. Conditional matched LR-10 vs DATA-D 100M test at seed 2026 (~11.2 h total), only if experiment 1 establishes baseline stability.

No third experiment is recommended. Stop after analysis.
