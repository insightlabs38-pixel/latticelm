# Post-long-runs decision

## Decision

**FACT — Tournament transfer classification: E.** The original tournament failed before training. Its successor produced valid LR-10 and LR-25 25M runs, but neither showed statistically supported external transfer versus DATA-D 25M. Both paid a large WikiText cost: +10.1% PPL for LR-10 and +14.3% for LR-25. LR-25 was only the controller's heuristic 25M selector, not an evidence-supported winner.

**FACT — Overnight branch:** the successor correctly took the classification-E DATA-D fallback and completed the existing DATA-D lineage to 100M. No LR run reached 50M/100M/200M, and no 24.4M capacity run occurred.

**INFERENCE — Current best competition candidate:** Co4-L DATA-D 100M. It is the strongest preserved, inference-verified candidate: WikiText 60.1219 PPL / 1.88821 BPB; HellaSwag 0.267576; ARC-Easy 0.292508; PIQA 0.541349; WinoGrande 0.505130. Only its HellaSwag gain over DATA-D 25M is borderline suggestive; the other benchmark movements remain ambiguous.

**INFERENCE — Bottlenecks:** synthetic-to-natural transfer and evaluation noise. LR learned the held-out synthetic distribution to low loss, but did not transfer externally and displaced useful natural tokens. Single-seed benchmark deltas are mostly smaller than paired uncertainty.

## Compute decision

Do not continue LR-25, do not search more mixture fractions immediately, and do not promote the unrun 24M model. Preserve DATA-D 100M as the current candidate.

Only two next experiments are presently justified:

1. **Replicate DATA-D 100M at seed 2026.** Same 15,949,760-parameter Co4-L, certified DATA-D-v2r1, 100M tokens; about 5.6 hours at 5,003 tok/s. Success: reproduce WikiText within 3% and improve at least one reasoning task with a paired 95% CI excluding zero across a seed-aggregated analysis. Stop at 25M/50M if validation is materially worse than both prior lineages. This resolves seed/evaluation uncertainty around the current candidate.
2. **Controlled LR-10 transfer test at 100M with a matched fresh DATA-D control, only after experiment 1 confirms stability.** Same Co4-L, 90/10 mixture, seed 2026, 100M tokens per arm; about 5.6 hours per arm at measured throughput. Success: LR arm beats its matched control on at least two external tasks with paired, seed-aware 95% intervals excluding zero while WikiText PPL cost stays under 5%. Stop LR at 25M if WikiText cost exceeds 8% and no external task is suggestive. This resolves whether LR transfer emerges with scale without trusting synthetic loss.

**POST-TRAINING RESEARCH PRIORITY: MEDIUM.** Verifier-backed hard-negative/answer ranking is conceptually better aligned than more template-mixture search, but it is speculative and should not run until rules are clarified. **ORGANIZER CLARIFICATION NEEDED BEFORE SUBMITTED CHECKPOINT: YES.**

No third experiment is justified until those results separate model variance from a real transfer effect.
