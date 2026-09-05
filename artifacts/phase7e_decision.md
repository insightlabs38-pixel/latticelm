# Phase 7E decision and Phase 7F recommendation

## Decisions

**SHOULD ~24M ACTIVE Co4 BE TESTED NEXT: NOT YET.** Co4-L retains a clear language-modeling capacity advantage, but reasoning is mixed and the planned broader-data control never ran. Capacity is not yet the cleanest next variable.

**SHOULD BASE TRAINING CONTINUE: YES, but only as a bounded 150M DATA-C diagnostic after the resume/data-manifest handoff is hardened.** The 50→100M language-model gains remain meaningful, available pools stay non-repeating through 150M, and 50M additional tokens project to 2.84 hours and approximately **$0.27** at the existing approximate $0.0959/hour C4A Spot rate. Stop if 100→150M common loss improves <0.02, WikiText PPL improves <3%, or two reasoning tasks materially regress. A 200M extension would add 5.68 hours/~$0.54 from 100M but would exceed the current 50M FineWeb pool at a 25% mixture and must not run without expansion.

**POST-TRAINING TIMING: AFTER CAPACITY/DATA TEST.** Design can begin now, but training should wait for the controlled DATA-D result. Ranking: (1) procedural continued pretraining/SFT—high expected reasoning benefit, moderate implementation/forgetting risk, low CPU cost, clearest rules; (2) sequence-level adversarial/contrastive ranking—moderate/high upside, moderate implementation risk, moderate forgetting risk, clear if negatives are generated deterministically; (3) RLVR—potentially high task-specific upside but highest implementation/variance risk and less broad rules clarity. No external teacher/distillation is required.

**TRITON PRIORITY: LOW.** Architecture may be stable, but the winning data/training regime and reasoning intervention are not. Keep the measured Co4 MOD/context chain as the first candidate after profiling; do not displace benchmark-quality experiments.

## Ranked Phase 7F actions

1. **Prepare and validate DATA-D-BROAD-v1.** Question: does broader unique data fix the reasoning plateau? Upside: cleanest test of Phase 7D’s high-priority hypothesis. Runtime: roughly 2–6 hours preparation/validation plus acquisition variability. Risk: provenance/dedup bugs. Stop unless exact DATA-C exclusion, benchmark/common-validation decontamination, mmap reproducibility, and resume fixture all pass.
2. **Train Co4-L DATA-D to exactly 25M.** Question: at fixed architecture/tokenizer/budget, does mixture breadth improve reasoning without unacceptable LM loss? Upside: controlled causal evidence. Runtime: ~1.4 training hours plus ~20 minutes evaluation at observed throughput. Risk: noisy near-chance benchmarks. Stop/declare negative if common loss and at least three reasoning tasks worsen beyond uncertainty; never select on one task.
3. **Continue verified Co4-L DATA-C 100→150M.** Question: is the 90→100M local flattening transient? Upside: inexpensive PPL/BPB gain with no corpus repetition. Runtime: ~2.84 hours; estimated VM cost ~$0.27. Risk: plateau and further PIQA/WinoGrande regression. Stop at 125M/150M checkpoints using the thresholds above.
4. **Design a deterministic procedural/SFT suite and sequence-ranking prototype; do not train it yet.** Question: can targeted, teacher-free supervision address weak reasoning? Upside: direct benchmark-relevant skills. Runtime: 1–3 engineering days; later pilot hours are small. Risk: contamination and forgetting. Stop unless generators/verifiers, held-out templates, and base-LM regression gates are auditable.
5. **Revisit ~24M Co4 only after DATA-D 25M and DATA-C 150M.** Question: do gains remain capacity-limited under the winning data regime? Upside: potentially stronger sub-50M model. Runtime: likely ~2–3× Co4-L per token; calibrate before training. Risk: spending compute on data-limited capacity. Stop if a short predeclared calibration does not beat Co4-L’s matched-token slope.

Do not run another seed now: the unresolved data comparison has higher information value.
