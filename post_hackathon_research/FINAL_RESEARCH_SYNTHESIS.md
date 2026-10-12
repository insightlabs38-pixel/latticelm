# Final post-hackathon research synthesis

Prepared 2026-10-10T09:31:58.256625+00:00. The working window began 2026-10-01T22:24:33.875000+00:00 and ends 2026-10-16T22:24:33.875000+00:00; the last 12 hours are reserved for evaluation and Hugging Face export. The user's one-day VM lifetime gap remains outside this 15-day working window.

The frozen 48.6M hackathon model and its 1.9B-token result are historical artifacts. Post-hackathon model IDs below represent new research trajectories. The research workspace is Git-free; source changes are archived only to the private Hugging Face research repository.

## Campaign status

| Phase | Status |
|---|---|
| final | RUNNING |
| phase1 | COMPLETED |
| phase2 | COMPLETED |
| phase3 | COMPLETED |
| phase4 | COMPLETED |
| phase5 | COMPLETED |
| phase6 | COMPLETED |
| phase7 | COMPLETED |
| phase8 | COMPLETED |
| phase9 | COMPLETED |

## Answers to the registered questions

**Modern Co4 versus dense:**
- 8M seed 1337: dense lower held-out loss; Co4 − dense loss +0.0196, BPB +0.0117, wall ratio 0.944, paired-window interval [0.014559610188007355, 0.034857973828911766], paired four-task accuracy interval [-0.020006250000000003, 0.005506249999999966]. A per-token advantage requires the loss and BPB signs to agree; wall advantage also requires comparable quality at no greater wall time.
- 8M seed 2026: dense lower held-out loss; Co4 − dense loss +0.0031, BPB -0.0127, wall ratio 0.918, paired-window interval [0.0007534228265285493, 0.017901642248034477], paired four-task accuracy interval [-0.01225, 0.0125]. A per-token advantage requires the loss and BPB signs to agree; wall advantage also requires comparable quality at no greater wall time.
- 16M seed 1337: held-out direction unresolved; Co4 − dense loss -0.0019, BPB -0.0124, wall ratio 1.051, paired-window interval [-0.01180789079517126, 0.004252386465668677], paired four-task accuracy interval [-0.014006250000000001, 0.0115]. A per-token advantage requires the loss and BPB signs to agree; wall advantage also requires comparable quality at no greater wall time.
- 32M seed 1337: held-out direction unresolved; Co4 − dense loss +0.0019, BPB -0.0020, wall ratio 1.029, paired-window interval [-0.006278814189136029, 0.013256011344492433], paired four-task accuracy interval [-0.01775625, 0.008749999999999999]. A per-token advantage requires the loss and BPB signs to agree; wall advantage also requires comparable quality at no greater wall time.
The initial matched study has two 8M seeds and one each at 16M and 32M. A second 16M Co4/dense seed was subsequently run; see SECOND_SEED_REPLICATION.md. The 6NT proxy omits Co4 operator overhead, so parameter matching alone does not establish FLOP efficiency.

**Data composition, selection, curriculum and late annealing:** The strongest provisional BPB screen was D4-16M-ANNEAL (-0.47% versus its control). See source-wise losses and broad capability before treating this as a supported recipe.

**Tokenizer and context:** The common-subset 4K/8K comparison measured 3.255 versus 3.643 weighted raw bytes/token. At 150M tokenizer tokens, WikiText BPB was 1.6498 versus 1.6535; the byte-matched 8K checkpoint BPB was 1.7310. The 16K screen was run by its frozen evidence/time decision. This subset cannot by itself explain the external systems' quality gap.
The full eight-task prompt audit found a maximum 23.90% of requests meaningfully truncated at context 256; the context-512 decision is in CONTEXT_512.md.

**Value residual:** AMBIGUOUS: held-out change -0.21%, BPB change +0.64%.

**Curriculum and capacity growth:** The registered 2×2 interaction and active-layer FLOP proxy are in CGLS_CO4.md.
D3 curriculum's endpoint held-out change was +0.13% and BPB change was +0.82%; early-versus-late sampled difficulty is recorded in DATA_CURRICULUM.md. A convergence claim requires the token curves, not only this endpoint.
The E2−E1−D3+D0 interaction was -0.00161 held-out loss and -0.0030 four-task raw accuracy; lower loss and higher accuracy indicate favorable interaction. The active-layer FLOP and wall comparisons are in CGLS_CO4.md.

**Developmental experts:** F0 fixed, F1 naive birth, F2 cloning and F3 clone plus catch-up completed. The expert report gives quality, utilization, router entropy and domain/expert mutual information; an MI difference alone is not evidence of a useful specialist.
F3 did not beat both fixed and random-birth controls on held-out loss and BPB: F3 3.31777/1.75580, F0 3.22777/1.72992, F1 3.25197/1.71903. F3 domain/expert MI was 0.0050 nats; attribution and endpoint diagnostics determine whether the quality result reflects specialization.

**PHM-4:** The single 16M FFN-only screen is reported in PHM4_SCREEN.md; parameter saving is not interpreted as arithmetic saving.
PHM-4 did not clear the bounded quality-per-wall signal: BPB change +0.87%, wall-time change +121.02%. Parameter matching alone is insufficient.

**Retention-aware post-training:** P0/P1/P2 and interpolation were evaluated. The largest P1 fraction under the retention gate was 1.00. P2 is a task-tuned diagnostic upper bound.
P1 versus its 16M parent changed four-task raw accuracy by +0.0047, WikiText BPB by +1.33%, and disjoint held-out skill NLL/token by -5.60782; source-wise retention and paired capability intervals govern transfer claims. The approximately on-policy and full-size decisions are separately recorded.

**32M promotion:** The tournament included 1 fresh 300M-token arms. See PROMOTION_32M.md for source-wise, BPB and capability evidence.

**Near-49M trajectory:** See NEXT_49M_TRAJECTORY.md. A 300–500M-token trajectory does not establish a 1.9B-token outcome.

**nCo4:** See STRETCH_NCO4.md for the engineering/scientific gate.

**Differential-Co4:** See STRETCH_DIFFCO4.md for the engineering/scientific gate.
nCo4 did not clear a bounded quality-per-wall pilot signal: BPB -0.89%, wall time +9.91% versus its matched AdamW control. A positive single pilot still requires replication.
Differential-Co4 cleared a bounded quality/CPU-cost pilot signal: BPB -1.09%, calibrated throughput -6.17% versus Co4. The attention operator profile and held-out loss comparison are in STRETCH_DIFFCO4.md.

## Future allocation

No 32M candidate cleared the registered quality and broad-capability guard. No substantial GPU allocation is justified by this campaign alone; retain the best control and investigate the most informative unresolved mechanism with a better-powered design.
No branch is labeled a definitive scientific failure from the current screening evidence; gated-out stretches are time decisions, not negative results.

## Limits

Capability evaluations are fixed 1,000-example screens unless a run explicitly records a fuller evaluation. Paired validation-window bootstrap does not replace model-seed uncertainty. Raw bytes in DATA-D training curves are source-ratio estimates. The 6NT proxy omits attention and optimizer overhead; wall-clock throughput is reported separately. Missing phase evidence is labeled directly above.
