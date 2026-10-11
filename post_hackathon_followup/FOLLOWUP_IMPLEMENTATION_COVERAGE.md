# Final-week implementation coverage

Post-submission work; scientific endpoints remain pending until their receipts, checkpoint hashes and evaluator outputs pass the controller.

| Prompt requirement | Implementation / evidence |
|---|---|
| Freeze and isolate existing state | Immutable input audit; isolated branch and frozen checkout; confirmed submission tag at `43832b4c7bfe8be007ca6b9d1042aa2041c7f280` |
| Dense300M schedule provenance | R32 audit of every LR/step, start/end/continuation events, source/config/data/tokenizer hashes, sampler counters, first three fresh losses, next-batch replay |
| Fresh dense300M matched design | F32-DENSE-S1337-FRESH300M registration; identical seed/data order/tokenizer/batch/optimizer/peak LR/WSD, exact parameter count; passing scientific-batch kill/resume certificate |
| Resilient controller | PID+boot+start ticks, durable atomic state, exclusive master lock, orphan adoption, periodic atomic checkpoint manifests with previous fallback, preemption polling, boot-enabled user systemd unit |
| Mandatory D4 and Differential | Fresh D4 seed2026/150M paired to R16; one fresh MOD/Differential seed7331/100M pair; automatic actual-batch 100-step/kill/resume/evaluator gates |
| Faithful BabyLM FCo4-A | Unavailable source implementation and ambiguous paper; user-confirmed exclusion of invented fidelity. Source-grounded B/C proceed independently |
| Source-grounded B/C | Pinned author fixture and license, literal oracle, separate prefix-chunked production operator, pure and hybrid LM families; all causal adaptations disclosed |
| C0 validity | Multiple-seed prefix/future/selector/gradient tests, forward/backward and one-step oracle equivalence, author compatibility, parameter accounting; required again for each production preflight |
| C1 bounded tuning | 5M fresh compatibility; base plus at most three LR/init alternatives only if unstable/non-learning or clearly poorly tuned; held-out selection frozen before downstream comparison |
| C2/C3/C4 | Fresh50M screen, up to two fresh100M families, one frozen winner with exactly one further seed; common shallow/current-MOD/deep controls; coherent paired LM/BPB/task gates. Historical8M source provenance is incomplete; primary deep/MOD comparisons await an explicit user choice of fresh replacements or recovered snapshots |
| Optional paired extension | Explicitly new100M WSD phase to total200M for both winning seeds and controls; exact parent model/optimizer/RNG/sampler restoration; disclosed LR restart, not uninterrupted200M |
| Optional depth and16M | Source-family stage1/2/conditional4 ladder, matched shallow controls, one16M first seed and only one conditional second seed |
| Optional Differential16M and combination | Matched16M/150M pair gated on independent replication; one source-family+Differential50M screen gated on both independent positive branches |
| Evaluation and fairness | DATA-D/source losses, paired WikiText rows/BPB/PPL, four task raw/normalized item evidence, labeled1000 screens/full endpoints, actual wall/throughput/RSS and separate disclosed compute/depth/parameter axes |
| Selective checks | Focused unit/oracle/controller tests and automatic actual-config100-step preflight; no competing full trainer or broad benchmark run during dense launch verification |
| Deadline and adaptation | Confirmed expiry Oct17 20:10UTC; scientific stop16:10UTC; actual-rate remaining-token projections, overhead/evaluation margin, unrun deadline outcomes never interpreted as negatives |
| HF migration | Existing inventory and private verified SHA/size/blob delta reuse; canonical training/validation shards/manifests and historical missing evidence; moderate batched background archive, persistent hash cache and rate-limit cooldown, representative real remote downloads/loads. Revised batch protocol is unit-tested; its real upload and complete migration remain pending |
| Storage | No historical deletion authorized or implemented. Current-master redundant rolling pruning only; separately approved passed smoke weights removed while evidence stays |
| Final reporting/closeout | Restartable synthesis answers all15 questions, comparison/summary/matrix/plots, final partial checkpoint archive, representative new endpoint load, isolated branch commit/push, exact final Git/HF receipt |

Implementation is distinct from completed experiments. Automatic gates can reject or defer a run with an explicit reason. The live schedule and receipts are authoritative for execution status.
