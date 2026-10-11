# Final-week follow-up execution plan

Post-submission research on `posthack-followup-2026-10`; frozen main/submission is `43832b4c7bfe8be007ca6b9d1042aa2041c7f280`. Historical scientific artifacts remain untouched. The separate registry, controller state and decision log are authoritative.

## Confirmed constraints

- Hard VM expiry: October 17, 2026, 16:10 EDT / 20:10 UTC. Scientific stop: 16:10 UTC that day, leaving four hours for evaluations, verification and closeout.
- Fresh dense300M replaces an invalid continuation of a completed100M WSD schedule. R32 fresh initialization, uninterrupted300M schedule and sampler provenance passed.
- Source-grounded IHMS pure and hybrid causal adaptations proceed. Faithful BabyLM FCo4-A is unavailable and is not invented; all causal, readout, hybrid and exact-tie adaptations are disclosed.
- Conservative promotion uses coherent held-out loss/BPB, paired uncertainty and a veto on clear task regression. Sample, depth, wall and compute remain separate axes. Two primary seeds maximum for the winning full-Co4 formation; no four-seed tournament.
- Delete only redundant current-master rolling checkpoints and separately approved passed smoke weights; retain evidence. No historical deletion.

## Execution order and gates

| Work | Budget/design | Gate |
|---|---|---|
| Fresh dense vs R32 MOD | seed1337, fresh300M; same tokenizer/data order/batch/optimizer/LR/WSD | Passed actual-batch100-step interruption/resume and evaluator gates; currently running |
| D4 replication | seed2026, fresh150M vs audited R16 | Actual-batch smoke/resume exercises late anneal; common source/BPB/full-task evaluation |
| Differential replication | fresh MOD/Differential pair, seed7331,100M each | Matched source order and recipe; coherent conservative replication gate |
| IHMS B/C C0 | Pinned author fixture, independent literal oracle, causal production | Forward, gradients, optimizer step, prefix/future/selection/gradient leakage including exact ties |
| C1 compatibility | Fresh5M per family; at most four predefined global LR/init arms | Extra arms only for instability/non-learning or clearly poor validation; freeze selection without downstream HPO |
| C2 screen | Fresh50M per viable family and shallow/deeper/current-MOD controls | Measured throughput before launch; loss/BPB and labeled1000-example tasks; at most two promoted families |
| C3 primary | Fresh100M promoted family and controls | Full evaluation; comparator-specific coherent quality/sample/depth/wall evidence |
| C4 confirmation | One winner, exactly one additional seed2026 | Same-direction evidence on the same comparator/axis across seeds1337/2026; reject clear task regression |

Historical8M controls have incomplete source provenance. Their primary comparisons remain held pending the user's choice of four fresh matched100M controls or recovered historical trainer snapshots. Fresh replacements are implemented and will be scheduled only for promoted comparisons, after explicit choice. R32/R16, mandatory replications and fresh C1/C2 work proceed independently.

## Optional work — implemented, supervisor decides execution

1. Paired100M-to200M extension for both winning seeds and controls if late learning persists. Preserve parent model/optimizer/RNG/sampler; register and disclose one new100M WSD phase and LR restart. Never label it uninterrupted200M.
2. Source-family depth1/2/conditional4 ladder with matched shallow controls. Reuse depth2 primary configuration where applicable; report Co4 stages and total blocks separately.
3. One16M/150M run and shallow control after two consistent8M seeds; one second16M seed only after favorable first evidence and a deadline fit.
4. One16M/150M matched MOD/Differential pair after independent Differential replication.
5. One8M/50M full-Co4+Differential screen only after independent success of both mechanisms.

PHM, Clifford, nCo4, MoE, CGLS, value residual, Engram, post-training, DeltaNet, BLT, Titans and multi-token work remain deferred.

## Runtime and scientific protection

One heavy worker at a time. Every long run requires an immutable registration, exact parameter/source/data/tokenizer checks,100-step actual-batch smoke, checkpoint/resume equality, evaluator loading, measured rate and a deadline projection. Full-Co4 adds configuration-specific C0 gates. Verification remains focused and small while the dense trainer runs.

Atomic checkpoint manifests retain current, previous known-good and scientific milestones. Before a resumed launch, a verified fallback is promoted in the manifest without deleting damaged evidence. Python/NumPy/Torch RNG, optimizer state, exact sampler/order, logical/processed tokens and schedule position are restored. The supervisor validates PID/boot/start ticks, adopts surviving children and restarts at boot; it cannot start a stopped Spot VM itself.

Default queue follows section24. When actual remaining core work threatens the reserve, scheduling switches to section36 priorities. No run launches unless its remaining training plus overhead/evaluation fits. Infrastructure interruptions do not consume scientific retry counts; code/data failures stop for inspection.

## Evaluation, migration and closeout

Use the common source-validation, WikiText BPB/PPL and four-task pipeline. Pair document/prompt/target identities; pin evaluator source hashes; publish predictions before result commit; require all artifact hashes before completion. Screens are labeled and never drive C1 tuning. Report exact parameters, source losses, wall/throughput, RSS, depth and explicitly limited compute proxies.

Private HF delta migration runs continuously at moderate priority, reuses verified remote duplicates, includes canonical training/validation shards and certifies reconstruction of compressed document indexes. Rate-limit cooldown, batched commits, per-file and ledger hash/size verification are implemented. Representative downloads/load tests cover the frozen model, a new research model and paired tokenizer. No complete migration claim precedes verified coverage.

At closeout: stop scientific workers, evaluate selected completed endpoints, preserve partial resume state, generate per-experiment reports and all15 synthesis answers, verify HF deltas and real loads, commit/push the isolated branch without changing main, and record exact final Git/HF receipts. Unrun/held/deadline-skipped experiments are not negative results.
