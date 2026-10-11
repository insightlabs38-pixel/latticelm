# Independent deep review — October 11, 2026

Post-submission work. This pass reviews implementation and prompt coverage, not completed scientific results. The active dense run's registered mathematical sources remain unchanged; no competing scientific-batch training was launched. User clarifications supersede the prompt for the fresh dense design, expiry, unavailable BabyLM source, submission revision, promotion policy and storage retention.

## Reproduced defects and corrections

| Finding | Risk | Correction/evidence |
|---|---|---|
| Exact relevance ties depended on padded future length | Full-Co4 causality failure; seed0 integer-valued probe at prefix 22 differed by 6.230769 in FP32 | Explicit earliest-index tie policy; independent stable-sort oracle and production cutoff/rank repair. Pure/Differential prefix, selection and gradient regressions added. No full-Co4 training had started |
| Public evaluator command identity used checkpoint path rather than experiment argument | Surviving evaluation could be misclassified as dead, allowing duplicate heavy work | Bind the exact local evaluator script, checkpoint parent, PID, boot and start ticks; unrelated scripts and runs rejected |
| Clean partial trainer exit returned0 | Deadline/graceful infrastructure stop counted as code failure | Accept an interrupted receipt only when its partial token count and hash match the verified checkpoint; failure receipts still take precedence |
| Result and predictions publication could tear | Valid expensive evaluation could halt the supervisor or lose paired evidence | Predictions commit first; result is the commit point; staged/final predictions recover both publication edges. JSON/fsync and per-task sample counts checked |
| Evaluation reuse did not pin evaluator implementation | Candidate/control outputs could silently use different pipelines | Persist exact evaluator/tokenizer-source hashes and protocol; changed protocol requires inspection; paired comparisons require identical pipeline and evaluation scope |
| Capability pairing checked doc ID/target only | Changed questions/prompts could invalidate paired statistics | Include document and prompt hashes, reject duplicates/mismatches and compare result/sample counts |
| Completion accepted incomplete output lists | Empty or incomplete receipt could mark a real experiment complete | Require source validation, protocol, WikiText, capability result and item evidence with hashes for real registrations; validation-only mode has its own required set |
| Final checkpoint could survive while validation JSON lagged | Resumed completed training would skip recomputation, then evaluator would fail on stale source losses | Recompute common source validation from the verified endpoint only when needed; write a recovery receipt and bind the regenerated output hash |
| Torn last metrics append was treated as corrupt committed history | Ordinary SIGKILL could halt analysis/funnel/plotting | Ignore only a malformed non-newline final record; preserve journals and reject malformed committed/interior records |
| A recovered fallback could be discarded at next save | Previously damaged head retained instead of the actual known-good checkpoint | Before resumed launch, promote the verified fallback in the manifest; preserve original metadata and all weight files |
| Hardlink staging changed ctime and defeated hash cache | Repeated multi-GB reads could consume training resources | Upload retained immutable artifacts directly; snapshot mutable reports only. Preserve all historical files |
| Single noisy sample-threshold crossing or step-rate-only win could promote | Isolated uncertainty/timing artifact interpreted as efficiency | Require adjacent distinct strong threshold positions; wall gain requires both better endpoint wall and favorable block interval. Timing rows use latest attempts and exclude cross-attempt blocks |
| Default queue lacked section36 scarcity priorities | Lower-priority work could consume the remaining scientific window | Switch queue ranking based on actual unfinished core work plus meaningful full-Co4 cost, preserving the fixed reserve and per-run fit guards |
| Closeout publication failure could leave FINALIZED reports | Failed Git/HF publication could overstate completion | Build finalized reports from a separate candidate state; restore live reports after any publication failure; mark durable state finalized only after both remote steps succeed |
| Required plan/report files missing; run directory ignored by Git | Incomplete prompt deliverables or final Git publication failure | Add FOLLOWUP_PLAN.md and generated per-run REPORT.md; archive reports and force-add only the exact small report allowlist |
| Compressed indexes relied on untested reconstruction assertion | Remote compressed data could fail to restore D4 inputs | Stream-decompress and verify original source SHA once per immutable gzip digest, cache the certificate; no raw historical deletion |

The previous review's historical 8M provenance blocker remains explicit. R32 and R16 audits passed; missing early trainer snapshots are not labeled equivalent. Fresh matched replacements are implemented but held pending the user's choice.

## Prompt cross-check

| Prompt sections | Reviewed implementation / limits |
|---|---|
| 1–3 | Immutable dated input audit, frozen model/main/tag, independent branch/namespace/controller. Added missing plan and per-experiment reports |
| 4–5,25 | Confirmed Oct17 expiry/reserve; observed C4A anchors; actual smoke/ongoing rates and remaining-token projections |
| 6,30–32 | Atomic state/manifests, identities/orphan adoption, boot service, checkpoint wall-time ceiling, full RNG/optimizer/sampler restore, finite loss/gradients, infrastructure/code/data distinctions and completion evidence |
| 7,18–19 | Independent C0 oracle; fresh5M compatibility; at-most-four held-out-only tuning settings; fresh50M/100M stages; at most two promoted families and one second primary seed |
| 8 | User-authorized fresh300M dense with audited R32 source/config/checkpoint/whole-WSD/sampler provenance; no hidden100M continuation or restart |
| 9–10 | Fresh150M D4 seed2026 and fresh100M MOD/Differential seed7331 pair. Quality-floor/85% anneal and deterministic sampler rule agree with the historical D4 recipe; required operational smokes remain exclusive and queued |
| 11–15,21,35 | Pinned IHMS source/author fixture; separate literal and production operators; full causal adaptation disclosed, including stable exact-tie policy. Unavailable faithful BabyLM A excluded per explicit user clarification |
| 16–17,26–27 | Exact analytical/measured parameters, shallow/deep/MOD controls, source loss/BPB/full vs screened tasks, paired uncertainty/task veto, separate wall/depth/sample/limited compute axes. Historical8M primary control choice remains pending |
| 20,22 | Explicit paired new-phase extension, stage1/2/conditional4 ladder,16M first/conditional second seed, Differential16M and one conditional combination implemented; supervisor gates execution |
| 23–24,36 | Deferred branches stay closed. One heavy worker. Default order plus time-scarcity priorities; implementation checks are small and do not launch a second scientific trainer |
| 28–29,33 | Private batched verified deltas, training/validation shards, compressed-index reconstruction, persistent cooldown/hash cache, representative real downloads, restricted retention, exact isolated Git/HF closeout. Live batched uploads have verified remote hashes/sizes at their exact returned revisions; complete migration and a new trained endpoint download remain pending |
| 34,37 | Live synthesis covers all 15 questions and explicitly distinguishes unfinished/held/skipped work, adaptations, noisy ties, negative results and supported claims |

## Verification scope and remaining work

**59 focused tests passed in 3.16 seconds** with one CPU thread. Full-Co4 registrations also reserve 300 seconds for validation/checkpoint overhead by limiting unsaved training work to 900 seconds. The active dense configuration stays immutable. Analysis cache identities now include analysis source, metric histories and item evidence.

Focused tests cover operator outputs/gradients/one optimizer step, exact ties, prefix/future/gradient leakage, parameter counts, scheduler/queue transitions, PID identity, publication crash edges, torn logs, partial exits, fallback retention, endpoint-validation recovery, archive protocol/cooldown and compressed reconstruction. These are small tests at one CPU thread. Actual production-batch smoke/resume and throughput gates run exclusively through the supervisor before each new long experiment.

This review does not certify future experimental outcomes, complete HF coverage or unavailable source fidelity. The outstanding scientific control choice remains pending. Live archival, supervisor code adoption and active-trainer integrity checks are recorded separately with timestamps/hashes; no historical deletion is performed.

## Live adoption and metadata evidence

- `DEEP_REVIEW_CONTROLLER_ADOPTION.json`: controller source commit `1c5230c42e2675fd20e5fc43d5a545c12a5ee2e4` adopted the same trainer PID/attempt, with increasing optimizer steps and unchanged registered scientific hashes.
- `DEEP_REVIEW_ACTIVE_CHECKPOINT_VERIFICATION.json`: current 28,000,256-token checkpoint hash, normalized actual config, 31,986,396 parameters, fresh empty optimizer/zero sampler receipt, complete 300M schedule position, optimizer LR, RNG presence and sampler draw accounting passed. Existing pre-launch certificates cover exact next-batch restore.
- `DEEP_REVIEW_LIVE_ARCHIVE_VERIFICATION.json`: independent remote metadata checks passed for a 16-file canonical-data commit. The two gzip indexes reconstruct to original source hashes. Mutable reports modified after that snapshot await the next delta; the live snapshot pass subsequently completed with 1,773 verified selected files and a verified remote manifest. Later revisions and a real new endpoint download remain pending.
