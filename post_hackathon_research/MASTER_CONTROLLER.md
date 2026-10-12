# Autonomous 15-day research master

The controller began at **2026-10-01 22:24:33 UTC**. Its working deadline is
**2026-10-16 22:24:33 UTC**. It stops launching training at **2026-10-16
10:24:33 UTC**, leaving twelve hours for synthesis and archival. The user's
approximately one additional VM day is outside this research window.

`scripts/posthack_master.py` is the only scheduler after the already-running
Phase 1 service. `latticelm-posthack-master.service` is enabled in the user
systemd instance and restarts on failure. The master first observes the
existing Phase 1 service; if that service is inactive before all eight arms are
archived, it starts the same service. It never launches a second Phase 1
trainer. The old `OnSuccess` phase handoffs were removed.

The master then runs, one phase process at a time:

1. Phase 2: value residual, full eight-task prompt audit and context-512 gate,
   common-subset 4K/8K tokenizer comparison and 16K model gate, Research-D,
   D1 broad mixture, and an optional fresh paired 8M extension.
2. Phase 3: disjoint capability selector, D2–D5, depth-growth 2×2 design,
   and one PHM-4 FFN screen.
3. Phase 4: fixed, random-birth, clone-birth, and clone plus catch-up experts.
4. Phase 5: pinned generic/task-tuned post-training, replay, reference KL,
   bounded on-policy replay and frozen BASE confirmation gates, and weight
   interpolation before any recovery training.
5. Phase 6: evidence-selected fresh 32M tournament and a conditional fresh
   49M trajectory.
6. Phase 8: evidence- and measured-time-gated 16M Co4/dense second seed and
   selected-mechanism second seed, reusing a common control when both launch.
7. Phase 7: time-gated nCo4 and Differential-Co4 pilots.
8. Phase 9: evidence- and time-gated adaptive expert birth, one scalar skip
   follow-up, and a small Engram revisit after required and named stretch work.
9. Finalizer: plots, hashes, synthesis, source snapshot, and private Hugging
   Face archive.

Every phase script is restartable from registered checkpoints and receipts.
`master_state.json` records the phase and attempts; `controller_state.json`
records runs, decisions, calibrated throughput, deadline, and live queue;
`LIVE_SCHEDULE.md` and `LIVE_PROJECTIONS.json` are refreshed while the master
waits for a phase child. Phase logs are `master-phaseN.log`, with per-run train
and evaluation logs under `runs/<experiment-id>/`. A phase failure is recorded
and retried with bounded backoff. The final reserve takes precedence over
further runs. At reserve time, the master signals the entire active phase
process group, waits for its signal-aware trainer to checkpoint, then runs the
finalizer.

The research workspace is detached from Git; the clean frozen checkout and
original Git metadata are preserved separately as described in
`GIT_ISOLATION.md`. The archive helper has an explicit allowlist, requires the
destination Hugging Face model repository to be private, and verifies uploaded
file size and LFS hash when available. **No Git or GitHub commit or push is
part of the controller.**

Read-only status commands:

```bash
systemctl --user status latticelm-posthack-master.service
systemctl --user status latticelm-posthack-phase1.service
cat post_hackathon_research/master_state.json
cat post_hackathon_research/LIVE_SCHEDULE.md
```
