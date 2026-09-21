# Final autonomous post-training system

`scripts/run_posttraining_master.py` is the only post-training orchestrator. It
starts in `WAIT_FOR_PRODUCTION_COMPLETE` and does no model or dataset work until
`artifacts/final_production_master_state.json` contains the semantic terminal
state `PRODUCTION_COMPLETE` and both production processes have exited. It never
invokes `sudo` or controls the production service.

The immutable base lineage, state backups, events, v2 reservoir, capability
map, candidates, final selection, evidence, and release are written below
`artifacts/posttraining/`. `BASE` remains a candidate through final selection.
Post-training checkpoints always use fresh optimizer state. Training objectives
are completion-masked SFT plus DATA-D-v4 replay, continuation ranking, RFT,
SimPO, and independently gated GRPO/RLOO RLVR. Optional methods fall through to
earlier certified candidates.

Time is interpreted in `America/New_York`:

- New-experiment cutoff: **September 30, 2026 at 3:00 PM EDT**
- Absolute finalization cutoff: **September 30, 2026 at 4:00 PM EDT**

No new branch begins after 3:00 PM. Workers stop at the next checkpoint at the
cutoff. The last hour is reserved for finalist evaluation, selection, identity
verification, export, evidence generation, and one bounded retry. Measured
objective throughput is projected with a 1.35 safety multiplier.

The production evaluation companion independently evaluates the first committed
SHA-verified checkpoint at or after 1.00B, 1.25B, and 1.50B tokens. It pins the
checkpoint, uses the existing privileged production pause/evaluate/resume
protocol, durably writes raw WikiText and corrected lm-eval results, restarts
production before reporting, verifies exact monotonic resume, and permanently
completes after 1.50B. It never changes a training decision.

Deployment on the production host:

```bash
sudo install -o root -g root -m 0644 systemd/latticelm-posttraining-master-20260921.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now latticelm-posttraining-master-20260921.service
systemctl status latticelm-posttraining-master-20260921.service
```

The installed waiter should report `WAIT_FOR_PRODUCTION_COMPLETE` and remain at
negligible CPU/RAM until the semantic handoff.
