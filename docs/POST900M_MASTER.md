# Post-900M autonomous master

`latticelm-post900m-20260912.service` is the restart-safe successor to the live
`latticelm-final-scale-20260909.service`. It may be enabled while its predecessor
is running. Application-level checks sleep for five minutes between observations,
and systemd ordering covers simultaneous boots.

The master uses a separate state, lock, event log, decision, certification,
scaling analysis, report, and handoff namespace under `artifacts/`. It does not
modify `final_scale_master_state.json`. A valid handoff requires predecessor
`COMPLETE`, no unresolved child, an immutable 900M checkpoint and matching SHA,
the exact 900M curve row, frozen model/config/backend identity, strict loading,
all optimizer/RNG/selector fields, and complete DATA-D-v3 certification.

Inspect it with:

```bash
systemctl status latticelm-post900m-20260912.service
journalctl -u latticelm-post900m-20260912.service
.venv/bin/python scripts/run_post900m_master.py --status
```

For a nonwaiting preflight use `--once`. For a certification-only drill against
an already completed predecessor use `--dry-run --state /tmp/post900m-drill.json`.
Neither mode starts training. The production service has no Codex/API dependency.

DATA-D-v4 is a distinct corpus of newly accepted material. Its global dedup index
is seeded from DATA-D-v3; its manifest records the predecessor hash. At 900M the
model, optimizer, scheduler, and RNG lineage continue, but a newly seeded,
deterministic data selector is declared. This is not described as bit-exact data
stream resume.

The production expansion target is 2.25B certified new tokens. This leaves
headroom beyond the 1.5B continuation and a possible later 2.0–2.05B decision;
the latter still requires a new explicit scaling decision.

An integrity failure produces `post900m_handoff.json` and exits 2 without automatic
restart. SIGTERM requests a safe child stop and exits 75 without automatic restart.
No secret is stored in the unit or repository; production publication must use
systemd credentials or the existing Hugging Face credential store.
