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

## Scaling and saturation response

The scaling classification is immutable and deterministic. `CONTINUE_TO_1_5B`
builds and certifies DATA-D-v4 before continuing the constant-LR trunk.
`INCONCLUSIVE` stops for a strategic decision and never claims 1.5B was run.

`SATURATING_OR_LOW_VALUE` enters the separate restart-safe
`scripts/run_saturation_response.py` controller. It preserves `BASE_512M`,
`BASE_750M`, and `BASE_900M`, and runs an additional 100M tokens with cosine
decay from `3e-4` to zero from all three. Each branch is first compared with its own parent; promoted winners
are then compared globally. This captures the best measured WikiText checkpoint,
an intermediate point, and the most DATA-D-trained point. The controller also
benchmarks B8/B16/B32 and a larger compiled
loss graph on the actual 32.68M model, and conditionally tests conservative SFT
using only deterministic, verifier-backed LatticeReason examples. Every promoted
candidate must pass the complete WikiText/GIBC surface and explicit retention
gates. No SFT run is mislabeled as RLVR; an uncertified RL policy worker is
skipped explicitly. DPO stays optional and requires stored verifier evidence.

Tournament state is isolated under `artifacts/saturation_response_*`. Run a
non-training drill with an isolated state:

```bash
.venv/bin/python scripts/run_saturation_response.py --dry-run \
  --state /tmp/latticelm-saturation-drill/state.json
```

Neither master invokes Codex, an external LLM judge, or a pretrained model.
