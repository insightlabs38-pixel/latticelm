# Follow-up input audit

This is post-submission research. No follow-up experiment has been launched.

Created UTC: 2026-10-11T01:06:39.315387+00:00

Historical main: `43832b4c7bfe8be007ca6b9d1042aa2041c7f280`. Frozen model, tokenizer, and data manifest hashes match the historical freeze record. Historical metrics and controller files are unchanged.

The companion JSON records SHA-256 hashes, sizes, configurations, receipts, and resume-state presence. It is a read-only snapshot. Hash verification does not establish causal correctness or exact sampler replay; those tests remain pending.

| Experiment | Tokens | SHA-256 |
|---|---:|---|
| A32-DENSE-S1337 | 100000000 | `03efcec3267bb73e823d7745f3bfc3c81a006f6f253f006c4e4778fc143f6a43` |
| A32-CO4-S1337 | 100000000 | `a851c2b767e885a174fda8ab5f7fe0bb84cc0d0f8f0a7514a1b5033fba25c2dc` |
| R32-CO4-CONTROL | 300000000 | `71f999a0ee3c0f32b8b8eda8e1ecb9a3e404e24499dcebe4908fb10f8f4c7f06` |
| R16-S2026-CONTROL | 150000000 | `86be653ef976397aebf4c4c52e7680d535e31516d5a892d5653a6329a49732ae` |
| D4-16M-ANNEAL | 150000000 | `8116e957ec7875c579f4458e9fca21c10c87d7976ec775768b0c3a203f23e583` |
| S2-8M-DIFFCO4 | 50000000 | `d5a9e0a03e0123534a183fb58fb01e34ae65b3f2746ad352dd5dda4338b42bab` |
| A8-CO4-S1337 | 100000000 | `a01b837828e05c4c211b86e6b70d37f56a0a478f203bf8e8b7f39270be90f966` |
| A8-CO4-S2026 | 100000000 | `6df3154e5a73d157ffebd06e5708422ce669106539f693b1ca1d70afa35e4e5b` |

## Scientific blocker

A32-DENSE-S1337 and A32-CO4-S1337 completed 12,208 steps against a 100M-token horizon. Both optimizer groups have LR 0.0. R32-CO4-CONTROL was a fresh 300M run (36,622 steps). The configs agree apart from the dense/Co4 architecture. At step 12,208, the 300M WSD schedule is still at peak LR. Changing the horizon now cannot undo the dense warmup and decay history; it would restart LR from zero to peak. It is not a matched uninterrupted 300M trajectory.

User design decision required: a fresh 300M dense run matched to R32-CO4-CONTROL (recommended), or a newly registered paired extension of both 100M checkpoints with identical explicit extension schedules. Either must be labeled as a new experiment.

JSON tuple/list serialization differences were normalized and all audited checkpoint configs match registrations. Optimizer and RNG state are present. The first check reported raw dict mismatch due solely to tuple/list representation; no scientific config mismatch was found.

Old FINAL_RESEARCH_SYNTHESIS.md says final RUNNING, whereas master_state.json says finalized and COMPLETED. This discrepancy is documented here without editing historical files.

Initial disk availability is approximately 14 GB. Remote HF delta inventory, source audit, next-batch replay, full-Co4 implementation, tests, controller, evaluations, and archive closeout remain pending. VM metadata identifies c4a-standard-16 and HYPERDISK-BALANCED; no exact expiry was exposed in the inspected metadata. User deadline clarification is pending.

The frozen record identifies production source commit 6e2c6f0f6f5278a2991c04708859baae7580368c; the exact submitted Git SHA has not yet been independently established. No submission tag was created.
