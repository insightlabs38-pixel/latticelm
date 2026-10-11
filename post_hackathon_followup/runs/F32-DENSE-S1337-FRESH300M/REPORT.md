# F32-DENSE-S1337-FRESH300M

Post-submission research. Updated 2026-10-11T03:48:07.934720+00:00.

Status: RUNNING. Incomplete; no final scientific conclusion.

Role: Fresh 300M dense trajectory closes MOD-Co4 vs dense at matched seed/data/optimizer/schedule
Seed: 1337; parameters: 31,986,396; token target: 300,000,000.
Source commit: `c9ed214a5131c2e891f9e77b5ecee74243be4f72`.
Checkpoint SHA256: `None`.
Initialization/schedule: fresh random initialization; empty optimizers; sampler cursor zero; no warm-start.
Full-Co4: null.


| Metric | Value |
|---|---|
| logical_tokens | 28803072 |
| processed_tokens | 0 |
| wall_seconds | None |
| tokens_per_second | None |
| peak_rss_bytes | None |
| heldout_loss | 3.5002628505229954 |
| bits_per_byte | None |
| perplexity | None |
| capability_scope | None |

Source losses: {"finemath": 3.4396474361419678, "fineweb": 3.5722316205501556, "fineweb_edu": 3.630252420902252, "wikipedia": 3.2329527735710144}.
Capability results: null.

Compute/depth claims require the paired comparison reports and their uncertainty; parameter matching does not imply compute matching.
Full-Co4 diagnostic statistics, when available, are in `architecture_diagnostics.json`. Configuration and source hashes are in the immutable run registry and `FOLLOWUP_RUN_SUMMARY.json`.
