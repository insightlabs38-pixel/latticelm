# F8-DIFF-S7331-FRESH100M

Post-submission research. Updated 2026-10-12T20:29:24.734009+00:00.

Status: COMPLETED. Endpoint checkpoint and evaluation evidence verified.

Role: Differential operator replicates under one matched fresh seed
Seed: 7331; parameters: 7,992,792; token target: 100,000,000.
Source commit: `c9ed214a5131c2e891f9e77b5ecee74243be4f72`.
Checkpoint SHA256: `8f8bd19b1ece212dbe65d62c2792297f58bb495ccac06668f9201e99346788d4`.
Initialization/schedule: See immutable registration..
Full-Co4: null.


| Metric | Value |
|---|---|
| logical_tokens | 100000000 |
| processed_tokens | 100007936 |
| wall_seconds | 9798.901962403004 |
| tokens_per_second | 10205.22507355271 |
| peak_rss_bytes | 12850167808 |
| heldout_loss | 3.3172667369246485 |
| bits_per_byte | 1.7883241180446954 |
| perplexity | 49.73803280361839 |
| capability_scope | full official task evaluation |

Source losses: {"finemath": 3.2863464653491974, "fineweb": 3.3803328573703766, "fineweb_edu": 3.4474030435085297, "wikipedia": 3.0449365973472595}.
Capability results: {"hellaswag": {"name": "hellaswag", "alias": "hellaswag", "sample_len": 10042, "acc,none": 0.2658832901812388, "acc_stderr,none": 0.00440899486864994, "acc_norm,none": 0.2713602867954591, "acc_norm_stderr,none": 0.0044375277328504974}, "arc_easy": {"name": "arc_easy", "alias": "arc_easy", "sample_len": 2376, "acc,none": 0.3047138047138047, "acc_stderr,none": 0.009444871667360117, "acc_norm,none": 0.29545454545454547, "acc_norm_stderr,none": 0.0093619871265564}, "piqa": {"name": "piqa", "alias": "piqa", "sample_len": 1838, "acc,none": 0.5331882480957563, "acc_stderr,none": 0.011640096923563308, "acc_norm,none": 0.5168661588683352, "acc_norm_stderr,none": 0.011659185184878962}, "winogrande": {"name": "winogrande", "alias": "winogrande", "sample_len": 1267, "acc,none": 0.4996053670086819, "acc_stderr,none": 0.01405248130604961}}.

Compute/depth claims require the paired comparison reports and their uncertainty; parameter matching does not imply compute matching.
Full-Co4 diagnostic statistics, when available, are in `architecture_diagnostics.json`. Configuration and source hashes are in the immutable run registry and `FOLLOWUP_RUN_SUMMARY.json`.
