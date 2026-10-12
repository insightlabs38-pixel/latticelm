# F32-DENSE-S1337-FRESH300M

Post-submission research. Updated 2026-10-12T20:29:24.709046+00:00.

Status: COMPLETED. Endpoint checkpoint and evaluation evidence verified.

Role: Fresh 300M dense trajectory closes MOD-Co4 vs dense at matched seed/data/optimizer/schedule
Seed: 1337; parameters: 31,986,396; token target: 300,000,000.
Source commit: `c9ed214a5131c2e891f9e77b5ecee74243be4f72`.
Checkpoint SHA256: `53725ce3a0a45ee5bd95d51e56c5a7f6dbc7ef0b31cd596b66aa93d8aa1ab1c4`.
Initialization/schedule: fresh random initialization; empty optimizers; sampler cursor zero; no warm-start.
Full-Co4: null.


| Metric | Value |
|---|---|
| logical_tokens | 300000000 |
| processed_tokens | 300007424 |
| wall_seconds | 72821.86579758799 |
| tokens_per_second | 4119.641768502128 |
| peak_rss_bytes | 15158497280 |
| heldout_loss | 2.8940543055534365 |
| bits_per_byte | 1.5233220681494901 |
| perplexity | 27.878251018248683 |
| capability_scope | full official task evaluation |

Source losses: {"finemath": 2.861763060092926, "fineweb": 2.9320915937423706, "fineweb_edu": 3.020929753780365, "wikipedia": 2.6481651663780212}.
Capability results: {"hellaswag": {"name": "hellaswag", "alias": "hellaswag", "sample_len": 10042, "acc,none": 0.26687910774746065, "acc_stderr,none": 0.004414246720076327, "acc_norm,none": 0.27673770165305717, "acc_norm_stderr,none": 0.00446471349368615}, "arc_easy": {"name": "arc_easy", "alias": "arc_easy", "sample_len": 2376, "acc,none": 0.3425925925925926, "acc_stderr,none": 0.009738105469984307, "acc_norm,none": 0.32323232323232326, "acc_norm_stderr,none": 0.009597218642045435}, "piqa": {"name": "piqa", "alias": "piqa", "sample_len": 1838, "acc,none": 0.5669205658324266, "acc_stderr,none": 0.011560864423151443, "acc_norm,none": 0.5375408052230686, "acc_norm_stderr,none": 0.011632896120570703}, "winogrande": {"name": "winogrande", "alias": "winogrande", "sample_len": 1267, "acc,none": 0.5043409629044988, "acc_stderr,none": 0.014051956064076884}}.

Compute/depth claims require the paired comparison reports and their uncertainty; parameter matching does not imply compute matching.
Full-Co4 diagnostic statistics, when available, are in `architecture_diagnostics.json`. Configuration and source hashes are in the immutable run registry and `FOLLOWUP_RUN_SUMMARY.json`.
