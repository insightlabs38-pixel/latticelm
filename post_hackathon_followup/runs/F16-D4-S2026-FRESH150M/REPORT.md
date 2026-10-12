# F16-D4-S2026-FRESH150M

Post-submission research. Updated 2026-10-12T20:29:24.715548+00:00.

Status: COMPLETED. Endpoint checkpoint and evaluation evidence verified.

Role: D4 BPB and source-wise direction replicates at paired seed 2026
Seed: 2026; parameters: 16,004,104; token target: 150,000,000.
Source commit: `c9ed214a5131c2e891f9e77b5ecee74243be4f72`.
Checkpoint SHA256: `24779d081cbe26035783113cab160a687710f4251d6903d3be5de6032538ab13`.
Initialization/schedule: See immutable registration..
Full-Co4: null.


| Metric | Value |
|---|---|
| logical_tokens | 150000000 |
| processed_tokens | 150003712 |
| wall_seconds | 21385.55844819601 |
| tokens_per_second | 7014.079167647517 |
| peak_rss_bytes | 13944508416 |
| heldout_loss | 3.0901934519410132 |
| bits_per_byte | 1.6296956038771668 |
| perplexity | 35.17129617303396 |
| capability_scope | full official task evaluation |

Source losses: {"finemath": 2.9941112995147705, "fineweb": 3.177150160074234, "fineweb_edu": 3.191264659166336, "wikipedia": 2.877132773399353}.
Capability results: {"hellaswag": {"name": "hellaswag", "alias": "hellaswag", "sample_len": 10042, "acc,none": 0.2649870543716391, "acc_stderr,none": 0.004404243675486242, "acc_norm,none": 0.2698665604461263, "acc_norm_stderr,none": 0.0044298311529146}, "arc_easy": {"name": "arc_easy", "alias": "arc_easy", "sample_len": 2376, "acc,none": 0.3354377104377104, "acc_stderr,none": 0.009688175165829564, "acc_norm,none": 0.3181818181818182, "acc_norm_stderr,none": 0.009557408782506306}, "piqa": {"name": "piqa", "alias": "piqa", "sample_len": 1838, "acc,none": 0.544069640914037, "acc_stderr,none": 0.011620422647622046, "acc_norm,none": 0.5195865070729053, "acc_norm_stderr,none": 0.01165686997928841}, "winogrande": {"name": "winogrande", "alias": "winogrande", "sample_len": 1267, "acc,none": 0.5295974743488555, "acc_stderr,none": 0.014027843827840017}}.

Compute/depth claims require the paired comparison reports and their uncertainty; parameter matching does not imply compute matching.
Full-Co4 diagnostic statistics, when available, are in `architecture_diagnostics.json`. Configuration and source hashes are in the immutable run registry and `FOLLOWUP_RUN_SUMMARY.json`.
