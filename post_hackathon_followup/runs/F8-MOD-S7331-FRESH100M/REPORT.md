# F8-MOD-S7331-FRESH100M

Post-submission research. Updated 2026-10-12T20:29:24.721759+00:00.

Status: COMPLETED. Endpoint checkpoint and evaluation evidence verified.

Role: Fresh paired MOD control for Differential replication
Seed: 7331; parameters: 7,991,800; token target: 100,000,000.
Source commit: `c9ed214a5131c2e891f9e77b5ecee74243be4f72`.
Checkpoint SHA256: `54394ce6b44096624f4ce8b25ffef44a7ce6d849bf784e20fbe5919b87048e0e`.
Initialization/schedule: See immutable registration..
Full-Co4: null.


| Metric | Value |
|---|---|
| logical_tokens | 100000000 |
| processed_tokens | 100007936 |
| wall_seconds | 8165.242968100996 |
| tokens_per_second | 12247.032989791995 |
| peak_rss_bytes | 12593455104 |
| heldout_loss | 3.2902163937687874 |
| bits_per_byte | 1.7591326049136025 |
| perplexity | 46.665174436519656 |
| capability_scope | full official task evaluation |

Source losses: {"finemath": 3.25666743516922, "fineweb": 3.345765918493271, "fineweb_edu": 3.4165841937065125, "wikipedia": 3.0317343175411224}.
Capability results: {"hellaswag": {"name": "hellaswag", "alias": "hellaswag", "sample_len": 10042, "acc,none": 0.26319458275243973, "acc_stderr,none": 0.004394671271021473, "acc_norm,none": 0.27106154152559253, "acc_norm_stderr,none": 0.004435993492583661}, "arc_easy": {"name": "arc_easy", "alias": "arc_easy", "sample_len": 2376, "acc,none": 0.2967171717171717, "acc_stderr,none": 0.009373559492986735, "acc_norm,none": 0.2988215488215488, "acc_norm_stderr,none": 0.009392656275408704}, "piqa": {"name": "piqa", "alias": "piqa", "sample_len": 1838, "acc,none": 0.5473340587595212, "acc_stderr,none": 0.01161343165087302, "acc_norm,none": 0.5239390642002176, "acc_norm_stderr,none": 0.01165244562107948}, "winogrande": {"name": "winogrande", "alias": "winogrande", "sample_len": 1267, "acc,none": 0.5019731649565904, "acc_stderr,none": 0.014052376259225643}}.

Compute/depth claims require the paired comparison reports and their uncertainty; parameter matching does not imply compute matching.
Full-Co4 diagnostic statistics, when available, are in `architecture_diagnostics.json`. Configuration and source hashes are in the immutable run registry and `FOLLOWUP_RUN_SUMMARY.json`.
