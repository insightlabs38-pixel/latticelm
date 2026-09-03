# Phase 5 GIBC snapshot — blocked by absent checkpoint objects

Authentication to the verified private repository succeeded on repeated attempts. Repository listings and downloads exposed Dense Round 3, tied Co4 Round 3, and Co4 seed-2026, but not the required parameter-matched Dense or untied-Co4 snapshots. Those two trained objects are also absent locally. A four-way competition-compatible snapshot therefore cannot be made without retraining completed Phase 4 controls. No GIBC task was queried and no score is reported or used for HPO.

Planned frozen settings are zero-shot, batch 8, full HellaSwag/ARC-Easy/PIQA/WinoGrande and WikiText-103, with an exact `lm-evaluation-harness` commit recorded. The CSV distinguishes available remote checkpoints from missing ones. This blocker must be resolved before the provisional architecture freeze becomes final.
