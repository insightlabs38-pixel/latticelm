# WSD next-phase decision

1. **Best checkpoint:** 850M WSD (30765620d4cf36715c82d97b5826be10b5caa165747db040d7a0dedd9a20b5a2).
2. **Material advantage:** WikiText improves with horizon, while official-task results cross: the intermediate endpoint leads ARC-Easy/WinoGrande and the late endpoint leads PIQA. Paired uncertainty does not establish broad late-horizon superiority.
3. **Did WSD solve the divergence?** WSD materially repaired WikiText for every branch, but did not create a broad monotonic official-task ordering.
4. **Does later stable training remain useful?** Not clearly across the competition surface.
5. **Further 32M scaling authorized?** false.
6. **Post-training base:** 850M WSD; immutable checkpoint at `/home/insightlabs38/latticelm/artifacts/checkpoints/base_750m-wsd-100m/milestone.pt`.
7. **Architecture timing:** Reconsider immediately after the bounded post-training tournament; earlier if no SFT candidate promotes.
8. **DATA-D-v4 needed?** false.
9. **Exact next phase:** BOUNDED_POSTTRAINING_FROM_CHEAPEST_PARETO_BASE.
10. **Change condition:** Reclassify only in a new revision if checkpoint/evaluation evidence changes, or if bounded matched follow-up establishes a clear material Pareto winner.
