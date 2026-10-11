# Historical 8M control provenance recovery

The bounded recovery pass resolved all four controls. Each run was certified separately; no fresh 100M replacement run is required. Historical files were read only. The active dense trainer continued without interruption.

## Exact source recovery

- Local Git: 1,502 objects enumerated; 1,004 blobs up to 512 KiB hashed, including unreachable objects. All nine current remote heads fetched to isolated provenance-audit refs; no additional historical post-hackathon branch existed.
- Original local archive recovered trainer `84111597bd74d89abcb23b1da336dbc12f0749abc7261822360a935a291e0891`.
- The latest source archive had overwritten older versions. Historical receipts identified the exact old HF revisions; downloads were checked against the recorded archive SHA256 before extracting any source.
- Revision `8c4f2508cb408bbc44931b25579ecc1e7fb9d2df`, archive SHA `de1bd2d9835acc03411c9afefae3a8991b3b2faa4ff9890db64e1919a84e1180`, recovered trainer `65bcf8ffcf59e19eef73f78dffefe9cf2fed81e734366973a4b4a9289520f121`.
- Revision `23ea388b563137f3a8e756baef590e11611a9099`, archive SHA `76f45e4ce625944da03a5821ec4c23e84d167ac7b96b504da902df21a828c7f6`, recovered trainer `58e2387640395864abd49e70e6bf188e20ae449afd064ede0edb5d8157792f87`.

## Per-run result

| Control | Executed trainer | Outcome |
|---|---|---|
| A8-CO4-S1337 | `84111597bd74d89abcb23b1da336dbc12f0749abc7261822360a935a291e0891` | Certified historical primary comparator |
| A8-DENSE-S1337 | `65bcf8ffcf59e19eef73f78dffefe9cf2fed81e734366973a4b4a9289520f121` | Certified historical primary comparator |
| A8-CO4-S2026 | `58e2387640395864abd49e70e6bf188e20ae449afd064ede0edb5d8157792f87` | Certified historical primary comparator |
| A8-DENSE-S2026 | `58e2387640395864abd49e70e6bf188e20ae449afd064ede0edb5d8157792f87` | Certified historical primary comparator |

## Dense seed-1337 discrepancy

Both exact snapshots were recovered. The launch snapshot `65bc` writes `trainer_sha256: sha256(Path(__file__))` at checkpoint time. This hashes the disk file again, rather than preserving the source loaded at process start. The disk file became `58e` during the uninterrupted run.

The final checkpoint contains all and only the recovered `65bc` payload fields, and lacks the new `58e` best-checkpoint and adaptive-birth fields. The single launch records `65bc`, the event history contains no resume, and the recovered program has no dynamic trainer reload. Together with the exact source/archive chain and full trajectory checks, this explains the discrepancy without assuming two unknown sources are equivalent.

MOD seed 1337 uses the original registered `8411` snapshot. The original trainer intentionally omitted trainer hashes in both launch/checkpoint records. Its checkpoint schema matches exactly, and the source amendment explicitly records that the first active run retained the loaded original trainer.

## Scientific checks and practical limits

Every run passed actual config/hash pairing, exact parameter count, source/tokenizer and all canonical training/validation shard hashes, optimizer partition, contiguous 12,208 steps, one 100M WSD with zero LR error, whole-run source order, sampler cursor/draw accounting and exact saved next-batch replay. The first three fresh losses matched exactly for all four controls, using recovered snapshot implementations in isolated one-thread processes. No model weights from these audit probes were saved.

The old optimizer partition source only differs by a later router-name exclusion. Actual baseline partitions match exactly; optimizer construction and WSD function ASTs are identical. Shared mathematical model/config/data/memory/Co4/sampler bytes match exactly. Earlier validation left eval mode active, but these dropout-zero, memory-disabled baseline outputs are exactly invariant under train/eval mode.

There is no initial production checkpoint. Certification combines the recovered exact source chain, launch/amendment and payload evidence, complete schedule/data-order records, next-batch reconstruction and short fresh replay. Historical 1000-example capability screens will be replaced by common full endpoint evaluations before primary claims. Seed uncertainty and historical host wall-time limitations still apply.

## Execution policy

`HISTORICAL_8M_CONTROL_POLICY.json` records the independently certified selections. The supervisor imports/evaluates them only when C2 promotion or the second-seed gate requires them. An unresolved per-run certificate would select only that run for a fresh fallback. Mixed historical/fresh comparator names use a stable matched-recipe key for two-seed consistency.

63 focused regression tests passed in 3.22 seconds, including per-control routing, source/certificate mutation rejection, discrepancy payload parsing and mixed-source seed pairing. Exact snapshot sources, certificates and original archive revisions are retained for migration.
