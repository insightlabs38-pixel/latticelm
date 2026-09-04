# Phase 7B `torch.compile` pilot

The frozen Co4-S workload (batch 8, context 128, two CPU threads) was measured for two warmup and five timed optimizer steps. Both branches began from identical weights and synthetic token batches. The first measured post-update loss was exactly equal (8.05167007446289; absolute difference 0.0).

| mode | mean step seconds | tokens/s | peak RSS |
|---|---:|---:|---:|
| eager | 0.526001 | 1,946.76 | not separately isolated |
| `torch.compile` | 0.481520 | 2,126.60 | 1,508,782,080 B (whole pilot) |

The measured ratio is 1.0924x, but this bounded pilot did not isolate gradients tensor-by-tensor or memory by branch. It is therefore preparation evidence, **not** authorization to mutate the eager canonical lineage and not a final speed claim.
