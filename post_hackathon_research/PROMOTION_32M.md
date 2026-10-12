# 32M promotion tournament

The 32M control and up to two independently selected mechanisms use fresh 300M-token trajectories, the same frozen 4K tokenizer, 32×256 scientific batch, modern Muon/AdamW recipe, and common evaluator. Mechanisms were selected from 16M evidence before 32M training. Raw-token, 6NT active-parameter FLOP proxy, and wall-clock views are separate.

| Arm | 16M parent mechanism | Held-out loss | WikiText BPB | Four-task raw accuracy |
|---|---|---:|---:|---:|
| R32-CO4-CONTROL | control | 2.90383 | 1.52771 | 0.4300 |

The full registry, token curves, source-wise losses, raw capability samples, checkpoint hashes, and measured wall times are archived with each arm. Small capability movements require paired uncertainty analysis before any promotion claim.
