# Phase 5 Co4 scaling

All CPU-heavy runs were sequential with two PyTorch threads. S1 used 524,288 tokens. Co4-L and XL used the canonical whole-corpus token stream; the earlier S/M token cache omitted line separators, so their rows are retained but are not used for cross-size selection. This discovered preprocessing mismatch is not hidden.

| model | params | tokens | wall s | tok/s | val loss | ppl | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| S | 8,923,392 | 524,288 | 267.73 | 1,958 | 4.773466 | 118.329 | invalid cross-stream control |
| M | 10,997,280 | 524,288 | 311.12 | 1,685 | 4.747700 | 115.319 | invalid cross-stream control |
| L | 15,949,760 | 524,288 | 412.59 | 1,271 | **4.359533** | **78.221** | advance |
| XL | 22,334,592 | 524,288 | 526.50 | 996 | 4.362815 | 78.478 | prune: slower and worse |
| L | 15,949,760 | 1,048,576 | 824.32 | 1,272 | **3.990510** | **54.082** | winner |

The first defensible diminishing-return boundary is between L and XL: XL adds 6.385M parameters and 27.6% wall time yet is 0.003282 loss worse at equal tokens. Co4-L is the selected scale for further work. FLOPs are intentionally not estimated.
