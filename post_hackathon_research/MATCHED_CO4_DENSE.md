# Modern matched Co4 versus dense

All runs use the final 4K tokenizer, certified DATA-D-v4, matched 32×256 batch, QK-Norm, GQA, RoPE, SwiGLU, untied embeddings, Muon/AdamW, and the 2/83/15 WSD schedule. Co4 and dense configurations at each scale differ by less than 0.1% in trainable parameters. Held-out aggregate loss uses the frozen DATA-D source weights. Raw-byte exposure is estimated from certified source-level ratios; it is not an exact per-document count.

| Pair | Co4 held-out | Dense held-out | Co4 BPB | Dense BPB | Co4 train h | Dense train h | Co4 est. raw GB | Dense est. raw GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8M seed 1337 | 3.2943 | 3.2747 | 1.7590 | 1.7473 | 2.45 | 2.60 | 0.327 | 0.327 |
| 8M seed 2026 | 3.2885 | 3.2854 | 1.7480 | 1.7607 | 2.34 | 2.55 | 0.327 | 0.327 |
| 16M seed 1337 | 3.0955 | 3.0974 | 1.6294 | 1.6418 | 6.01 | 5.72 | 0.490 | 0.490 |
| 32M seed 1337 | 3.0678 | 3.0659 | 1.6177 | 1.6197 | 6.96 | 6.76 | 0.327 | 0.327 |

The paired bootstrap uses the same 32 source-validation windows for each pair. Its interval covers that fixed window sample, not model-seed uncertainty. The 8M pair has two seeds; 16M and 32M each have one. Capability screens use fixed first 1,000 examples per task and require a full-suite confirmation before a small point-estimate gain is treated as real.

| Pair | Co4 four-task raw mean | Dense four-task raw mean | Co4 transfer raw mean | Dense transfer raw mean |
|---|---:|---:|---:|---:|
| 8M seed 1337 | 0.4073 | 0.4147 | 0.2108 | 0.2033 |
| 8M seed 2026 | 0.4118 | 0.4120 | 0.2088 | 0.2165 |
| 16M seed 1337 | 0.4105 | 0.4118 | 0.2605 | 0.2585 |
| 32M seed 1337 | 0.4163 | 0.4210 | 0.2273 | 0.2328 |
- **8M seed 1337**: Co4 − dense held-out loss +0.0196; BPB +0.0117; paired-window 95% interval [0.014559610188007355, 0.034857973828911766]; four-task paired item-bootstrap accuracy interval [-0.020006250000000003, 0.005506249999999966]; wall-time ratio 0.944.
- **8M seed 2026**: Co4 − dense held-out loss +0.0031; BPB -0.0127; paired-window 95% interval [0.0007534228265285493, 0.017901642248034477]; four-task paired item-bootstrap accuracy interval [-0.01225, 0.0125]; wall-time ratio 0.918.
- **16M seed 1337**: Co4 − dense held-out loss -0.0019; BPB -0.0124; paired-window 95% interval [-0.01180789079517126, 0.004252386465668677]; four-task paired item-bootstrap accuracy interval [-0.014006250000000001, 0.0115]; wall-time ratio 1.051.
- **32M seed 1337**: Co4 − dense held-out loss +0.0019; BPB -0.0020; paired-window 95% interval [-0.006278814189136029, 0.013256011344492433]; four-task paired item-bootstrap accuracy interval [-0.01775625, 0.008749999999999999]; wall-time ratio 1.029.

Local three-point fits of held-out loss against log trainable parameters use the seed-1337 8M, 16M and 32M endpoints: co4 slope -0.1633; dense slope -0.1505. Different scales have different token budgets, so these are descriptive local trends, not a scaling-law extrapolation.

Token-matched and parameter-matched conclusions follow directly from the table. The 6NT proxy is close for these models but does not include Co4 MOD arithmetic or measured hardware utilization; a precise FLOP-efficiency conclusion requires an operator-count audit. Raw task means above remain screening estimates, with normalized task variants in the JSON evidence.
