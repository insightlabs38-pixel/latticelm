# Post-MOD value residual Co4

The bounded 16M screen mixes the first block's post-MOD V with each later block's post-MOD V, initially at 0.5/0.5. The causal GQA attention backbone, tokenizer, data order, optimizer and 150M-token WSD schedule match the modern Co4 control. No cross-layer skip or other architecture change is in this first screen.

| Measure | Co4 control | Value residual | Change |
|---|---:|---:|---:|
| Frozen-mixture held-out loss | 3.09548 | 3.08910 | -0.21% |
| WikiText-103 BPB | 1.62941 | 1.63979 | +0.64% |
| Four-task raw accuracy | 0.4105 | 0.4143 | +0.0038 |

The four-task paired item-bootstrap 95% interval for value residual minus control is [-0.0095, 0.01675]; it does not include training-seed uncertainty. Source-wise losses, normalized task variants, raw samples and checkpoint hashes are in the run archives. A one-scalar long-skip follow-up is separately evidence/time gated after the mandatory and named stretch work.
