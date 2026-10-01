# Final production results

## Selected model

The selected model is **BASE**, the final 1.9B-token Co4 causal checkpoint. It has 48,636,168 trainable parameters. Its checkpoint SHA-256 is `94b91a2d39c2889209cb47b2e04b2c0d307a25610c40c5a124b595020eb762b0`.

| Metric | Result |
|---|---:|
| DATA-D validation loss | 2.630494 |
| WikiText-103 perplexity | 21.9298 |
| WikiText-103 bits per byte | 1.413462 |
| HellaSwag accuracy | 0.27096 |
| ARC-Easy accuracy | 0.36785 |
| PIQA accuracy | 0.56692 |
| WinoGrande accuracy | 0.50987 |

The benchmark values come from the final checkpoint evaluation with the corrected joined causal-boundary evaluator. The post-training proxy is not included among these results.

## Recipe and data identity

- Parameters: 48,636,168; trained tokens: 1,900,000,000.
- Certified DATA-D-v4 available tokens: 2,259,629,459.
- Architecture: 12-layer Co4 causal; width 552; FFN 1,728; context 256; vocabulary 4,096; real 6Q/2KV GQA; QK-Norm; untied embeddings.
- Optimizer: Muon/AdamW hybrid at peak LR 0.0008; 90.65% of parameters assigned to Muon and 9.35% to AdamW.
- Schedule: WSD, 2% warmup / 83% stable / 15% cosine decay.
- Validation loss: 2.914503 at 1.5B tokens and 2.630494 at 1.9B tokens.

The tokenizer SHA-256 is `294925d61198dfa0863f91193349f6ebead420d06a3fb650a5122d2264a4ebf7`; the DATA-D-v4 manifest SHA-256 is `9d2811cfb2824b0dd541af89ebfb64403b3ec7143db97d52c4046aafb8eeae7f`. The model weights are not stored in this repository. See [`../manifests/final_production_provenance.json`](../manifests/final_production_provenance.json) for the compact provenance record and [`../metrics/final_pretraining_metrics.json`](../metrics/final_pretraining_metrics.json) for machine-readable metrics.

## Checkpoint selection

The final checkpoint had lower validation loss, WikiText BPB, and higher reported benchmark accuracy than the measured end-stable and mid-decay milestones. The detailed three-checkpoint table is in [`final_checkpoint_comparison.md`](final_checkpoint_comparison.md). Post-training study followed this production evaluation; it retained BASE after no fully evaluated candidate established a broad improvement. See [`posttraining_final_report.md`](posttraining_final_report.md).
