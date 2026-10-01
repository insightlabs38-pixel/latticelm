# Training

## DATA-D-v4

The final run used 1.9B tokens from DATA-D-v4. Its certified manifest reports 2,259,629,459 available unique tokens, leaving headroom beyond the consumed budget. The recorded source mixture is approximately 10.07% FineMath, 20.11% FineWeb, 44.83% FineWeb-Edu, and 25.00% Wikipedia. The training data pipeline freezes source revisions, filters and deduplicates documents, removes benchmark contamination, tokenizes the accepted corpus, and emits shard and manifest hashes. Dataset shards and tokenizer files are local generated inputs and are not versioned here.

The final provenance manifest records the dataset-manifest SHA-256 and tokenizer SHA-256. Reproducing the exact run requires matching both hashes, not merely using sources with the same names.

## Optimizer and schedule

The final configuration uses a Muon/AdamW hybrid at peak learning rate 0.0008. Muon is assigned to eligible hidden two-dimensional matrices; AdamW owns embeddings, output head, norms, and learned latents. Their parameter shares are 90.65% and 9.35%. The schedule is warmup-stable-decay (WSD): 2% linear warmup, 83% stable learning rate, and 15% cosine decay. Training uses batch size 32 at context 256 and gradient clipping at 1.0.

Validation loss moved from 2.914503 at 1.5B tokens to 2.630494 at 1.9B. This late improvement is reported against the final production validation protocol, not the post-training reasoning proxy. The selected checkpoint has SHA-256 `94b91a2d39c2889209cb47b2e04b2c0d307a25610c40c5a124b595020eb762b0`.

## Run and checkpoint policy

The canonical recipe is [`../configs/final_production.json`](../configs/final_production.json). The production trainer consumes an already built DATA-D-v4 manifest and tokenizer, saves restart-safe checkpoints under `artifacts/final_production_run/`, and emits a curve, events, and state for operations. Those run outputs are excluded from Git. Selected metrics and hashes are preserved in [`../artifacts/reports/final_production_report.md`](../artifacts/reports/final_production_report.md) and the curated provenance manifest.

See [Reproducibility](REPRODUCIBILITY.md) for setup and verified command interfaces.
