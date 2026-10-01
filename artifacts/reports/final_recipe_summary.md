# Final architecture and training recipe

The final BASE model is a 48,636,168-parameter, 12-layer Co4 causal language model. It uses width 552, a 1,728-wide SwiGLU feed-forward path, 6 query heads sharing 2 key/value heads, per-head QK-RMS normalization, a context of 256, a vocabulary of 4,096, and untied input/output embeddings.

The model was trained on 1.9B tokens from DATA-D-v4. Its certified manifest contains 2,259,629,459 available unique tokens. The optimizer is Muon/AdamW hybrid at peak LR 0.0008. Muon covers 44,089,344 parameters (90.65%); AdamW covers 4,546,824 parameters (9.35%). WSD allocates 2% of steps to warmup, 83% to stable peak LR, and 15% to cosine decay.

Final results and model/data/tokenizer hashes are in the [production report](final_production_report.md) and [provenance manifest](../manifests/final_production_provenance.json). The canonical production config is [`../../configs/final_production.json`](../../configs/final_production.json). Checkpoints and exact data/tokenizer files are managed outside Git.
