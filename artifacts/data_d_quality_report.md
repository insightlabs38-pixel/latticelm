# DATA-D-BROAD-v1 quality report

DATA-D PREPARED: NO

The acceptance gate stopped before corpus materialization. The pinned BabyLM 2026 Strict source has no eligible unique training documents after mandatory global deduplication against DATA-C. DATA-C contains 187,825,322 BabyLM training tokens and used every nonempty source line except the frozen 1% validation tail (1,917,891 tokens). The validation tail was not admitted to training.

Consequently, no source mixture, document-length statistics, tokenizer statistics, quality sample, or production shards are reported. The canonical tokenizer SHA-256 is `4f313ebc481a77e8ad2179cf2d7a3836b28d50773ef9f62ef08831bf076637e5`.
