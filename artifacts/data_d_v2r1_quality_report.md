# DATA-D-BROAD-v2r1 quality report

DATA-D-v2r1 PREPARED: YES

TOTAL CLEAN UNIQUE TRAIN TOKENS: 100,011,495

TOTAL RETAINED TRAIN DOCUMENTS: 56,135

REALIZED FINEWEB-EDU: 49.994801%

REALIZED WIKIPEDIA: 25.006893%

REALIZED FINEWEB: 24.998305%

The canonical 4,096-token byte-level BPE hash is `4f313ebc481a77e8ad2179cf2d7a3836b28d50773ef9f62ef08831bf076637e5`. Eight immutable little-endian int32 training shards and five validation shards passed checksum, byte-size, tokenizer, boundary, stable-ID, and read-only mmap checks. Loss-bearing training batches are exactly 50/25/25.

The independently auditable normalized source cache and per-document decision ledger are retained at `artifacts/data/phase7f_v2r1/canonical-100m/` and are deliberately excluded from Git.
