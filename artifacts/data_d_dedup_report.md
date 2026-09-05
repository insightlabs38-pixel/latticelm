# DATA-D-BROAD-v1 deduplication report

Status: NOT RUN ON A PRODUCTION CORPUS

The deterministic implementation supports canonical normalization, exact document SHA-256, exact paragraph matching, 64-bit SimHash candidate matching, and frozen source priority with DATA-C winning collisions. Synthetic tests prove exact-document, paragraph, near-duplicate infrastructure, and processing-order independence.

Production deduplication was not run because the declared mixture is infeasible. The pinned BabyLM source is unchanged and DATA-C already consumed all of its eligible training partition. Any DATA-D BabyLM training document would therefore be a DATA-C duplicate or a frozen validation document.

EXACT DOC REMOVALS: unavailable

PARAGRAPH DEDUP REMOVALS: unavailable

NEAR-DUP REMOVALS: unavailable
