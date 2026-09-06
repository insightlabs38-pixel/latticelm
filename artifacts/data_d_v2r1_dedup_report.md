# DATA-D-BROAD-v2r1 deduplication report

INDEPENDENT DEDUP VERIFICATION: PASS

The independent verifier recomputed normalized SHA-256, paragraph hashes, and 64-bit SimHash for all 57,304 retained train and validation documents, proved no retained exact/paragraph/near duplicate, and checked all 33,509 replayed DATA-C FineWeb-Edu stable IDs.

Removal counts:

- Exact document: 20
- Exact paragraph: 283
- SimHash near duplicate (Hamming distance at most 3): 5

DATA-C was seeded first and wins every collision. The full SQLite index, normalized source cache, and document audit ledger remain available for certification audit.
