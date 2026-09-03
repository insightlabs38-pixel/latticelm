# Phase 5 report

## Execution and integrity

The checkout and private Hugging Face repository were inspected repeatedly. Authentication succeeded, but the asserted Phase 4 untied-Co4 and parameter-matched checkpoints and four named Phase 4 artifacts were absent. Dense/tied Round-3 and Co4 seed-2026 snapshots were present. This is a specific remote-content blocker, not an authentication failure; benchmark values are not invented.

## Results

Co4-L (15.950M) wins scaling. It reached 4.359533/78.221 at 524K tokens and 3.990510/54.082 at 1.049M. Co4-XL was dominated at S1. Five 262K-token HPO trials were completed; 4e-4, weight decay 0.05, warmup 0.01, betas (0.85, 0.98), clipping 1.0 and cosine decay won Round 1 at 4.881274/131.799. The planned promotion was halted before a comparable completion, so the constant-3e-4 optimizer remains frozen rather than selecting from insufficient evidence.

The SLlama primary paper was read. `SLlamaInspiredExperimental` truthfully implements RRHP, SPMLP and whole-layer sharing but omits underspecified PWA; it is not called faithful. No trained result is claimed.

## Decisions

1. Best validation model: verified Phase 4 untied Co4 (3.516811), pending checkpoint recovery.
2. Best reasoning model: undetermined because the required checkpoints are absent.
3. Best measured quality/CPU-hour among valid scaling runs: Co4-L.
4. Best parameter efficiency: Co4-L; XL adds parameters without quality.
5. Best overall GIBC candidate: undetermined.
6. Strongest challenger: SLlama-inspired, implementation-only.
7. Frozen provisional final architecture: untied Co4-L, AdamW, 15.950M parameters, 4K pinned tokenizer, fixed context 128. The freeze is provisional because GIBC transfer could not be run.

The Co4-L S2 checkpoint was checksum-verified and persisted privately at `experiments/co4-l-s2` (revision `793104f0907066c6cd56ace980159c00e3fa8f6b`). Context curriculum, final GIBC, and matrix analysis are blocked by the missing incumbent checkpoint or downstream selection gate; their artifacts state this explicitly. No benchmark task was repeatedly queried, no data mixture was changed, and no Triton speedup is claimed.
