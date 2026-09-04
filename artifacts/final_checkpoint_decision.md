# Phase 7B checkpoint decision

**BEST BASE CHECKPOINT: NOT YET DETERMINED — canonical run stopped at the 1,000,448-token recovery point before any selection milestone.**

**FINAL TRAINING TOKENS: 1,000,448 (incomplete operational attempt; not a final base model).**

**SELECTED DATA REGIME: 75% BabyLM 2026 Strict / 25% decontaminated FineWeb-Edu.**

**SELECTED ARCHITECTURE: untied LatticeLM-Co4-S.**

**100M EXTENSION RECOMMENDATION: NO DECISION.** A 100M extension is prohibited before the 50M gate and, in any case, requires expanding and re-decontaminating the approximately 20M-token FineWeb-Edu sample.

The only new recovery result is common-validation loss 4.562346 / perplexity 95.8080 at 1,000,448 tokens. It cannot support a 10M/25M/50M comparison, a final checkpoint choice, or a 100M marginal-value decision. Blank milestone and benchmark cells are intentional and must not be interpreted as zero.
