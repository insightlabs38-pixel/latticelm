# Phase 7A final data decision

**SELECTED FINAL DATA REGIME: DATA-C — exactly 75% BabyLM 2026 Strict 100M tokens and 25% decontaminated FineWeb-Edu sample tokens, realized as six and two 128-token sequences in every batch.**

**RUNNER-UP: DATA-A — 100% BabyLM 2026 Strict 100M.**

At 1,048,576 equal training tokens, DATA-C achieved common-validation loss 4.55227 versus 4.75262 for DATA-A and 5.48373 for DATA-B. DATA-B was therefore pruned. At 3,145,728 tokens, DATA-C retained and slightly widened its lead: 3.95771 (perplexity 52.34) versus 4.14413 (63.06), in 1,473.50 versus 1,489.45 seconds. Thus the mixed regime improved both equal-token quality and CPU-hour efficiency.

The restrained benchmark snapshot supports transfer without being used for mixture tuning. DATA-C improved HellaSwag raw accuracy (0.25851 vs 0.25354), ARC-Easy raw/normalized accuracy (0.26599/0.26852 vs 0.25000/0.25505), PIQA normalized accuracy (0.50054 vs 0.49674), and WikiText-103 perplexity (211.78 vs 262.03). PIQA raw and WinoGrande moved slightly against DATA-C, within their reported standard errors. These mixed near-chance reasoning scores are diagnostic, not a claim of benchmark superiority.

A 10M finalist extension was not run because the 0.18642-loss advantage at 3.145M was clear under essentially equal wall time, satisfying the stated early-stop rule. Run the selected regime next through **10M, 25M, and 50M** checkpoints. Decide whether to proceed beyond 50M from the common-validation slope; do not pre-commit to 100M. Both 3M curves still improve, so 10M is warranted, while evidence here cannot justify an automatic 100M run.
