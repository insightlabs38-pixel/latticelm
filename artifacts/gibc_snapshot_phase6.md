# Phase 6 GIBC architecture-decision snapshot

## Protocol

This was a **single**, zero-shot architecture-decision snapshot, not an HPO loop. Multiple-choice tasks used `lm-evaluation-harness` 0.4.13 task defaults and the native adapter in `scripts/evaluate_gibc.py`; WikiText-103 used the harness-compatible native rolling scorer in `scripts/evaluate_wikitext103.py` on `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, test split. All scoring used the pinned tokenizer (SHA-256 `4f313ebc481a77e8ad2179cf2d7a3836b28d50773ef9f62ef08831bf076637e5`), per-document BOS, context 128, batch 8, and two PyTorch threads. The CSV records checkpoint hashes, wall times, variants, and the Dense model's disclosed shorter 1M training budget.

## Results

| model | train tokens | HellaSwag acc / norm | ARC-Easy acc / norm | PIQA acc / norm | WinoGrande acc | WikiText-103 ppl |
|---|---:|---:|---:|---:|---:|---:|
| Co4-S | 3,145,728 | 0.25423 / **0.24577** | 0.24790 / 0.25126 | 0.51687 / **0.50163** | 0.49882 | 263.307 |
| Co4-L | 3,145,728 | **0.25493** / 0.24407 | **0.25463** / 0.26010 | 0.52448 / 0.50054 | **0.50355** | **245.850** |
| Dense-16M | 1,048,576 | 0.25393 / 0.24427 | 0.24832 / **0.26641** | **0.52720** / 0.49347 | 0.49171 | 594.718 |

Co4-L transfers its language-modeling advantage strongly to WikiText-103 (6.63% lower perplexity than Co4-S). Its raw accuracy is slightly higher on all four reasoning tasks, but changes are tiny relative to the reported per-task standard errors; normalized HellaSwag and PIQA are slightly worse. This is not compelling evidence of a reasoning-capability increase. Dense is not an equal-token 3M reasoning comparison and is included only because it was scientifically pruned at 1M after clear validation domination.

The snapshot therefore supports the Co4 family over Dense, but it does **not** make Co4-L's additional capacity competition-decisive.
