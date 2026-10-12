# Developmental experts

All arms use the 4K tokenizer, eight-block Co4 backbone, top-one routing, one active FFN expert per token and block, 100M logical tokens, the same 25% difficulty-curriculum transition, and a registered FineMath source increase from about 10% to 20%. F0 starts with eight experts; F1–F3 start with four and expand to eight at the transition. F2/F3 clone mature parents, and F3 directs FineMath tokens toward the children for 5M tokens within the fixed budget.

| Arm | Held-out loss | WikiText BPB | Hours | Domain/expert MI (nats) | Expert utilization entropy |
|---|---:|---:|---:|---:|---:|
| F0-8M-FIXED-EXPERTS | 3.22777 | 1.72992 | 3.86 | 0.0017 | 2.0794 |
| F1-8M-NAIVE-BIRTH | 3.25197 | 1.71903 | 3.73 | 0.0079 | 2.0510 |
| F2-8M-CLONE-BIRTH | 3.30691 | 1.75028 | 3.72 | 0.0022 | 2.0589 |
| F3-8M-CLONE-CATCHUP | 3.31777 | 1.75580 | 3.86 | 0.0050 | 2.0609 |

The routing log contains per-step token fractions by expert and domain, router entropy, load-balancing loss, and catch-up status. Domain/expert mutual information describes attribution; a positive quality difference is required before claiming that specialization helped. The active-parameter count excludes dormant experts, while the registered total count includes all eight. The 6NT active-parameter proxy omits routing and irregular-dispatch overhead, so wall time remains a separate fairness axis.

F3 parent/child mean output cosine on actual FineMath layer activations: 0.4217; mean relative parameter distance: 1.0031. The first-router FineWeb-Edu versus FineMath gradient cosine was 0.07442399114370346. These are fixed-window endpoint diagnostics, not population estimates.

For every arm, `expert_diagnostics.json` records source-wise forced-expert loss, one-expert ablation effects, parent/child route-swap counterfactuals, layer-level representation similarity, parameter distance, and one domain gradient-conflict estimate.
