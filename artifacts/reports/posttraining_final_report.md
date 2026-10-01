# Post-training selection report

## Selection

The final model remains **BASE**, the 1.9B-token pretraining checkpoint. Targeted ranking and joint training raised accuracy on a fixed 256-example reasoning proxy, but that proxy did not predict the full GIBC evaluation. The BASE checkpoint SHA-256 remained `94b91a2d39c2889209cb47b2e04b2c0d307a25610c40c5a124b595020eb762b0` through the original salvage run.

| Candidate | DATA-D loss | DATA-D vs BASE | WikiText BPB | BPB vs BASE | Proxy accuracy | GIBC mean | GIBC vs BASE |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 2.6305 | 1.000× | 1.4135 | 1.000× | 0.2686 | 0.4289 | — |
| BASE + 10M recovery, 5% | 2.6717 | 1.016× | 1.4503 | 1.026× | 0.2764 | 0.4288 | -0.0001 |
| BASE + 10M recovery, 17% | 3.1540 | 1.199× | 1.6879 | 1.194× | 0.3672 | 0.4151 | -0.0138 |
| Mixed objective, 15% | 3.0731 | 1.168× | 1.6421 | 1.162× | 0.3418 | 0.4154 | -0.0135 |

Proxy accuracy is the 256-example screening score. GIBC is the full-suite mean. Lower DATA-D loss and lower WikiText BPB are better. Only the 5%, 17%, and mixed-objective 15% candidates in this table received the additional full GIBC evaluation. The remaining candidates have incomplete broad-evaluation evidence and are not claimed as winners.

## Interpretation

Joint/ranking specialization increased targeted proxy accuracy, but directly specialized finalists showed major language-model retention regressions. Recovery annealing reduced that damage. A small recovery blend stayed near BASE on retention and broad GIBC, with only a small proxy change; larger blends preserved more of the proxy gain but fell below BASE on GIBC. The mixed-objective candidate followed the same tradeoff. This supports a narrow conclusion: targeted proxy performance did not transfer to the full evaluation in these experiments.

The follow-up screened additional blend weights. The 16% blend was the strongest screen-only frontier point, but had no completed GIBC evaluation. The 17% blend's completed GIBC mean was lower than BASE, so BASE remained the only defensible selection under the broad-evaluation gate.

The full local run summaries contain detailed milestone curves, finalist identity, hashes, and run status. Compact comparison data is retained in [`../metrics/final_posttraining_comparison.json`](../metrics/final_posttraining_comparison.json) and the decision in [`../metrics/final_selection.json`](../metrics/final_selection.json).
