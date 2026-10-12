# Research-D data, curriculum, and WSD annealing

All arms reuse certified DATA-D-v4 token shards and the same 16M Co4 backbone, tokenizer, optimizer and 150M-token horizon. D0 is the modern matched Co4 control. D1 changes only source mixture; D2 adds a 12.5% within-source capability-selected component from SciQ/CommonsenseQA/QASC train-only selector features; D3 biases early sampling by the registered two-signal difficulty score; D4 changes the final WSD-decay mixture and FineWeb-Edu quality floor; D5 combines early curriculum and late anneal only if its frozen gate passes.

| Arm | Weighted held-out loss | WikiText BPB | Four-task raw | Transfer raw | Worst source loss |
|---|---:|---:|---:|---:|---:|
| A16-CO4-S1337 | 3.09548 | 1.62941 | 0.4105 | 0.2605 | 3.23325 |
| D1-16M-REBALANCED-S1337 | 3.08879 | 1.64143 | 0.4210 | 0.2435 | 3.20195 |
| D2-16M-CAPABILITY | 3.08963 | 1.62981 | 0.4173 | 0.2390 | 3.21435 |
| D3-16M-CURRICULUM | 3.09946 | 1.64273 | 0.4125 | 0.2253 | 3.23283 |
| D4-16M-ANNEAL | 3.09604 | 1.62169 | 0.4118 | 0.2430 | 3.20723 |

## Held-out loss by source

| Arm | FineWeb-Edu | FineWeb | Wikipedia | FineMath |
|---|---:|---:|---:|---:|
| A16-CO4-S1337 | 3.23325 | 3.13011 | 2.83967 | 3.04579 |
| D1-16M-REBALANCED-S1337 | 3.20195 | 3.08635 | 2.90457 | 3.04499 |
| D2-16M-CAPABILITY | 3.21435 | 3.13364 | 2.84405 | 3.05431 |
| D3-16M-CURRICULUM | 3.23283 | 3.14296 | 2.83484 | 3.07387 |
| D4-16M-ANNEAL | 3.20723 | 3.16106 | 2.88111 | 3.00296 |

D5 was gated out: D3/D4 evidence or failure guard did not clear. The exact D3/D4 quality, capability, source-regression, and protected-time inputs are frozen in `d5_combination_decision.json`.
The capability selector scored 1,620,201 indexed documents. Within-source score quantiles, selected counts, pinned training revisions, inherited DATA-D contamination registry, and randomly sampled text snippets for inspection are in `research_d/capability_selector_report.json`. Semantic overlap beyond the inherited exact/near-duplicate checks remains a limitation.

## Observed curriculum difficulty

The following values are cumulative accepted difficulty means from the registered Research-D sampler. The complete step series is preserved in the metrics file.

| D3 logical tokens | Accepted difficulty mean |
|---:|---:|
| 10,002,432 | 0.5897 |
| 37,502,976 | 0.6144 |
| 75,005,952 | 0.6267 |
| 150,000,000 | 0.6324 |

The difficulty score is a within-source mean of document length and first-256-token lexical-diversity percentiles, not an external measure of reasoning difficulty. D4's high-quality FineWeb-Edu threshold was registered before evaluation; source distribution and token exposure remain visible in the run registry and sampler metrics.
