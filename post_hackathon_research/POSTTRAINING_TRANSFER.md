# Retention-aware post-training transfer

The parent is D4-16M-ANNEAL (SHA-256 `8116e957ec7875c579f4458e9fca21c10c87d7976ec775768b0c3a203f23e583`). P0 trains on disjoint generic SciQ/QASC reasoning QA with 50% natural DATA-D replay. P1 adds CommonsenseQA training examples for broad capability alignment and a frozen-parent reference KL on the same replay. P2 uses official training splits of the four competition tasks, with exact held-out-prompt hash exclusion; it is a **task-tuned diagnostic upper bound** and is not comparable to pretraining-only systems. All screens use a fresh AdamW optimizer, 3e-5 LR, one million logical target tokens, and the 4K tokenizer.

The implemented objective is `0.5 CE_completion + 0.5 CE_natural + 0.5 λ KL(p_ref || p_θ)`; λ is 0 for P0/P2 and 0.10 for P1/P3/P4. KL is evaluated only on natural replay tokens. Completion loss masks prompt and padding tokens. Registered SFT data order shuffles each pinned training source with a fixed seed, then interleaves sources round-robin so a one-million-token screen sees every included skill rather than only the first dataset in the file.

| Arm | Held-out loss | WikiText BPB | Four-task raw accuracy | Held-out skill NLL/token | BPB change vs parent |
|---|---:|---:|---:|---:|---:|
| D4-16M-ANNEAL | 3.09604 | 1.62169 | 0.4118 | 7.85239 | +0.00% |
| P0-16M-GENERIC-REPLAY | 3.11952 | 1.65568 | 0.4160 | 2.52157 | +2.10% |
| P1-16M-ALIGNED-REPLAY-KL | 3.12379 | 1.64332 | 0.4165 | 2.24457 | +1.33% |
| P2-16M-TASK-TUNED-DIAGNOSTIC | 3.11333 | 1.61535 | 0.4243 | 3.05756 | -0.39% |
| P3-16M-ONPOLICY-REPLAY | 3.12377 | 1.64503 | 0.4173 | 2.24848 | +1.44% |

P2 is labeled task-tuned in the registry, data manifest, checkpoint lineage, and this report. A small raw-accuracy change is not a supported broad gain without paired uncertainty analysis. Weight interpolation is evaluated before any recovery training.

The approximately on-policy replay arm was run: time gate passed. When run, four of sixteen natural replay examples per step have eight continuation tokens sampled autoregressively from the current model; generation input tokens are counted separately in `processed_tokens`.
Interpolation retained P1 fraction `1.00` under the pre-registered <=2% WikiText BPB and <=5% source-loss retention gates. The complete interpolation frontier and selected capability screen are in `posttraining/interpolation/report.json`.


## Frozen BASE confirmation gate

The bounded 49M child was gated out: 16M retention/transfer evidence insufficient. The frozen BASE checkpoint itself was not modified.