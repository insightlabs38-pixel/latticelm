# Differential-Co4 pilot

The same 8M Co4 backbone projects Q/K/V once, applies MOD once, QK-Norm and RoPE once, then splits each head into two Q/K subspaces. Two causal softmax maps read the same value stream. Their difference uses the published exponential lambda reparameterization, with lambda initialized at 0.8 and fixed scale correction. Same total Q/K/V widths and the modern Muon/AdamW optimizer are used. The matched Co4 50M milestone has the same seed, data order and token budget.

| Arm at about 50M tokens | Held-out loss | WikiText BPB | Calibration TPS |
|---|---:|---:|---:|
| Co4 baseline milestone | 3.53939 | 1.89704 | see baseline calibration |
| Differential-Co4 | 3.46590 | 1.87628 | 10163 |

In a full 32×256 forward/backward CPU profile, SDPA operators accounted for 13.0% of operator self time in Co4 and 0.1% in Differential-Co4. The profile excludes sampler, Python and optimizer time; measured training TPS remains the wall-clock comparison.
The two attention-map calls increase attention work; a lower per-token loss would not by itself establish FLOP or wall efficiency. Tests cover the disabled exact-control path, causal future-token isolation, finite gradients and checkpoint loading.

Primary reference: [Differential Transformer](https://arxiv.org/abs/2410.05258).
