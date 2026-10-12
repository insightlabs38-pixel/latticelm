# Hyperspherical nCo4 pilot

The 8M pilot implements unit-direction embedding/hidden/head and projection weights, post-MOD unit Q/K directions with learned logit scale, learned attention/FFN spherical interpolation steps, and learned FFN/logit scaling. Normalization is functional in the forward pass, so no in-place optimizer projection or optimizer-state replacement occurs. Both arms use the same AdamW 3e-4 recipe, 4K tokenizer, data order, seed, batch and 50M tokens. This is a bounded nGPT-style mechanism screen, not a claim to reproduce every reference implementation detail.

| Arm | Held-out loss | WikiText BPB | Four-task raw accuracy | 12/16-thread calibrated TPS |
|---|---:|---:|---:|---:|
| AdamW Co4 | 3.56980 | 1.96984 | 0.3985 | 13699 |
| nCo4 | 3.56827 | 1.95236 | 0.4017 | 12206 |

A positive signal requires quality at matched wall/FLOP, not simply a lower loss under a different optimizer. Tests cover unit hidden norms, finite gradients, checkpoint loading, and the common evaluator.

Primary reference: [nGPT](https://arxiv.org/abs/2410.01131).
