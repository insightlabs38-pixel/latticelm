# Curriculum-guided Co4 depth growth

The 2×2 design compares full/static D0, full/curriculum D3, progressive/static E1, and progressive/curriculum E2. Progressive models begin at five of ten blocks, grow to seven at 18.75M tokens, and to ten at 37.5M. Newly active blocks clone the preceding block and begin with a zero output gate; the trainer verifies immediate logit parity at each event. Existing optimizer state is retained, and dormant new parameters acquire state on their first gradient.

| Arm | Weighted held-out loss | Four-task raw accuracy screen | 6NT FLOP proxy (10¹⁵) | Training hours |
|---|---:|---:|---:|---:|
| A16-CO4-S1337 | 3.09548 | 0.4105 | 14.404 | 6.01 |
| D3-16M-CURRICULUM | 3.09946 | 0.4125 | 14.404 | 6.01 |
| E1-16M-GROWTH-STATIC | 3.12533 | 0.4180 | 13.205 | 5.55 |
| E2-16M-GROWTH-CURRICULUM | 3.12770 | 0.4170 | 13.264 | 5.57 |

Loss interaction `E2 − E1 − D3 + D0` = **-0.00161**; a negative value favors a synergistic curriculum/growth interaction.
Raw capability interaction on the same four tasks = **-0.0030**; a positive value favors synergy.

The FLOP column uses a 6NT proxy with the measured active layer count at every step. It includes embedding/head parameters and excludes sampler, optimizer, and Co4 elementwise overhead. Token-matched endpoints have different active-layer FLOPs; capability-screen uncertainty limits claims from small percentage-point changes.
