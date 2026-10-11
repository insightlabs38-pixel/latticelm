# Full Co4 source audit

Post-submission research. Source decisions are separate from outcome claims.

## Primary sources and immutable pins

- Zain, Naseem, Adeel, [Single layer tiny Co4 outpaces GPT-2 and GPT-BERT](https://aclanthology.org/2025.babylm-main.24/), BabyLM 2025; PDF SHA256 `fb736beb8c45af196e6384d727a8c07c2fdc7068735da15517579c8a897030b2`.
- Adeel and Bilal, [Scalable Machines with Intrinsic Higher Mental-State Dynamics](https://arxiv.org/abs/2603.13453); PDF SHA256 `e0931fe41c1a6ba7c7f0a50da733a450f0a86f34bd0180dcedd51366b4c6b2f3`.
- Author [IHMS repository](https://github.com/ARIA-Funded-TREND/IHMS), commit `e25e9119173d59fdb9e4548ef640734dbd19f067`.
- Author [HMS repository](https://github.com/ARIA-Funded-TREND/HMS), commit `72aceea25b0bac49699d05deaa51860e6dfa5724`.
- [Google Cloud Spot documentation](https://docs.cloud.google.com/compute/docs/instances/create-use-spot).

Downloaded papers and pinned repositories are local in `sources/`; their pins are recorded in `PRIMARY_SOURCE_MANIFEST.json`. Author repositories carry CC-BY-NC-4.0. Adapted source must retain attribution.

## FCo4-A: BabyLM reproduction is unavailable

The BabyLM paper describes learned input embeddings and initial Q/K/V agent representations, two heads, a single decoder stage, a causal triangular mask, autoregressive cross entropy, and no conventional Transformer FFN. It does not define enough of the triadic update, credit/readout, agent initialization, normalization, residual path, or mask placement to recover an exact implementation. Text mentions 24 latent agents while the figure caption describes 12. Appendix variants differ in LR, batch size, and schedule. These cannot be collapsed into one invented faithful recipe.

No public LM implementation was located in the paper or the pinned author repositories. The user independently searched and confirmed this limitation on October 11, 2026, and instructed that missing details must not be invented or labeled faithful FCo4-A.

**Decision:** faithful FCo4-A remains unavailable. A separately registered paper reconstruction is allowed by the user, but would require a complete table of inferred choices and could not support a faithful reproduction claim. It is lower priority than the source-grounded IHMS branch.

## FCo4-B: released 2026 implementation is authoritative

Relevant files are `Vision/object_recognition/attention.py`, `transformer.py`, and `vit.py` in the pinned IHMS repository. The released vision operator and paper differ materially:

| Component | Released code | Paper |
|---|---|---|
| Q modulation | `Q_X + Q_cls * K_X` | Equations 7/10 use latent/context updates |
| K modulation | `K_X + K_cls * Q_X` | Equations 8/11 use latent/context updates |
| V modulation | `V_X**2 + 2*V_X + (Q_cls*K_cls)*(1+abs(V_cls))` | Eq. 9 uses the updated Q_m*K_m and optional mu |
| Selection | Independent top-k across token dimension for each feature, separately for Q/K/V | General relevance/salience description |
| Output | CLS plus modulated V plus scattered selected-value attention updates | Several readout alternatives described |
| FFN/residual | LayerNorm, attention residual, LayerNorm, GELU MLP residual | Claims also discuss no conventional FFN |
| Precision/LM schedule | No released LM experiment | Not enough information for an LM recipe |

The released code defaults to one learned CLS latent and one head in modulated blocks, selects 12 features per head/channel, and has a hardcoded 13-item reshape. Modulated attention has no final output projection. The released pure configuration has six modulated blocks; a hybrid configuration inserts ordinary attention at blocks 3 and 6. The released block includes an MLP with expansion 4 and GELU. Mu updates are explicitly optional and absent from the released main-result code. These details must be preserved or individually disclosed as adaptations; replacing V's latent product by updated Q*K would be a distinct equation variant.

Vision code is noncausal. A final triangular attention mask cannot repair noncausal top-k selection. The LM adaptation must restrict every selector to the available prefix, separately for Q, K, and V; maintain the source's feature-wise selection; define token-level readout explicitly; and pass prefix, future-perturbation, selection-index, and gradient-leak checks. The source's scattered readout requires an explicit prefix-local adaptation so later queries cannot update earlier positions.

**Decision:** implement B as an explicitly labeled causal LM adaptation of this pinned released operator. Do not claim that it reproduces the unavailable BabyLM LM. Keep a literal prefix-by-prefix reference and a separate production implementation. A tiny compatibility fixture must check the raw modulation against the released class. Selection is source-required for this branch, so the conditional top-k restriction is satisfied.

## FCo4-C and controls

A hybrid is a separate family and must keep its ordinary attention/FFN costs visible. Prepare source-inspired modulated/ordinary block order, matched shallow dense and MOD controls, and a deeper dense control. The initial implementation must expose depth, latents, heads, top-k, initialization scale, and readout choices in frozen records. Compute accounting includes actual wall time and a disclosed proxy; top-k prefix reference is quadratic and cannot claim the source's near-linear scaling.

The implemented hybrid's ordinary MHSA processes causal token states and carries the prefix-local latent unchanged. The released vision MHSA jointly processes a global CLS token. This is an additional disclosed causal LM adaptation; raw modulated-operator equivalence does not establish whole-hybrid equivalence. Standard control blocks use four attention heads; modulated blocks retain one head/latent as required by the released operator.

## VM interruption assumptions

Metadata confirms this VM is preemptible, automatic restart is disabled, and maintenance action is TERMINATE. No dedicated 120-second notice is assumed. The controller polls `instance/preempted`, requests a flush if available, and relies on periodic atomic checkpoints and persistent disk. A user systemd service can recover the controller when the VM boots; it cannot start a stopped VM.

## Current validity status

The matched dense/control work is independent of the unresolved BabyLM reproduction. No full-Co4 result is valid or claimed before operator equivalence, causal tests, parameter accounting, smoke, exact resume, and evaluator compatibility pass.
