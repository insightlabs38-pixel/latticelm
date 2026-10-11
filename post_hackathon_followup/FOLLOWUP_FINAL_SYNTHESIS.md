# Follow-up final synthesis

Post-submission research. Updated 2026-10-11T02:30:32.812399+00:00.
Status: LIVE — incomplete endpoints are explicitly unresolved.

## 1. At 32M/300M, does existing MOD-Co4 differ meaningfully from dense?

Unresolved: paired endpoint evaluation is not yet complete. This is a separately registered fresh dense300M comparison. Historical dense100M had ended its WSD; it was not silently continued.

## 2. Does D4 annealing reproduce?

Unresolved: paired endpoint evaluation is not yet complete. The sampler cycle, source draw counts and transition receipt preserve the exact late mixture. BPB-only movement is distinguished from coherent recipe promotion.

## 3. Does Differential-Co4 reproduce?

Unresolved: paired endpoint evaluation is not yet complete. One fresh matched seed7331 pair supplements the historical pilot; no automatic four-seed tournament.

## 4. Which exact full-Co4 architectures were implemented?

IHMS author implementation pinned to e25e9119173d59fdb9e4548ef640734dbd19f067: feature-wise independent Q/K/V top-k, latent-modulated triadic Q/K/V and scatter readout. B is pure modulated; C is modulated+MHSA. Faithful BabyLM A remains unavailable under the user-confirmed source limitation.

## 5. Faithful, adapted or hybridized?

The released IHMS operator is independently fixture-tested. Its autoregressive LM adaptation is explicitly adapted: causal prefix-local selection/readout, learned token/position embeddings, untied LM head, widths/depths, initialization and inherited optimizer. C additionally hybridizes with causal MHSA. Neither is claimed to faithfully reproduce the unavailable BabyLM code.

## 6. Did causal leak tests pass?

Tiny oracle/author/causality tests implemented; configuration-specific C0 certificates are pending or failed. Consult certificates and C0 receipts. Tests cover multiple seeds/prefix lengths, changed future tokens, selection indices and future embedding gradients; forward, gradients and one optimizer step match an independent literal oracle.

## 7. Did full Co4 beat shallow dense, current MOD or deeper dense?

Comparator-specific paired loss/BPB/task evidence appears in C2 promotion and winner decisions. No evaluated winner selected yet. A depth tie against deep dense does not establish an advantage over parameter-matched shallow dense; attribute those axes separately.

## 8. Per token, parameter, FLOP proxy, wall time or depth?

FOLLOWUP_RUN_SUMMARY.json reports each axis independently. Learning curves record quality vs tokens and wall time; total parameters, stages/blocks, peak RSS, 6NT and extra triadic work are disclosed. A model has no efficiency claim from parameter count alone.

## 9. Did the winning formation reproduce across two seeds?

Unresolved: paired endpoint evaluation is not yet complete. Two primary seeds maximum:1337 and2026. Confirmation requires consistent direction on the same comparator/axis and rejects clear task regressions; item confidence intervals do not measure training-seed uncertainty.

## 10. What hyperparameters were architecture-specific and why?

At most four C1 settings per family: inherited LR/init, LR×0.5, LR×2, or initσ0.01. Extra arms open only for unstable/non-learning or clearly worse validation. Selection uses held-out loss, never downstream scores. Each selected setting is frozen before fresh C2/C3 runs; see C1 selection decisions.

## 11. Does the published large depth/sample claim hold in causal LM?

Supported only for the matched comparisons and replicated axes documented above; otherwise unresolved or not reproduced here. Quadratic causal prefix processing does not verify published near-linear scaling. A negative result does not disprove the source vision/RL architecture.

## 12. Which architecture deserves GPU-scale work?

No full-Co4 architecture qualifies without coherent two-seed evidence. Differential/D4 are candidates only if their own replication gates pass.

## 13. Which lines should stop?

Failed or non-promoted compatibility/screen/replication branches stop at their frozen gates. Previously deferred PHM, Clifford, MoE, CGLS, value residual, Engram, post-training, DeltaNet, BLT, Titans and multi-token work remain deferred. No claim that an unrun conditional experiment failed.

## 14. Are all irreplaceable artifacts safely remote?

Verified coverage is determined by FOLLOWUP_ARCHIVE_MANIFEST.json and HF_REPRESENTATIVE_LOAD_RECEIPT.json; upload exit alone is insufficient. Historical remote duplicates are reused, canonical shards/manifests are mirrored as missing delta, and raw document indexes can be reconstructed from verified gzip plus original source SHA. Migration/closeout is ongoing; complete coverage is not yet claimed.

## 15. What exact Git/HF state defines the research?

Reporting source revision `c9ed214a5131c2e891f9e77b5ecee74243be4f72`; branch `posthack-followup-2026-10`, frozen main/submission `43832b4c7bfe8be007ca6b9d1042aa2041c7f280`. HF `insightlabs38-pixel/LatticeLM-research`, prefix `post-hackathon-followup/2026-10-11`, last verified receipt d1e2f1f5436dbb2e30bf5ba6db761c6f8534f84e. Final closeout commit/revision are recorded in FOLLOWUP_CLOSEOUT_RECEIPT.json and the archive manifest.
