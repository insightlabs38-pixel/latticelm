# Follow-up final synthesis

Post-submission research. Updated 2026-10-12T20:29:24.834792+00:00.
Status: LIVE — incomplete endpoints are explicitly unresolved.

## 1. At 32M/300M, does existing MOD-Co4 differ meaningfully from dense?

Candidate `F32-DENSE-S1337-FRESH300M` against `CONTROL-R32-CO4-CONTROL`: loss delta -0.00977837, 95% paired interval [-0.01673524618148804, -0.002895634844899178]; BPB delta -0.00438447, interval [-0.005904554904227413, -0.002856395280048923]; clear task regression False; wall ratio 0.9512. Gate closed/not promoted. See `decisions/dense-300m-closure.json` for source-wise evidence and limits. This is a separately registered fresh dense300M comparison. Historical dense100M had ended its WSD; it was not silently continued.

## 2. Does D4 annealing reproduce?

Candidate `F16-D4-S2026-FRESH150M` against `CONTROL-R16-S2026-CONTROL`: loss delta 0.00509162, 95% paired interval [-0.00298002067953348, 0.01304946634918451]; BPB delta -0.00521105, interval [-0.00651527535967071, -0.003748956482277909]; clear task regression False; wall ratio 0.9892. Gate closed/not promoted. See `decisions/d4-replication.json` for source-wise evidence and limits. The sampler cycle, source draw counts and transition receipt preserve the exact late mixture. BPB-only movement is distinguished from coherent recipe promotion.

## 3. Does Differential-Co4 reproduce?

Candidate `F8-DIFF-S7331-FRESH100M` against `F8-MOD-S7331-FRESH100M`: loss delta 0.0270503, 95% paired interval [0.020634501054883004, 0.03363640710711478]; BPB delta 0.0291915, interval [0.02754903048557248, 0.03088266160215893]; clear task regression False; wall ratio 1.2. Gate closed/not promoted. See `decisions/differential-replication.json` for source-wise evidence and limits. One fresh matched seed7331 pair supplements the historical pilot; no automatic four-seed tournament.

## 4. Which exact full-Co4 architectures were implemented?

IHMS author implementation pinned to e25e9119173d59fdb9e4548ef640734dbd19f067: feature-wise independent Q/K/V top-k, latent-modulated triadic Q/K/V and scatter readout. B is pure modulated; C is modulated+MHSA. Faithful BabyLM A remains unavailable under the user-confirmed source limitation.

## 5. Faithful, adapted or hybridized?

The released IHMS operator is independently fixture-tested. Its autoregressive LM adaptation is explicitly adapted: causal prefix-local selection/readout, learned token/position embeddings, untied LM head, widths/depths, initialization and inherited optimizer. C additionally hybridizes with causal MHSA. Neither is claimed to faithfully reproduce the unavailable BabyLM code.

## 6. Did causal leak tests pass?

Passed all 11 tested production configurations. Tests cover multiple seeds/prefix lengths, changed future tokens, selection indices and future embedding gradients; forward, gradients and one optimizer step match an independent literal oracle.

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

Reporting source revision `578254776c269e9fd012ad3dd71e085095059714`; branch `posthack-followup-2026-10`, frozen main/submission `43832b4c7bfe8be007ca6b9d1042aa2041c7f280`. HF `insightlabs38-pixel/LatticeLM-research`, prefix `post-hackathon-followup/2026-10-11`, last verified receipt f8aa76b2af7c2eca2b6369b6a060250c9fd23f56. Final closeout commit/revision are recorded in FOLLOWUP_CLOSEOUT_RECEIPT.json and the archive manifest.
