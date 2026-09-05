# Phase 7E closeout report

## Execution audit

Phase 7E is **PARTIAL** relative to the full optional plan and complete for DATA-C scaling. The master started at `2026-09-05T05:15:58.126956Z`, exited at `2026-09-05T09:59:23.549737Z`, and elapsed 17,005.423 seconds (4:43:25.423). Measured cumulative model-training time was 20,268.695 seconds from initialization through 100M; Phase 7E added 15,233.420 seconds (4:13:53.420) after the existing 25M state. There were zero Spot interruptions, subprocess failures, retries, upload retries, or infrastructure downtime. The soft and hard deadlines were not reached.

Preflight, DATA-C 25→50M, 50M evaluation, FineWeb-Edu expansion, DATA-C 50→100M, 100M evaluation, and final-report generation completed. DATA-D preparation was skipped immediately because the controller explicitly had no implementation (`RuntimeError: prerequisite stage unavailable: DATA_D_PREPARATION`); DATA-D training/evaluation were therefore **NOT REACHED**. No ~24M experiment ran. The process table, user/system services, stale lock PID, and GPU check confirmed the master and trainer had exited and no checkpoint/upload was active.

## Reached DATA-C milestones

| Metric | 25M | 50M | 100M |
|---|---:|---:|---:|
| Parameters | 15,949,760 | 15,949,760 | 15,949,760 |
| Exact training tokens | 25,000,000 | 50,000,000 | 100,000,000 |
| Tokens/parameter | 1.5674 | 3.1348 | 6.2697 |
| Common validation loss | 3.328604 | 3.208180 | 3.134722 |
| Common validation PPL | 27.8994 | 24.7340 | 22.9823 |
| BabyLM validation loss | 3.145104 | 3.058672 | 2.978331 |
| FineWeb-Edu validation loss | 3.854534 | 3.694580 | 3.588872 |
| WikiText-103 PPL | 102.5510 | 79.6793 | 72.5823 |
| WikiText BPB | 2.134354 | 2.018033 | 1.975032 |
| HellaSwag | 0.260108 | 0.262995 | 0.264190 |
| ARC-Easy | 0.284512 | 0.296296 | 0.297559 |
| PIQA | 0.520131 | 0.541893 | 0.533732 |
| WinoGrande | 0.481452 | 0.505919 | 0.496448 |
| Cumulative train time | 5,035.276s | 10,053.671s | 20,268.695s |
| Interval train time | 1,012.581s | 2,026.035s | 2,044.569s (90→100M) |
| Average train tok/s | 4,964.97 | 4,981.67 | 4,894.75 |
| Peak RSS | 3,132,907,520 | 3,039,940,608 | 3,278,340,096 |
| Training checkpoint SHA-256 | `35bbbe4d…88de` | `444b41c1…13fc` | `3bf27b60…cb4` |

The 25M immutable HF bundle is revision `4bd98a6f87f5d4594cd7fe0ba3aa28c659e5ae32`. The 50M metrics and checkpoint hash are preserved, but its rolling weights were replaced before closeout and no 50M HF bundle exists: checkpoint verification is **NOT POSSIBLE**, not inferred. The 100M checkpoint was strict-loaded, exported to Safetensors, round-trip logits matched bit-for-bit, and uploaded to `experiments/co4-l-data-rich-100m/` at revision `95d27f1c012893d12e34005b067a58a666fd472b`; its Safetensors SHA-256 is `b7ee53034c53a3a14c8c3152bb953f00056327dbc87f5455a4bdbd5bad78ba6c`.

## Checkpoint and resume verification

The local 100M rolling `latest.pt` and `fallback.pt` hashes both equal the recorded `3bf27b6097771e37b3c58074ae7514193d79c098e85007949b56cef4b7603cb4`; `previous.pt` independently matches its sidecar. The 100M payload contains model, AdamW optimizer, LambdaLR scheduler, Python RNG, PyTorch RNG, per-source draw counters (`baby=585952`, `web=195318`), exact token/step counters, cumulative time, lineage/source identity, and the original Phase 7C manifest hash. The expanded corpus and tokenizer hashes are externally bound by `phase7e_data_c_manifest.json`. Two independent restores produced the same next-batch hash. Thus 100M is **INFERENCE-VERIFIED** and **EXACT-RESUME-VERIFIED** as the joint checkpoint + frozen manifest/data set. The 25M bundle is inference-verified and its original recovery was previously exact-resume-verified; 50M is metrics-only and **NOT EXACT-RESUME-VERIFIED**.

Remote `best/` and `latest/` previously pointed to the Phase 7B Co4-S 10M bundle (loss 3.625211 under the same common validation). They were scientifically stale. After verification they were safely updated to the 100M Co4-L bundle at revisions `071c3b9080a5356bb92e5bc456515b27821d2f65` and `050cc5bb916189fa8226f45a019a7414911351d8`, while immutable named paths remain unchanged.

## DATA-C expansion audit

The FineWeb-Edu pool expanded atomically from 20,000,582 to 50,000,020 tokens and from 13,875 to 33,509 retained training documents (19,634 added); 40,314 documents were scanned. It used pinned revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, streaming shuffle seed 1337/buffer 1000, minimum 50 normalized words, language score ≥0.9, benchmark fingerprint screening, and deterministic 5% validation exclusion. The original 80,002,328 bytes were verified as an exact prefix before replacement. The one-shard expanded mmap is 200,000,080 bytes, readable, and hashes to `ef4e00fe85cd7136cd1c173c89ee9cc9ec38e19016fedb514e9e19b2f029482a`; document IDs hash to `1626b873…2dc7` and tokenizer identity is `4f313ebc…637e5`.

The script reports aggregate filtering only; separate exact-dedup, near-duplicate, and per-cause decontamination removal counts were not emitted. It did not perform a distinct dedup pass beyond deterministic source IDs/sampling and benchmark screen. At 100M only 25M FineWeb tokens had been drawn from the expanded 50M pool, so **no FineWeb repetition occurred**. BabyLM exposure (75M) also remained below its 187.8M-token pool.

## Scaling interpretation

From 25→50M common loss improved 0.120425 (0.004817 per added million tokens), PPL fell 3.16535, WikiText PPL fell 22.87176, and BPB fell 0.116321. All four reasoning tasks improved: HellaSwag +0.002888, ARC-Easy +0.011785, PIQA +0.021763, WinoGrande +0.024467.

From 50→100M common loss improved 0.073458 (0.001469 per million), PPL fell 1.75177, WikiText PPL fell 7.09699, and BPB fell 0.043001. HellaSwag (+0.001195) and ARC-Easy (+0.001263) improved, while PIQA (-0.008161) and WinoGrande (-0.009471) regressed. The 90→100M common-loss improvement was only 0.000739, so the most recent local slope is near-flat even though the broader 50→100M language-model trajectory remains positive.

**DATA-C LANGUAGE-MODELING TRAJECTORY: MODERATE / approaching plateau.**

**DATA-C REASONING TRAJECTORY: MIXED.** Individual benchmark uncertainty is large and no aggregate score is used to hide regressions.

## DATA-D audit

DATA-D-BROAD-v1 was **NOT PREPARED** and its 25M pilot was **NOT RUN**. The manifest and three reports are explicit placeholders, not claims of realized data. Consequently source proportions, corpus hashes, mmap/reproducibility checks, exact dedup, paragraph/near dedup, and controlled DATA-D-vs-C deltas remain unfinished. **BROADER-DATA RESULT: INCONCLUSIVE.**

## Competition status

The current verified winner is Co4-L DATA-C 100M: 15,949,760 parameters, WikiText PPL 72.5823/BPB 1.97503, HellaSwag 0.26419, ARC-Easy 0.29756, PIQA 0.53373, and WinoGrande 0.49645. It trained on a GCP `c4a-standard-16` Spot AArch64 VM at about 4,895 tok/s; cumulative model time was 5.63 hours. Relative to previously researched sub-50M references, no global-record claim is warranted. Language modeling is the strongest relative area; WinoGrande is weakest. More unique base data is most likely to improve PPL/BPB, while reasoning—especially PIQA/WinoGrande—likely requires broader or targeted procedural/contrastive data.

## Preservation limitations

The autonomous controller never uploaded post-25M recovery states because no HF token was present in its environment. Closeout preserved the surviving 100M exact-resume state and logs on HF. The 50M weight checkpoint cannot be recovered from filenames, metrics, or hashes and is explicitly not claimed as verified.
