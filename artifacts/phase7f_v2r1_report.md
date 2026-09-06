# Phase 7F-v2r1 acceptance report

DATA-D-v2r1 PREPARED: YES

TOTAL CLEAN UNIQUE TRAIN TOKENS: 100,011,495

TOTAL RETAINED DOCUMENTS: 56,135 training; 1,169 validation

REALIZED FINEWEB-EDU: 49.994801%

REALIZED WIKIPEDIA: 25.006893%

REALIZED FINEWEB: 24.998305%

COMMON-VAL REGISTRY SIZE: 2,605 windows / 500,000 tokens

COMMON-VAL CONTAMINATION REMOVALS: 95

BENCHMARK CONTAMINATION REMOVALS: 69

TRAIN/VAL DISJOINTNESS: PASS

FULL EXACT-RESUME TEST: PASS (bit-exact CPU eager)

INDEPENDENT DEDUP VERIFICATION: PASS

FRESH-PROCESS NEXT-BATCH REPRODUCTION: PASS

TOP-LEVEL MANIFEST SHA256: `f42df21984b27e77b6901a6335f751cee911efd1ff4579aba038a577255d601a`

BUILDER GIT COMMIT: `77cb221b702f2250345b58fe52a79cafb9e3f133`

ACCEPTANCE GATES: PASS

## Controlled Co4-L result at exactly 25M

The model was initialized randomly with seed 314159 and trained on the GCP C4A/aarch64 eager-PyTorch environment. It has 15,949,760 parameters and consumed exactly 25,000,000 loss-bearing tokens: 12,500,000 FineWeb-Edu, 6,250,000 Wikipedia, and 6,250,000 FineWeb.

- Common validation loss: 4.464799 (DATA-C delta +1.136195; worse)
- DATA-D balanced validation loss: 3.695686
- FineWeb-Edu validation loss: 3.694157
- Wikipedia validation loss: 3.579136
- FineWeb validation loss: 3.936519
- WikiText-103 PPL: 83.40936 (delta -19.14165; better)
- WikiText BPB: 2.03912 (delta -0.09523; better)
- HellaSwag: 0.262896 (delta +0.002788)
- ARC-Easy: 0.283249 (delta -0.001263)
- PIQA: 0.534276 (delta +0.014145)
- WinoGrande: 0.498027 (delta +0.016575)

The native checkpoint SHA-256 is `6760880ba5196ae9af8688eb126a65db1ef7f0fd80613549c01692ba0676a894`. The verified immutable Hugging Face revision is `0af296eaedabc77622edb2c04d5f3a01001a42a0`; safetensors SHA-256 is `12906299109758ff67e407de0cc73a7f0a8a52a8ec2b50e40531302eabe25660`.

Final classification: **E — mixed/inconclusive**. No follow-up experiment was launched.
