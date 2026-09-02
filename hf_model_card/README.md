---
library_name: pytorch
pipeline_tag: text-generation
license: other
---

# LatticeLM — private research checkpoint storage

**RESEARCH CHECKPOINTS — NOT FINAL GIBC RELEASE**

LatticeLM is a CPU-native small-language-model research project for Global
Innovation Build Challenge V2, Track 01 (Foundational LLM). Models are trained
from random initialization without pretrained language-model weights,
distillation, or an external language-model teacher.

The research tournament compares a dense causal Transformer, a tiny
reference-style Engram conditional-memory adaptation, and a causal adaptation
of the Co4/IHMS MOD mechanism. All candidate models must remain below 50 million
trainable parameters.

Training data is the public BabyLM 2026 Strict-Small corpus, pinned by revision
and source hashes in each checkpoint's `training_manifest.json`. Official GIBC
evaluation sets are not training data and are not stored here.

This private model repository stores selected safetensors inference weights,
tokenizers, manifests, and strategically useful private resume state. It is not
the source-code repository. Intermediate checkpoints may be incomplete and may
not represent the architecture ultimately submitted to GIBC.

## Layout

- `best/`: best qualifying held-out validation checkpoint.
- `latest/`: replaceable cloud-failure recovery checkpoint and resume state.
- `experiments/`: immutable scientifically important checkpoints.
- `tests/`: tiny end-to-end persistence smoke tests.

Every checkpoint directory includes `SHA256SUMS`; public/inference weights are
always `model.safetensors`. Pickle-based files, where present, are private
optimizer/RNG resume state and are never presented as model weights.
