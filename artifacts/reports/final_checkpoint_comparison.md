# Final checkpoint comparison

The final production evaluation compared three semantically selected checkpoints from the same 1.9B-token run. `end-stable` marks the end of the stable learning-rate segment, `mid-decay` is within cosine decay, and `final` is the completed run.

| Checkpoint | DATA-D validation loss | WikiText BPB | HellaSwag | ARC-Easy | PIQA | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|
| end-stable | 2.917166 | 1.550678 | 0.26827 | 0.35101 | 0.54679 | 0.50592 |
| mid-decay | 2.794589 | 1.469379 | 0.26698 | 0.36322 | 0.55332 | 0.48777 |
| final | **2.630494** | **1.413462** | **0.27096** | **0.36785** | **0.56692** | **0.50987** |

The final checkpoint was selected by the frozen recipe. No measured checkpoint challenged it on the recorded selection rule. These values are from the production checkpoint evaluation, not the separate 256-example post-training proxy.
