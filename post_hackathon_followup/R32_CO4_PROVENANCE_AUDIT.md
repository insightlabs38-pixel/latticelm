# Historical 300M Co4 provenance

PASS: one fresh initialization and one 300M WSD trajectory.

Verified 36,622 contiguous steps, every recorded LR (max error 0.0), one TRAIN_START / TRAIN_END, no RESUMED event, source/config/checkpoint hashes, all data-shard hashes, exact final sampler cursor and next-batch replay. A fresh model plus empty Muon/AdamW optimizers reproduces the first three recorded losses within 3e-5.

No initial checkpoint was archived. This conclusion rests on the converging evidence above, rather than an unrecorded assertion about initialization. See the companion JSON for values and hashes.
