# Curated experimental artifacts

This directory contains years of iterative research evidence. Historical files already tracked at its root and in the experiment folders remain unchanged. New final evidence is curated into the following files:

- [`reports/final_production_report.md`](reports/final_production_report.md): final BASE metrics, recipe, and immutable model/data identities.
- [`reports/final_checkpoint_comparison.md`](reports/final_checkpoint_comparison.md): comparison of production checkpoints at the selected milestones.
- [`reports/final_recipe_summary.md`](reports/final_recipe_summary.md): architecture, DATA-D-v4, optimizer, schedule, and run summary.
- [`reports/posttraining_final_report.md`](reports/posttraining_final_report.md): finalists, broad evaluation, and the BASE selection.
- [`reports/recovery_experiment_report.md`](reports/recovery_experiment_report.md): recovery and mixed-objective follow-up.
- [`metrics/`](metrics/): compact final metrics, selection comparison, and validation curve.
- [`manifests/final_production_provenance.json`](manifests/final_production_provenance.json): architecture, run budget, data/tokenizer/checkpoint hashes, and optimizer partition.

Raw checkpoints and optimizer state are omitted because they are hundreds of megabytes each and the local checkpoint tree totals roughly 129 GB. Run-state files, locks, PIDs, temporary files, raw event/log streams, token shards, and intermediate candidate directories are execution state or reproducible bulk output. Generated plots and QA captures are not primary evidence. Their presence on a researcher’s machine does not make them part of the submission.

The raw material remains on disk in this checkout. Ignore rules keep it out of normal Git status while allowing selected reports, metrics, and manifests to be versioned. Previously tracked historical research artifacts remain tracked; this curation does not erase or relocate them. The final checkpoint can be identified by its SHA-256 and retrieved from the model storage referenced in the report.
