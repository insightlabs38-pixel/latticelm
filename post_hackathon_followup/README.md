# Post-submission follow-up research

This work is isolated on `posthack-followup-2026-10`. Historical main, the frozen hackathon model, and historical metrics remain preserved.

The user authorized a fresh matched dense 300M run after the old 100M dense schedule was shown unsuitable for continuation. R32 provenance passed; the scientific-batch smoke, forced trainer/master interruption, exact recovery comparison, and evaluation loader gates passed. `F32-DENSE-S1337-FRESH300M` launched under the independent boot-enabled controller on October 11, 2026. See the immutable input audit for the original concerns and the later provenance and numerical diagnosis reports for their resolutions.

Hard VM expiry: October 17, 2026 at 16:10 EDT (20:10 UTC). Scientific training stops four hours earlier to reserve evaluation and verified delta migration time. Actual progress is in `FOLLOWUP_LIVE_SCHEDULE.md`; no unfinished experiment is a result.

Faithful BabyLM FCo4-A cannot be reconstructed unambiguously from public artifacts. The user confirmed this limitation; source-grounded IHMS causal adaptation is the active full-Co4 branch. See `FULL_CO4_SOURCE_AUDIT.md`.

Historical 8M controls have incomplete trainer-source provenance: three final checkpoints identify an older trainer whose exact source snapshot is missing, and the earliest control records no trainer hash. They are not silently imported into the new primary comparisons. A user choice is pending between four fresh matched 100M controls and recovery of the missing historical source snapshots. This holds only the affected full-Co4 primary comparison/promotion gates; dense300M, D4, Differential and fresh compatibility/screens proceed independently. R32 and R16 controls have passed version-specific endpoint, source, whole-schedule and sampler checks.

The HF uploader batches missing deltas and verifies each file and its archive ledger at the returned revision. A persistent cooldown handles the observed repository commit rate limit. Real verification of the revised batching protocol remains pending until the cooldown expires; full remote coverage is not yet claimed.

Credentials are outside the repository with restricted local permissions. The private Hugging Face archive receives verified deltas. The controller resumes when the VM boots; it cannot restart a stopped Spot VM itself.
