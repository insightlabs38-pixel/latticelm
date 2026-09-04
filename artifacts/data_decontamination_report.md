# Phase 7A practical decontamination

FineWeb-Edu candidates were screened before either training or validation assignment against the validation and test text available for HellaSwag, ARC-Easy, PIQA, WinoGrande, and WikiText-103. Benchmark material was used only to construct fingerprints.

Text was NFKC-normalized, lowercased, and reduced to alphanumeric words. A document was removed when it contained either a normalized reference string of at least 80 characters or at least two matching 13-word n-grams. The implementation uses deterministic 64-bit BLAKE2b n-gram fingerprints. This is a practical exact/long-overlap screen, not a proof of perfect decontamination and not a semantic paraphrase detector.

Practical decontamination screening found/removed **9 candidate overlaps** under the documented heuristic among 16,677 streamed candidates: seven nonexclusive WikiText-103 flags and two HellaSwag flags. No ARC-Easy, PIQA, or WinoGrande overlap crossed the threshold. A further 2,090 candidates failed the minimum-length or English-score quality filters; 13,875 entered training and 703 entered the independently hashed validation partition. Document-level removals are in `data_decontamination.csv`; aggregate machine-readable counts are in `data_decontamination_summary.json`.
