# DATA-D-BROAD-v1 decontamination report

Status: SYNTHETIC TESTS PASS; PRODUCTION SCREEN NOT RUN

The implementation detects normalized exact containment, repeated long 13-word n-grams, and bounded SimHash near duplicates, quarantining whole matching documents with source, rule, benchmark, document ID, match fingerprint, and version metadata. Synthetic positives and a clean negative pass without embedding real benchmark examples in tests.

Production removal counts are unavailable, not zero: HellaSwag, ARC-Easy, PIQA, WinoGrande, WikiText-103, and common-validation screening did not run. No claim of perfect decontamination is made.
