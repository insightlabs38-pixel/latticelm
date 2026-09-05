# Phase 7F decision

The controlled comparison is inconclusive because DATA-D-BROAD-v1 is internally infeasible under its declared constraints, not because of a model result.

SHOULD DATA-D CONTINUE TO 50M/100M: INCONCLUSIVE

SHOULD ~24M MODEL BE TESTED NEXT: NOT YET

SHOULD DATA-C 100M CONTINUE TO 150M: LOWER PRIORITY

SHOULD POST-TRAINING BEGIN NEXT: NOT YET

## Top three proposed Phase 7G actions (not launched)

1. Resolve the specification explicitly: permit already-consumed BabyLM as an intentional repeated component, approve a genuinely non-overlapping BabyLM partition/version, or replace that 15% with a newly approved source.
2. Complete the 1M–5M acquired-source fixture and full model/optimizer exact-resume test, then audit provenance and DATA-C overlap before materializing 100M tokens.
3. After the corrected acceptance gate passes, run exactly one from-scratch Co4-L 25M comparison with the frozen Phase 7D controls.
