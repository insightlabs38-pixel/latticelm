# Post-training study

Post-training experiments tested whether targeted reasoning specialization could improve the broad capability profile of BASE. The experimental path screened optimizer and learning-rate choices, supervised fine-tuning and replay, ranking and joint objectives, recovery annealing, interpolation, and a mixed-objective follow-up. More expensive RFT and RLVR work was gated by preflight and time/resource eligibility; those branches are not presented as completed results.

## What the study found

Ranking and joint objectives raised the targeted reasoning proxy substantially. A proxy-only winner was not sufficient evidence for model selection: candidates were evaluated on DATA-D retention, WikiText BPB, and, for selected finalists, the full GIBC task suite. The strongest directly specialized candidates had very poor language-model retention and weak broad task accuracy. Recovery annealing improved retention, but larger recovery blends still failed the full GIBC improvement gate.

The follow-up study found a real proxy/retention frontier. A 5% BASE/recovery blend stayed close to BASE on retention and GIBC, with only a small proxy gain. A 17% blend gained more on the proxy but scored lower on GIBC. A 15% mixed-objective candidate followed the same pattern. The targeted proxy and GIBC therefore measured meaningfully different behavior; this is a useful result about evaluation and selection, not evidence that proxy gains transfer broadly.

| Candidate | DATA-D vs BASE | WikiText BPB vs BASE | Reasoning proxy | GIBC mean |
|---|---:|---:|---:|---:|
| BASE | 1.000× | 1.000× | 0.2686 | 0.4289 |
| 10M recovery, 5% | 1.016× | 1.026× | 0.2764 | 0.4288 |
| 10M recovery, 17% | 1.199× | 1.194× | 0.3672 | 0.4151 |
| Mixed objective, 15% | 1.168× | 1.162× | 0.3418 | 0.4154 |

DATA-D and WikiText ratios below 1.0 indicate better retention than BASE; ratios above 1.0 indicate worse loss/BPB. Proxy values are accuracy on a fixed 256-example screening set. GIBC is the full-suite mean. The 5%, 17%, and mixed 15% candidates received the additional full GIBC evaluation; other screened candidates without GIBC results remain unselected because their broad-evaluation gate is unmeasured.

BASE remains the selected model. The 17% recovery and mixed 15% candidates had higher proxy accuracy, but their GIBC means (0.4151 and 0.4154) were below BASE (0.4289). The 5% candidate was close to BASE on GIBC (0.4288) without a defensible broad improvement. The exact curated comparison is in [`../artifacts/metrics/final_posttraining_comparison.json`](../artifacts/metrics/final_posttraining_comparison.json); reports contain the full selection and run context.

## Evidence

- [Final post-training report](../artifacts/reports/posttraining_final_report.md)
- [Recovery experiment report](../artifacts/reports/recovery_experiment_report.md)
- [Curated comparison data](../artifacts/metrics/final_posttraining_comparison.json)
- [Final selection](../artifacts/metrics/final_selection.json)

![Post-training and retention comparison](../video/final/screenshots/06_recovery_evaluation.png)
