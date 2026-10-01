# Recovery and mixed-objective experiments

Recovery annealing blended the pretrained BASE weights with fine-tuned candidate weights to study the tradeoff between specialized proxy accuracy and language-model retention. Ranking- and joint-specialized branches first produced large gains on the fixed reasoning proxy, alongside severe DATA-D and WikiText degradation. The recovery/interpolation search then measured whether smaller candidate weights could retain more of the base model's language behavior.

| Candidate | Blend weight | DATA-D ratio | WikiText BPB ratio | Proxy accuracy | GIBC mean |
|---|---:|---:|---:|---:|---:|
| BASE | 0% | 1.000× | 1.000× | 0.2686 | 0.4289 |
| 10M joint recovery | 5% | 1.016× | 1.026× | 0.2764 | 0.4288 |
| 10M joint recovery | 17% | 1.199× | 1.194× | 0.3672 | 0.4151 |
| 500K mixed objective | 15% | 1.168× | 1.162× | 0.3418 | 0.4154 |

The 5% recovery blend was close to BASE on retention and GIBC but added little proxy accuracy. The 17% blend preserved a larger proxy gain, with higher language-model losses and lower GIBC. The mixed objective showed the same separation between proxy and GIBC. Consequently, post-training selection retained BASE.

The follow-up's remaining blend weights were screened on DATA-D, WikiText, and the proxy. Their full GIBC gate was not run, so these candidates remain exploratory. See the [selection report](posttraining_final_report.md) and [machine-readable comparison](../metrics/final_posttraining_comparison.json).
