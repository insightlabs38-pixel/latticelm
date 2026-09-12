# Post-900M report

Classification: **INCONCLUSIVE**

The immutable 900M checkpoint `68fe5a3609919e7201557dfaf54320bd1adda10181904ceb09012735fd8afa99` passed model, checkpoint, dataset, exact-resume-field, and numerical-health certification. The complete 32M trajectory was analyzed with all-history and late-window asymptotic power-law fits.

WikiText PPL: 47.343549
WikiText BPB: 1.778074
HellaSwag: 0.263792
ARC-Easy: 0.308923
PIQA: 0.531012
WinoGrande: 0.484609

Post-training is eligible only after the best pretrained base checkpoint is selected. No external pretrained teacher, judge, reward model, or evaluation data is permitted. 40M and 49M were screened out before training; they were not failed trained models.
