# External deployed-probe validation on additional encoders

Standalone experiment. Three benchmarks (CIFAR-100, miniImageNet, CUB-200), seeds 0/1/2, five calibration images per base class, retention ratio 0.95, joint error budget 0.05. Existing table-S8 split, logistic-probe fitting, base-only tuning and paired betting certification are reused. Both variants retain the same external probe and use the same 3,636-setting base-only search. Novel support is five-shot. Final-session results only.

Per-query alignment and fixed-moment alignment each select their coefficients independently using base-only tuning data. All base-calibration recovery rates are asserted to be zero. Failed certifications contribute zero novel accuracy.

Confidence intervals use 20,000 paired bootstrap resamples of the nine dataset-seed differences, with RNG seed 2026, matching the existing experiment. These intervals describe these nine paired units, with only three datasets and shared frozen features.

| Encoder | Method | Admitted | Certified novel (%) | Conditional novel (%) | Paired improvement [95% CI] (pp) | Wins |
|---|---|---|---|---|---|---|
| b32 | AnchorCE_full | 9/9 | 53.10 | 53.10 | 13.17 [6.47, 19.84] | 7/9 |
| b32 | Wrap_fixed_mom | 9/9 | 39.93 | 39.93 | Reference |  |
| l14 | AnchorCE_full | 9/9 | 71.22 | 71.22 | 3.09 [-4.52, 10.36] | 5/9 |
| l14 | Wrap_fixed_mom | 9/9 | 68.14 | 68.14 | Reference |  |

The experiment tests the external-probe wrapper with four available sources. It does not independently isolate the necessity of each source. Generalization to other deployed-head families remains an empirical question.

Run commands

```powershell
python run_encoder_validation.py --encoder b32
python run_encoder_validation.py --encoder l14
python summarize.py
```
