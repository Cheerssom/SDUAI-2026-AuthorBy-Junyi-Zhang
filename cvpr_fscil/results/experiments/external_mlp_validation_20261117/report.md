# Two-layer nonlinear MLP external-head validation

Standalone experiment. Three benchmarks (CIFAR-100, miniImageNet, CUB-200), seeds 0/1/2, five calibration images per base class, retention ratio 0.95, joint error budget 0.05. Table-S8 splits, base-only alignment tuning and paired betting certification are reused. External head is Linear(input, 256) -> ReLU -> Linear(256, classes). Architecture fixed before evaluation. Class-weighted cross entropy, Adam learning rate 0.001, 300 full-batch steps, initialization seed 2026. Weight decay independently selected from [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] using base tuning accuracy, both for the deployed head and pseudo-base tuning head. Calibration and test data are excluded from fitting and selection. Both variants retain the same external probe and use the same 3,636-setting base-only search. Novel support is five-shot. Final-session results only.

Per-query alignment and fixed-moment alignment each select their coefficients independently using base-only tuning data. All base-calibration recovery rates are asserted to be zero. Failed certifications contribute zero novel accuracy.

Confidence intervals use 20,000 paired bootstrap resamples of the nine dataset-seed differences, with RNG seed 2026, matching the existing experiment. These intervals describe these nine paired units, with only three datasets and shared frozen features.

| Encoder | Method | Admitted | Certified novel (%) | Conditional novel (%) | Paired improvement [95% CI] (pp) | Wins |
|---|---|---|---|---|---|---|
| b16 | AnchorCE_full | 8/9 | 48.81 | 54.91 | 5.83 [-7.47, 17.49] | 6/9 |
| b16 | Wrap_fixed_mom | 9/9 | 42.98 | 42.98 | Reference |  |

Mean deployed MLP base accuracy: 86.70%.
AnchorCE_full per-dataset certified novel accuracy: {'cifar100': 16.15000000000001, 'mini': 74.07500000000005, 'cub200': 56.206173147280275}
AnchorCE_full empirical test floors met: 9/9.
Wrap_fixed_mom per-dataset certified novel accuracy: {'cifar100': 31.866666666666692, 'mini': 65.08333333333339, 'cub200': 31.982175487347916}
Wrap_fixed_mom empirical test floors met: 9/9.

The experiment tests the external-probe wrapper with four available sources. It does not independently isolate the necessity of each source. This is one nonlinear architecture with a prespecified width and optimizer schedule.

Run commands

```powershell
python run_mlp_validation.py --encoder b16
python summarize.py
```
