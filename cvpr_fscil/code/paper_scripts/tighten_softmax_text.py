from pathlib import Path
p=Path(__file__).resolve().parent/'main.tex';s=p.read_text()
a=s.index('Every comparator shares');b=s.index(r'\subsection{Recognition across',a)
s=s[:a]+r'''All comparators share encoders, partitions and support. HyCal transfers the author's cosine/Mahalanobis metrics, count weighting and CDF calibration to the same prompts and frozen features; base-only pseudo-expansion selects its regularization and offset. Supplement M gives the pinned implementation and numerical fixtures. Dense prototypes, caches, removal and raw/fixed-temperature fusion provide controlled readouts; Supplements A--H contain auxiliary FeNeC and analytic comparisons.

We average orders within partitions. Means, sample standard deviations and paired 95\% bootstrap intervals describe ten class assignments on fixed images; intervals resample partition means 20,000 times. Average incremental accuracy averages five arrivals. Constrained paired contrasts use jointly feasible partitions.

'''+s[b:]
a=s.index('The fixed-reference design isolates');b=s.index('Pets illustrates',a)
s=s[:a]+r'''The vocabulary-growth study measures zero base-score drift and within-base ranking churn, with identical final predictions across two arrival orders under fixed class-ID ties. Supplement A gives the three-partition stability curves; Supplement I gives ten-partition paired classifier-value estimates. New classes can still win the expanded decision.

'''+s[b:]
a=s.index(r'\paragraph{Calibration coverage and retained error structure}');b=s.index(r'\subsection{Frozen-model transfer',a)
s=s[:a]+r'''\paragraph{Calibration coverage and retained error structure}
For the original retained readout, $G=0$ makes protection coincide with retention. At cap 40, all six dataset/encoder conditions admit the final session in 10/10 partitions with all test targets met. L/14 novel accuracy is 72.76\% on CIFAR, 88.51\% on Pets and 80.74\% on ImageNet-R. Supplements L and N report complete arrival paths, early fallback and both certificates. Calibration uses 1,600 base labels on CIFAR, 600 on Pets and 3,636--3,778 on ImageNet-R, additional to five-shot support. The nested-cap study fixes all models and tuning parameters.

'''+s[b:]
p.write_text(s)
p=Path(__file__).resolve().parent/'supplementary.tex';s=p.read_text()
s=s.replace('supplementary_summary.json','supplementary_summary.json')
# Keep a long affine-head description comfortably inside the supplement line.
s=s.replace('class-uniform inverse-frequency-weighted cross-entropy','cross-entropy with class-uniform inverse-frequency weights')
p.write_text(s)
