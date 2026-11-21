from pathlib import Path
p=Path(__file__).resolve().parent/'main.tex';s=p.read_text(encoding='utf-8')
def para(start,end,new):
 global s
 a=s.index(start);b=s.index(end,a);s=s[:a]+new+'\n\n'+s[b:]
para('We evaluate CIFAR-100', 'Means and sample standard deviations',r'''We evaluate CIFAR-100~\cite{krizhevsky2009cifar}, Pets~\cite{parkhi2012cats} and ImageNet-R~\cite{hendrycks2021manyfaces} with frozen CLIP L/14 and B/32, five-shot support, ten frequency-stratified partitions and two five-session orders. Base/novel counts are 40/40, 15/15 and 100/100; arrivals add eight, three or 20 classes. Base fitting is long-tailed; ImageNet-R splits exclude byte-identical duplicates. Novelty is task-relative, with CLIP pretraining exposure retained (Supplements C, I, K).

Comparators share encoders, partitions and support. Matched HyCal transfers the author's cosine/Mahalanobis metrics, count weighting and CDF calibration, with base-only regularization/offset selection (Supplement M). Dense prototypes, caches, removal and raw/fixed fusion isolate readout components; Supplements A--H give auxiliary comparisons.''')
para('Optional target recalibration uses',r'Fixed-reference moments adapt',r'''Optional target recalibration changes only the offset using independent labeled base queries and matched per-class counts. Novel accuracy rises to 86.82\%/58.96\% at 90.76\%/78.87\% base accuracy (L/14/B/32), with all ten paths admitted and targets met (Fig.~\ref{fig:sketchMain}). Supplement R gives disjoint pools and full policies.''')
s=s.replace('Dense accuracy conditional on its six admissions is 61.00\\%; comparing this with the retained ten-partition mean is not a paired conditional contrast.',r'Dense achieves 61.00\% conditional on its six admissions; the retained 72.00\% averages all ten partitions.')
p.write_text(s,encoding='utf-8')
