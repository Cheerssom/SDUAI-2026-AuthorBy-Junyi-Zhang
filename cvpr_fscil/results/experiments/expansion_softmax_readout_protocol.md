# Frozen-feature softmax-readout transfer

Locked 2026-10-27 before new fit, selection or evaluation outcomes.

Question: do query-dependent reference moments improve recognition and
retention-constrained admission when the learned base evidence is fitted by
multiclass cross-entropy rather than ridge regression?

Primary scope: ImageNet-R, existing CLIP ViT-L/14 and ViT-B/32 caches, original
ten class partitions, two orders and five arrivals. No new encoder features,
data download, test split, support choice or calibration image is introduced.
The earlier six-condition ridge/dense fully retuned study supplies the frozen
controls, not newly fitted comparisons.

Base heads: affine multiclass softmax trained on the original fitting images,
with inverse-frequency weights normalized to a class-uniform mean. Optimize
weighted cross-entropy plus lambda/2 times the squared slope norm; leave the
intercept unregularized. Initialize weights and intercept to zero. Use full-batch
Adam, learning rate .05, betas .9/.999, epsilon 1e-8, 300 steps. Select lambda
from {1e-5,1e-4,1e-3,1e-2} by closed-base macro accuracy on base tuning queries,
with first-grid tie breaking. Full-base and pseudo-base fits select separately.
Save full optimizer state every 50 steps and reuse completed fits exactly.
No calibration or actual novel test label participates in fitting/selection.

Compare three normalization modes: query centering/query scaling; no centering
with query scaling; and fixed base-tuning moments. Every mode independently
selects base and novel coefficients from {0,.25,.5,1,2,4}, plus 101 margin
quantiles, by pseudo-expansion harmonic macro accuracy: 3636 settings per mode.
Identical prompts, reference banks, five-shot novel supports and scalar-offset
tie rules apply. Selection is saved before evaluating calibration/test queries.

Original ridge reference correctness and the absolute 95% retention target are
copied from the frozen retuning archives for each partition/order/session. This
keeps the accuracy contract identical to the ridge/dense study. Relative to this
common reference, a new softmax readout may recover old errors; no G=0 claim is
made for that cross-head comparison. Query-score preservation remains a property
of each fixed readout under vocabulary growth. The ridge mean identity 1/B does
not apply to softmax-fitted weights.

Report fixed-offset average incremental macro accuracy; exact test-label
frontiers at .80/.85/.90/.95/.99 of the common original reference; and the same
fixed-sample direct paired accuracy test with cap-40 independent calibration,
95% confidence over the five prespecified arrivals per method/order/partition.
Protection results are additional diagnostics. Report fallback-inclusive novel
accuracy, admission counts, empirical base accuracy/target attainment and
admission-conditioned means. Compare each calibrated policy to its own
common-floor frontier to separate empirical capacity from selection headroom.
Average orders within each partition for paired inference. Keep no-admission
and infeasible counts explicit; no test-label interpolation or tuning.

Pilot partition 0 checks execution and checkpoint correctness. The full run
uses the same locked grid and all ten partitions regardless of pilot direction.
Feature, input, code and prediction hashes are saved. GPU jobs are serialized;
CPU threads=2, CUDA process allocation capped at 50%. Honor the shared PAUSE
marker and persist atomic fit/selection/session checkpoints. No paid compute.
