# Cardinality-based reference interpolation

Fixed 2026-10-27 before this rule's new tuning or evaluation outcomes.
This is a post-development extension: B0=30 was proposed after inspecting
the existing eta=.5 Pets results. It is a fixed design constant, not a
derived empirical-Bayes optimum, an iid variance law, or a parameter-free
estimator. No improvement is presupposed.

Primary rule: eta=min(1,B/30), where B is the number of references actually
used at the current fitting stage. Thus pseudo/full Pets use7/30 and15/30;
CIFAR uses20/30 and40/30; ImageNet-R uses50/30 and100/30. Recompute the full
base-only pseudo-expansion fusion selection using this rule; freeze the
resulting coefficients before independent calibration and evaluation.
Do not reuse eta=.5 pseudo-tuned Pets weights as though they implemented
eta=7/30. A complete reference bank is fixed within each arrival path.

Secondary interpretation: compute eta once from the deployed initial
catalog's full base count, and freeze that same eta in pseudo tuning and
deployment. Call this catalog-frozen interpolation, not a per-stage
reference-cardinality rule. It reconstructs the existing Half Pets and
Query CIFAR/ImageNet-R records and is a previously observed control.
Both interpretations are reported; no test-driven winner selection.

Use all six existing CLIP/dataset conditions, ten partitions, two orders,
five arrivals, five-shot support and the original frozen common reference.
Each newly selected primary readout receives3636 base-only fusion settings.
Use original cap40 calibration and direct/protection fixed-sequence tests.
Report AIA, common95% target capacity with feasible counts, final direct
base/novel accuracy, conditional accuracy, finite admissions and test-target
attainment. Save all paired primary-vs-Query/None/Fixed/Half contrasts.

Keep original archived inputs, fits, supports and baseline records unchanged.
Atomically save data/code hashes, parameter selection and session predictions;
independently reconstruct all new threshold tests and verify restart. Serial
GPU lock, two CPU threads,50% CUDA process budget and shared PAUSE remain.
