# Cross-pretraining encoder validation

Locked 2026-10-27 before new encoder features or recognition outcomes.

Use google/siglip-base-patch16-224, pinned revision
7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed. This changes pretraining objective
and model family while retaining a ViT backbone; it is not a CNN experiment.
No prediction is made about the direction or size of a SigLIP gain.

ImageNet-R only: the original ten partitions, two orders, five arrivals,
five-shot supports and byte-disjoint fitting/tuning/calibration/test identities.
Extract frozen image features with the model's own official preprocessing,
and eight-template text features using the original prompt list. Normalize
each embedding and the average normalized prompt embedding. Apply this same
feature/preprocessing interface to every transferred readout. The SigLIP text
and visual encoders remain frozen; learned base heads are weighted ridge fits.
Treat cosine embeddings as the shared representation; do not selectively add
SigLIP's logit scale/bias to one readout. Preserve raw feature/model hashes.

Select ridge regularization from the original {.01,.1,1,10} closed-base
tuning grid, separately for full and pseudo base fits. Query, no-centering
with query scaling, and fixed-global moments each independently select
base/novel coefficients {0,.25,.5,1,2,4} and101 tuning-margin quantiles.
Dense prototypes use the same independent two-coefficient grid. The six
readouts (learned/dense x query/none/fixed) each receive3636 fusion settings.
Transfer HyCal's original feature-level algorithm with the same data access,
regularization grid and eight prompts as the previous matched control. This
is not the original published HyCal benchmark or native SigLIP recipe.

The independently tuned learned/query readout supplies this encoder's
common closed-base reference and95% absolute retention target. Tune one
reference once before calibration/test scoring; use it for every method.
Report fixed-offset AIA, test-label capacity at80/85/90/95/99% retention,
and independently selected direct/protection policies under the same cap40
calibration and five-session95% confidence allocation. Disclose fallback,
finite counts, actual base accuracy and test-target attainment. Average
orders within ten partitions for paired bootstrap intervals. Do not compare
absolute target laws across encoders as if the references were identical.

Feature extraction is batch-resumable and encoding uses eval mode with no
parameter updates. Atomic batches, selections and sessions use source hashes.
Two CPU threads, serialized GPU,50% CUDA process budget, batch starts16 and
halves on OOM. Honor the shared PAUSE marker; no paid compute. Record
actual elapsed time and disk use rather than assuming a few-minute runtime.
