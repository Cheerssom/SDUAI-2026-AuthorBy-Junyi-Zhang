# Convolutional-backbone class-expansion validation

Locked 2026-10-27 before RN50 feature extraction, tuning or outcomes.

Use the official OpenAI CLIP RN50 (ModifiedResNet convolutional backbone,
with native attention pooling), checkpoint SHA256
afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762.
Official source:
https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
Use the installed official clip implementation, recording source hashes.
No new dependency or pretraining is required. Do not assume CNN features
will improve any readout or guarantee a direction for the contrasts.

ImageNet-R reuses the exact original byte-hashed image identities and pool
roles, ten partitions, two arrival orders, five arrivals, and five-shot
support indices. Native Resize224/CenterCrop224/BICUBIC/RGB normalization
and CLIP tokenizer (context77) extract normalized image features and the
original eight-template normalized text-embedding means. Every readout
uses the same cosine features; native global logit scale is not selectively
applied. No classification-model weight or preprocessing is tuned.

Full and pseudo-base weighted ridge heads select regularization from the
original .01/.1/1/10 grid on base tuning only. Retained and dense heads with
Query, None (no centering, query scaling) and Fixed moments each receive an
independent3636-setting fusion/offset search. HyCal uses the same matched
feature-level transfer and regularization search as the SigLIP control.
The encoder-specific learned/query readout supplies the common closed-base
reference, absolute95% target and fallback for every readout. The pseudo/full
base reference counts50/100 also make the proposed min(1,B/30) rule equal
to Query in this experiment; no new cardinality rule is selected on RN50.

Report AIA averaged across arrivals and both orders within partitions,
final common-floor capacity, direct and sufficient-protection policies,
actual base/novel test accuracy, finite admission and target-attainment
counts, fallback-inclusive and conditional means. Capacity uses test labels
for description; calibration selects offsets independently with original
cap40 images,95% retention and the same confidence allocation over five
prespecified arrivals. Confidence is not simultaneous over all configurations.

Atomic feature batches/selection/session records, source/data/model hashes,
native feature/logit witness, raw-prediction/threshold reconstruction and
warm-restart checks are required. Two CPU threads,50% CUDA process budget,
shared GPU serialization lock and PAUSE marker apply. Start image batches16,
halve on OOM. No public push or formal submission is performed.
