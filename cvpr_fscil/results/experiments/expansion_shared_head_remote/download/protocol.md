# Shared-expert group-gate diagnostic (frozen before new outputs)

Scope: SUN397 first, CLIP L/14 and B/32, ten original partitions, two five-arrival orders, 5-shot supports and original fitting/tuning/calibration/test memberships. ImageNet-R may follow with the same definitions. All numeric preparation, model computation, selection, calibration, validation and summaries run in the cloud; local machine only writes source/protocol, transfers existing immutable inputs and receives results.

For each actual or pseudo-base stage, freeze the two output experts using the existing base-only Full setting (a0,u0) and its Ridge weights/reference moments:
 hB(x)=argmax_b [zT(x,b)+a0*zL(x,b)],
 hN(x)=argmax_n [zT(x,n)+u0*zP(x,n)].
Both diagnostic gates output exactly hB if routed base and hN if routed novel. The actual closed-base predictor is the same original Full reference for both. Pseudo experts use the cached Wp and the same already-base-only-selected (a0,u0). No experts are reselected from the new results.

Compare two routing margins:
 Full-gate: mF=max_b[zT+a*zL] - max_n[zT+u*zP].
 Prototype-gate: mP=max_b[zT+a*zP_base] - max_n[zT+u*zP].
All branch normalized scores use fixed base-reference banks with the original C=30 stage-wise rule. These gate-specific a/u influence routing only; their maximizing label is NOT emitted. They are shared-expert diagnostics, not the original Full/Dense classifiers, and their output is not asserted to equal a global argmax of their own scoring matrix.

Each gate independently evaluates a,u in {0,.25,.5,1,2,4} and101 pseudo-tuning offset quantiles (3636 settings), maximizing macro harmonic accuracy using the fixed shared output experts already during pseudo selection. Class-ID tie rule: offset>margin selects novel; at equality, choose the lower of shared expert class IDs. This fully specifies even unusual tied gate-max vs emitted-class situations.

Same independent cap40 base calibration, same original candidate quantiles from actual base-only tuning, ratio=.95,delta=.05 across five prespecified arrivals, direct paired accuracy test. Because output hB and closed-base reference coincide for both gates, common/self are mathematically the same reference. Save both fields and assert identical p-values/offsets. Base labels cannot be recovered by choosing a novel label, so G=0; verify empirical correctness containment on calibration and test for every accepted/fixed threshold and sampled events. Record absolute closed/base accuracy, novel accuracy, fixed-offset AIA, final capacity, conservative actual-base matching, finite/met, D/G, and paired intervals across partition means. Match using exact discrete events and no interpolation.

Atomic selection/session checkpoints; feature/weights/source-selection hashes. Full per-trial offsets/harmonics and compact cal/test predictions archived, independent feature-to-score witness. All planned outcomes are retained. No original experiments or paper edits.
