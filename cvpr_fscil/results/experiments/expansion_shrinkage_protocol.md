# Reference-moment shrinkage and cardinality intervention

Locked2026-10-27 before new shrinkage selection or evaluation outcomes.

Question: can tuning-only interpolation between query and fixed reference
moments improve the stability of the readout on Pets? Reference scores are
correlated class responses, not iid samples: no O(1/B) sampling law or causal
explanation of the existing dataset difference is assumed.

Use all six original dataset/CLIP-encoder conditions, ten partitions,
two orders and five arrivals. Fits, data roles, supports, prompt/reference
banks, original ridge reference and cap40 calibration match the frozen
full-fusion retuning study. No feature extraction or model refitting.

Define mu_eta=eta*mu_query+(1-eta)*mu_fixed and
sigma_eta=eta*sigma_query+(1-eta)*sigma_fixed, with epsilon floor1e-6.
Evaluate eta in{0,.5,1}; eta0 and1 exactly reproduce fixed and query moments.
Do not call this estimator empirical Bayes without an explicit probabilistic
derivation. Each eta receives the original3636-setting independent coefficient/
offset search on the base-only pseudo-expansion tuning queries.

Two prespecified selection policies: (a) eta in{0,.5,1} from the same tuning
harmonic objective, first-grid tie breaking; (b) eta in{0,.5,1} plus the existing
uncentered/query-scaled scheme, same objective. The latter explicitly permits
uncentered normalization rather than assuming moment interpolation dominates
it. Record budgets10908 and14544 versus3636 per fixed scheme. Include an
equal-total-budget expanded Query search if making an adaptive-versus-Query
efficiency claim; otherwise report the extra search access explicitly.
Fusion and eta are frozen at their pseudo-base choice for actual arrivals;
do not vary eta with actual B after selection. Existing query/none/fixed
results remain unmodified and separately reported.

Evaluate all candidates and selection policies under the original common
95% absolute retention floor and identical independent direct calibration.
Report AIA, common-floor capacity, paired Query/None/Fixed comparisons,
all-partition novel accuracy, admission and target-attainment counts, plus
how often each eta/scheme was selected. No success or elimination of Pets
variability is presupposed. Test images do not select policies.

Cardinality diagnostic: on each original selected model, hold its class
scores/weights, candidate set and fitting fixed; change only the fixed
reference-bank subset. Repeated nested subsets of sizes2,4,7,15 and the
available full bank, using deterministic seeds0..19, measure deviations
in query moments from full-bank moments and fluctuations in the base--novel
margin. Use base-tuning queries only for this intervention. Evaluate text,
learned and visual reference banks separately and jointly. The diagnostic
can support sensitivity to reference choice, not an iid asymptotic theorem
or a complete causal attribution of cross-dataset outcomes.

Save source/input hashes, atomic selections and sessions; reuse existing
Query/None/Fixed artifacts after verifying endpoint agreement. Limit2 CPU
threads and50% CUDA allocation, honor PAUSE and support restart.
