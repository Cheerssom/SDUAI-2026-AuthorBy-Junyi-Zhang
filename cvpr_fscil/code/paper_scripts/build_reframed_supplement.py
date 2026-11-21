"""Supplement and numeric evidence guide for final reference contracts."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];E=H/'revision_evidence';D={p.stem:json.loads(p.read_text()) for p in E.glob('*.json')};out=[]
def add(s):out.append(s)
def longtable(header,rows,ncol,caption,label):
 return '\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{'+'l'*ncol+'}\n\\caption{'+caption+'}\\label{'+label+r'}\\'+'\n\\toprule\n'+header+r' \\ \midrule\endfirsthead'+'\n\\toprule\n'+header+r' \\ \midrule\endhead'+'\n'+'\n'.join(' & '.join(r)+r' \\' for r in rows)+'\n\\bottomrule\n\\end{longtable}\\endgroup\n'
add(r'''\FloatBarrier
\section{Reference contracts and full deployment readouts}\label{app:refcontracts}
Common-reference certification tests every candidate against the same preselected closed-base Full predictor. Self-reference certification tests preservation of each method's own frozen closed-base predictor, with its own fallback. The latter can change the absolute accuracy requirement. Both use the same independent base calibration images, margin candidates and fixed-time paired test, with $r=.95$, confidence .95, and allocation over five arrivals. Source-only Sketch uses no target calibration labels; its empirical test requirements are defined using the corresponding reference on Sketch.

The local completion study reuses frozen cal/test predictions and verifies 3,000 common-reference policies against the original records. HyCal's true closed-vocabulary predictions are reconstructed from cached cosine/Mahalanobis branches with only the base columns present before CDF normalization. Expanded-vocabulary base winners are never substituted for a closed HyCal reference. SUN cache self-reference uses an independent CPU FP32 reconstruction with the original frozen affinity parameters, without fitting or image encoding; its common-reference means and admission counts reproduce the GPU results. Zero-recovery containment holds for every invariant within-base readout relative to itself, including Dense. It is not exclusive to a Ridge head.

Tables list final means; SUN/Food average two orders within ten partitions, Sketch has one source-defined order. The reference floor is $.95 A_{\rm ref}$ measured on the fixed test pool; it is not a recalibration on test labels. Test labels evaluate selected policies only. Raw partition records, accepted offsets and input hashes accompany the release.
''')
rows=[]
for a in D['references']['summary']+D['cache_references']['summary']:
 if a['dataset']=='sketch' and a['calibration']!='source_cal':continue
 rows.append([a['dataset'],a['encoder'],a['method'].replace('_',r'\_'),a['reference'],f"{.95*a['closed_base']:.2f}",f"{a['base']:.2f}",f"{a['novel']:.2f}",f"{a['finite']}/{a['total']}",f"{a['met']}/{a['total']}"])
add(longtable('Data & Enc. & Readout & Ref. & Floor & Base & Novel & Finite & Met',rows,9,'Completed common/self contracts (\\%). Main-table Full uses cardinality-aware moments; Sketch uses the unshrunk frozen source readout. Counts identify useful admission separately from empirical target attainment.','tab:completedreferences'))
rows=[]
for a in D['source_references']['summary']:
 if a['method'] not in ['retained_query','dense_query','retained_none','retained_fixed']:continue
 rows.append([a['dataset'],a['encoder'],a['method'].replace('_',r'\_'),a['reference'],f"{.95*a['closed_base']:.2f}",f"{a['base']:.2f}",f"{a['novel']:.2f}",f"{a['finite']}/20",f"{a['met']}/20"])
for a in D['hycal_references']['summary']:
 rows.append([a['dataset'],a['encoder'],'HyCal',a['reference'],f"{.95*a['closed_base']['mean']:.2f}",f"{a['base']['mean']:.2f}",f"{a['novel']['mean']:.2f}",f"{a['finite']}/20",f"{a['met']}/20"])
add(longtable('Data & Enc. & Readout & Ref. & Floor & Base & Novel & Finite & Met',rows,9,'Source readout sensitivity after independent 3,636-setting fusion tuning. These historical Query variants use $\\eta=1$ at both pseudo and full stages; they are not relabeled Cardinality-aware Full. HyCal uses its own separately selected original regularization.','tab:sourcerefcontracts'))
add(r'''Switching to a self reference materially changes admission: Dense can preserve its own lower base ceiling while admitting more novel queries. This comparison answers a different question from requiring the same absolute base accuracy for all systems. HyCal's self-reference final novel accuracies are 52.79/33.88\% on CIFAR, 34.31/30.73\% on Pets, and 61.00/30.78\% on ImageNet-R (L/14/B/32). True closed-to-expanded base ranking changes are .012--.793 points. The data therefore separate reference-ceiling mismatch from ranking changes and do not attribute all rejected admission to recovery variance.

\FloatBarrier
\section{Cache and shared-expert attribution}\label{app:cacheexpert}
The original SUN Dense readout already selects independent base/prototype coefficients and a global novel offset over $6\times6\times101=3,636$ settings. It is therefore a calibrated-stacking comparator; no identical algorithm is counted a second time. The additional controls are exponential caches with class sums or class means, scored as $100q_T+\alpha q_{\rm cache}$. Each independently evaluates six affinities, six mixing weights and 101 base-only offset quantiles. They share the same fitting/support access, cap-40 calibration and common accuracy requirement.
''')
rows=[]
for a in D['sun_cache']['summary']:
 cv=a['capacity']['mean'];rows.append([a['encoder'],a['method'].replace('_',r'\_'),f"{a['aia']['mean']:.2f}",'--' if cv is None else f'{cv:.2f}',f"{a['base']['mean']:.2f}",f"{a['novel']['mean']:.2f}",f"{a['finite']}/20",f"{a['met']}/20"])
add(longtable('Enc. & Cache & AIA & Capacity & Base & Novel & Finite & Met',rows,8,'SUN cache controls under the original common Full reference (\\%). A capacity -- denotes no jointly feasible partition.','tab:cacheabsolute'))
rows=[]
for a in D['sun_cache']['contrasts']:
 if a['comparator']=='cache_mean':rows.append([a['encoder'],{'Full-minus-cache certified novel':'Policy','Full-minus-cache oracle capacity':'Common-floor capacity','Full policy minus cache oracle matched to Full actual base accuracy':'Matched to Full base'}[a['metric']],f"{a['mean']:+.2f} [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}]",str(a['n'])])
add(longtable('Enc. & Full minus class-mean cache & Gain (pp) & Paired n',rows,4,'Cache contrasts with paired 95\\% partition intervals. B/32 has 20/20 admission for both methods. L/14 selected-policy differences include fallback. The matching direction fixes Full actual base accuracy and searches the cache frontier, with no interpolation.','tab:cachecontrasts'))
add(r'''The shared-expert diagnostic fixes both emitted labels: the original Full Ridge+text winner $h_B$ and original Full text+prototype winner $h_N$. Two gates form margins from either learned or prototype base evidence, with separate $a,u,d$ selection over the same 3,636 base-only settings, but always emit those shared experts. The same replacement is applied during pseudo-base tuning. Gate-specific score maxima determine only routing, not the emitted class. Thus these controls are explicitly shared-expert diagnostics, not the original Dense or Full classifiers. Both have $G=0$ relative to the same closed-base expert; common and self tests coincide.
''')
rows=[]
for a in D['shared']['contrasts']:
 f=lambda k:f"{a[k]['mean']:+.2f} [{a[k]['ci95'][0]:+.2f}, {a[k]['ci95'][1]:+.2f}]"
 rows.append([a['encoder'],f('capacity'),f('policy')])
add(longtable('Enc. & Capacity difference & Certified novel difference',rows,3,'Shared experts: learned-score gate minus prototype-score gate (pp), paired 95\\% intervals over ten partitions.','tab:sharedexperts'))
add(r'''The prototype-score gate has a small advantage in this control. Consequently, the larger original Full--Dense differences cannot be assigned uniquely to the learned-score margin: aligned evidence, within-group classifiers and selected operating points contribute jointly. The moment-permutation experiment isolates query correspondence through a different intervention. Four independent CPU FP32 cache witnesses reproduce GPU class predictions on 128 queries per condition (maximum score error $6.87\times10^{-5}$); all 400 shared-expert method/session records are checked from saved predictions.

\FloatBarrier
\section{CUB-200: official sequence and matched frozen-feature transfers}\label{app:cubtransfer}
We retain the official TOPIC CUB session indices: 3,000 training images from 100 base classes, followed by ten arrivals of ten classes with five supports each. Source indices are pinned to commit \nolinkurl{e76d37c9bdb64474b323ee97c312148fb704e91f}. The official 11,788-image archive has MD5 \nolinkurl{97eceeb196236b17998738112f37df78}. Both CLIP revisions, eight templates, every image hash and task membership are fixed before readout evaluation.

The standard track refits on all official base training images, with an internal five-per-class tuning split for model selection, then uses the official incremental supports and test membership. The certification track reserves five tuning and ten calibration images per base class before fitting on the other 15, identically for all methods. It uses split seeds 0,1,2 on one fixed class order and allocates the .05 confidence budget over ten arrivals. No official test labels select parameters or calibration thresholds. Three split intervals condition on the official sequence and do not measure uncertainty over new class orders.

Full, None, Fixed and Dense independently select 3,636 base-only pseudo-expansion settings. Full equals its unshrunk case here because actual/pseudo banks contain 100/50 classes. SimpleCIL uses nearest normalized means. RanPAC transfer uses 10,000 Gaussian random features with ReLU and the analytic ridge solution, implemented in dual form; a base-only 80/20 split selects ridge regularization once, which stays fixed across arrivals. This differs from original PETL and per-arrival ridge selection. BiMC transfer uses author-provided CUB descriptions, .1 visual calibration with temperature 16, .5 name/description fusion, visual/text mixing .3, and the non-ensemble softmax readout. HyCal uses the previously verified fused cosine and regularized covariance scoring with fixed regularization .001. These are matched frozen-CLIP transfers, not claims to reproduce original published backbone or training scores.
''')
rows=[]
for a in D['cub']['summary']:
 if a['track']=='standard':rows.append([a['encoder'],a['method'].replace('_',r'\_'),f"{a['aia_top1']:.2f}",f"{a['final_top1']:.2f}",f"{a['aia_macro']:.2f}"])
add(longtable('Enc. & Method & AIA Top-1 & Final Top-1 & AIA macro',rows,5,'CUB standard-track accuracies (\\%), official fixed class/support sequence. No interval over independent class orders is implied.','tab:cubstandard'))
# Conditional intervals use fixed existing split records, not new fitting or test selection.
raw=R/'experiments/expansion_cub_remote/download/readouts_v2';rng=np.random.default_rng(2026);display=[];rows=[]
for a in D['cub']['summary']:
 if a['track']!='certified':continue
 for ref in ['common','self']:
  v=[]
  for seed in range(3):
   z=json.loads((raw/a['encoder']/'certified'/f'seed{seed}'/'session10.json').read_text());q=next(r for r in z['rows'] if r['method']==a['method'])[ref];v.append(100*q['novel'])
  x=np.array(v);boots=x[rng.integers(3,size=(20000,3))].mean(1);lo,hi=np.quantile(boots,[.025,.975]);q=a[ref];rows.append([a['encoder'],a['method'].replace('_',r'\_'),ref,f"{q['base']:.2f}",f"{q['novel']:.2f} [{lo:.2f}, {hi:.2f}]",f"{q['finite']}/3",f"{q['met']}/3"]);display.append(dict(encoder=a['encoder'],method=a['method'],reference=ref,values=v,ci95=[float(lo),float(hi)]))
(E/'cub_split_intervals.json').write_text(json.dumps(display,indent=2)+'\n')
add(longtable('Enc. & Method & Ref. & Base & Novel [95\\% interval] & Finite & Met',rows,7,'CUB certified-track final accuracies (\\%). Conditional split-bootstrap intervals use only three training/calibration splits; they do not represent class-order uncertainty. All raw split results are included.','tab:cubcertified'))
add(r'''All 704 method/session records are retained. Independent reconstruction checks 3,352 accepted/rejected statistical tests directly from correctness counts and fixed mixture weights. Fitting/tuning/calibration index separation and all 135 returned-file hashes are verified. The feature witnesses compare cached mixed-precision extraction with FP32 encoding on eight images (minimum cosine .9999956/.9999835 for B/32/L/14).

\FloatBarrier
\section{Additional interpolation-constant sensitivity}\label{app:additionalconstants}
Constants $C\in\{10,20,30,50\}$ evaluate the fixed family $\eta_B=\min(1,B/C)$; the main constant remains 30. Each true base-cardinality task refits its complete Ridge predictor before the original study; the additional constants reuse those frozen weights and data roles and independently tune fusion under the same 3,636-setting budget. They do not replace the main rule by the test-best constant. The study spans $B=5,10,15,20,30,50,100$, ten partitions, two orders and five arrivals on ImageNet-R with both encoders.

The 2,800 additional method/session results are reconstructed from saved predictions, while 1,400 prior C20/C30 control records remain byte-identical. At $B=100$, all four constants yield the same operation and selections; 400 method/session outcomes are identical to C30. Small banks show both recognition and statistical-power effects, rather than a universally optimal constant.
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{constant_sensitivity_new.pdf}
\caption{True base cardinality and prespecified interpolation constants. Each task uses its own refitted base classifier; the main rule remains C30.}
\end{figure}
''')
rows=[]
for a in D['constants']['rows']:
 def value(k):
  x=a[k];return f"{x['mean']:.2f}" if isinstance(x,dict) else str(x)
 rows.append([a['encoder'],str(a['B']),str(a['C']),value('capacity'),value('novel'),str(a.get('finite','--')),str(a.get('met','--'))])
add(longtable('Enc. & B & C & Capacity & Certified novel & Finite & Met',rows,7,'All prespecified constant/cardinality conditions (\\%). Means average orders within partitions. Full paired intervals and per-partition values are included in the constants summary.','tab:additionalconstants'))
(H/'supplement_reframed_evidence.tex').write_text('\n'.join(out),encoding='utf-8')
# Reading guide and historical statements follow the final main-paper organization.
p=H/'supplementary.tex';s=p.read_text();s=s.replace('Main Fig.~\\ref{main-fig:capacity} visualizes the shared-retention frontiers; the tables in this section additionally retain partially feasible sets and exact counts.','The tables in this section retain the historical shared-retention frontiers, partially feasible sets and exact counts. Main Fig.~\\ref{main-fig:capacity} presents the separate moment-permutation controls.')
a=s.index('Main Table~\\ref{main-tab:recognition} also reports final novel accuracy');b=s.index(' It retains normalized',a);s=s[:a]+'The historical Full-query recognition and no-fallback tuning-selected outcomes are retained in Table~\\ref{tab:historicaldualview}, separate from the unified Full comparison in the main paper.'+s[b:]
legacy=r'''\begin{table}[htbp]
\centering
\caption{Historical Full-query recognition, no-fallback tuned novel accuracy and common-reference certified novel accuracy (\%). Recognition uses the compact source grid; the tuned and certified columns share independently fitted readouts. This preserves earlier numbers without relabeling them as unified Full.}
\label{tab:historicaldualview}
\resizebox{\linewidth}{!}{\input{table_cvpr_dualview}}
\end{table}
'''
s=s.replace('\\subsection{Recent HyCal baseline under matched access}',legacy+'\n'+r'\subsection{Recent HyCal baseline under matched access}')
s=s.replace('The main retention/value evaluation uses the common expanded grid in Section~\\ref{main-sec:calibration}.','This historical retention/value evaluation uses its expanded grid in Section K; unified main comparisons use 3,636 settings.')
s=s.replace('\\input{supplement_tnorm}',r'\input{supplement_tnorm}'+'\n\n'+r'\input{supplement_reframed_evidence}')
s=s.replace('Section AB gives the explicit fixed-cohort T-norm comparison.','Section AB gives the explicit fixed-cohort T-norm comparison. Sections AC--AF cover reference sensitivity, cache/shared-expert attribution, the official-sequence CUB transfer study, and additional fixed constants.')
s=s.replace('The main paper follows recognition, query-adaptive alignment, constrained admission and frozen-model transfer.','The main paper develops the fixed-reference design principle, recognition gains, query-moment and shared-expert controls, reference sensitivity, and frozen-model transfer.')
p.write_text(s)
p=H/'supplement_new_controls.tex';s=p.read_text().replace('distinct from the original fixed-fusion models in main Table~\\ref{main-tab:recognition}.','distinct from the historical Full-query source models in Table~\\ref{tab:historicaldualview}.');p.write_text(s)
print('Supplement AC--AF generated')
