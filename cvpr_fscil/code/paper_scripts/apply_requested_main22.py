from pathlib import Path
import json
H=Path('paper/cvpr2027');p=H/'main.tex';s=p.read_text(encoding='utf-8')
def change(a,b):
 global s
 assert a in s,a[:100]
 s=s.replace(a,b,1)
change('The alignment must adapt to queries and preserve its decision frame across arrivals.','The alignment must adapt to queries and preserve its decision frame across arrivals. We study long-tailed base supervision, asymmetric few-shot expansion and explicit retention certification, complementing balanced FSCIL benchmarks that emphasize unconstrained average accuracy.')
change("Our first-principles structural design instead derives the query gate from fixed references and the ridge identity, avoiding an additional routing fit to that surrogate distribution.","Our structural design derives a parameter-free query gate from fixed references and the ridge identity. The same anchored evidence also supports learned routing when a larger base vocabulary supplies richer pseudo-expansion tasks.")
a=s.index('BiMC~',s.index('\\paragraph{Calibration and evidence alignment}'));b=s.index('\n\\section{Reference-anchored',a)
s=s[:a]+r'''Cohort score normalization has a long history: T-norm uses test-dependent cohort responses, while Z-norm estimates target-dependent score distributions~\cite{auckenthaler2000score,mariet hoz2005unified}. Vision--language calibration similarly rescales sample logits~\cite{murugesan2024robust}; BiMC~\cite{chen2025bimc} and HyCal~\cite{lee2026hycal} align heterogeneous metrics. Our contribution connects fixed base cohorts to vocabulary-growth invariance, the ridge mean identity and a zero-recovery correctness coupling. Calibrated seen/unseen recognition~\cite{chao2016calibrated} supplies a related operating trade-off; independent offset selection specializes risk-controlled selection~\cite{angelopoulos2021ltt} and bounded-mean betting~\cite{waudby2024betting} to class-macro retention.
''' .replace('mariet hoz','mariethoz')+s[b:]
change('To satisfy Condition 1 without fixing the query response, we anchor branch moments to base references.','To satisfy Condition 1 without fixing the query response, we anchor branch moments to base references. The standardization is a cohort/T-norm operation~\\cite{auckenthaler2000score,mariethoz2005unified}; fixing its cohort across arrivals connects it to the invariance and error-coupling results below.')
change(r'''\begin{proof}
Every term in Eqs.~\eqref{eq:reference}--\eqref{eq:base-novel} for an existing class remains fixed. Hence each pairwise score difference remains fixed.
\end{proof}''',r'''This follows because every existing-class term in Eqs.~\eqref{eq:reference}--\eqref{eq:base-novel} remains fixed.''')
change('Supplements V and W compare this gate with base-only logistic and MLP routers; Supplement X tests which query statistics carry its recognition benefit.','Router choice interacts with the available base vocabulary: Pets and CIFAR provide only 7 and 20 pseudo-base classes, whereas SUN397 provides 97. Closed-form and learned gates share anchored evidence; Section~4.3 compares their observed operating regimes.')
change('Test-label sweeps measure capacity; independent calibration chooses deployable offsets.',r'''\textbf{Oracle frontier capacity} uses test-label sweeps to maximize novel accuracy within this fixed score family at a stated base target. \textbf{Certified deployed accuracy} evaluates the offset selected on independent calibration data, including zero novel accuracy on fallback. Matching actual base accuracy compares frontier points at the same achieved requirement.''')
a=s.index('\\caption{Full-query recognition');b=s.index('\n\\label{tab:recognition}',a)
s=s[:a]+r'''\caption{Full-query recognition and independently certified deployment (\%). AIA: all-class accuracy averaged over five arrivals and two orders, mean $\pm$ partition standard deviation. Tuned: final novel accuracy at the base-only tuning-selected offset, without a retention constraint or fallback. Certified: final novel accuracy under the common Ridge-reference 95\% requirement; parentheses count finite admissions out of ten, with zero on fallback. Tuned and certified columns use the same independently fitted readouts, order 0 and cap-40 calibration for certification. Retained uses Full-query ($\eta=1$); Table~\ref{tab:prospective} evaluates Full. Bold marks the largest AIA and certified mean.}'''+s[b:]
s=s.replace(r'\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}\input{table_cvpr_dualview}',r'\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}\input{table_cvpr_dualview}')
a=s.index('\\caption{New-domain recognition');b=s.index('\n\\label{tab:prospective}',a)
s=s[:a]+r'''\caption{New-domain recognition and query correspondence, Cardinality-aware Full. Readouts independently select 3,636 base-only settings. (a) Full minus Dense: oracle capacity at the shared 95\% base floor, capacity at Dense's achieved base accuracy, and certified deployed novel accuracy. Both methods admit and meet targets on all 20 final paths for each dataset/encoder. (b) Three permutations preserve moment distributions but break query correspondence. Paired 95\% intervals average two orders within ten partitions; frontiers use no interpolation.}'''+s[b:]
s=s.replace(r'\tabcolsep}{7pt}\renewcommand{\arraystretch}{1.08}\input{table_cvpr_prospective}',r'\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.08}\input{table_cvpr_prospective}')
change('Novelty is task-relative, with CLIP pretraining exposure retained (Supplements C, I, K).','Novelty is task-relative, with CLIP pretraining exposure retained. Food-101 and SUN397 add 50/50 and 195/195 base/novel classes; ImageNet-Sketch tests frozen-source rendition transfer. These six datasets span object, fine-grained, scene and rendition recognition under retention-certified asymmetric expansion, rather than the standard balanced CUB-200/miniImageNet FSCIL protocol.')
change('Comparators share encoders, partitions and support. Matched HyCal transfers the author\'s cosine/Mahalanobis metrics, count weighting and CDF calibration, with base-only regularization/offset selection (Supplement M). Dense prototypes, caches, removal and raw/fixed fusion isolate readout components; Supplements A--H give auxiliary comparisons.',"Comparators share encoders, partitions and support. Matched HyCal transfers the author's cosine/Mahalanobis metrics, count weighting and CDF calibration, with base-only regularization/offset selection. The supplement's evidence guide locates implementation/protocol details (A--M), calibration and transfer (N--R), classifier/router controls (S--V), and new-domain/mechanism studies (W--AA).")
change(r'''Table~\ref{tab:recognition} and the original source/Sketch studies use Full-query ($\eta=1$). The main Cardinality-aware (Full) model is evaluated in Table~\ref{tab:prospective} and Supplements U, W--Y; its rule is applied to each actual or pseudo-base bank. Equal full-bank weights do not imply identical fitted models when pseudo-base weights or tuning grids differ. Each comparison uses its own matched protocol.''',r'''Table~\ref{tab:recognition} and the original source/Sketch studies use Full-query ($\eta=1$): retained fusion searches $5\!\times\!6\!\times\!33=990$ coefficient/offset settings. Table~\ref{tab:prospective} uses Cardinality-aware Full, applying $\eta_B=\min(1,B/30)$ at both pseudo and actual base stages, with 3,636 settings per readout. The independent-retuning study uses the latter budget with Full-query. Comparisons are paired within a cohort; equal full-bank weights do not equate models selected under different pseudo-base weights or grids.''')
change('Fixed vocabulary-growth tests yield zero base-score drift and ranking churn, and identical final predictions across arrival orders (Supplements A, I and J).','HyCal retains nonzero tuned novel recognition in every condition (Table~\\ref{tab:recognition}); its certified zeros denote rejection under the shared accuracy floor, not zero recognition capacity.')
change('The diagnostic uses the combined decision margin. Supplement P reports the separate offset distributions and all six conditions.','The diagnostic uses the combined decision margin on fixed readouts (Supplement P).')
change('all six conditions meet every final target (Supplement P).','all six conditions meet every final target.')
change('Supplement Y varies the actual base-class count and refits every classifier, comparing five prespecified interpolation constants with $C=30$ fixed as the main rule.','Refitted base-cardinality experiments compare five prespecified constants with $C=30$ fixed.')
a=s.index('\\paragraph{Query correspondence and new-domain recognition}');b=s.index('\n\\subsection{Useful admission',a)
s=s[:a]+r'''\paragraph{Query correspondence and new-domain recognition}
Table~\ref{tab:prospective}(a) separates the value of retaining a head from that of alignment. On fine-grained Food-101, Dense already reaches 94.57\% L/14 oracle capacity and Full differs by $-0.04$ points; query alignment still adds up to 3.23 certified novel points over None (B/32), while SUN397 ($B=195$) yields 7.79/10.14-point capacity gains over Dense at matched base accuracy. On ImageNet-R, all three shuffled-moment controls reduce capacity: the statistics' marginal distribution alone does not reproduce query alignment. Fixing prototype centering costs 1.70/1.67 points and fixing learned dispersion costs 0.03/1.41 (L/14/B/32). Within SUN novel queries sharing the same base winner, gains remain 7.74/9.74 points, locating them in group competition.

\paragraph{Closed-form and learned routing}
The base vocabulary sets the available pseudo-expansion tasks. With Pets/CIFAR's 7/20 pseudo-base classes, Full-query's L/14 deployed gains over Logistic/MLP range from 1.66 to 21.75 points (Supplement V; the 1.66-point interval includes zero). On SUN397's larger vocabulary, Logistic/MLP improve Full's oracle capacity by 0.88--1.52 points; the logistic router adds 13 parameters. Thus anchored evidence supports both closed-form and learned routing, with empirical benefits depending on the available task. These cross-dataset contrasts do not isolate vocabulary size as the cause.
''' +s[b:]
change("HyCal's matched transfer receives no certified novel admission in this condition.","HyCal's 58.33\\% tuned novel accuracy becomes zero certified admission here. Two issues are distinct: its base ceiling meets the shared floor in only 2/10 jointly feasible partitions (0/10 on CIFAR and B/32 ImageNet-R), before any statistical test; its candidate-dependent normalization can also change old rankings. The latter permits recovery relative to a frozen reference, but $G>0$ alone does not identify the cause of certification failure.")
# Remove scattered pointers while keeping necessary cohort identifiers and theoretical proof pointers.
s=s.replace(' (Supplement S).','.').replace(' (Supplement T).','.').replace(' (Supplement R).','.').replace(' (Supplement Z).','.')
s=s.replace(' All partitions are reported in Supplement P.','').replace(' All ten partitions, absolute accuracies and admission counts are in Supplement R.','')
p.write_text(s,encoding='utf-8')
# Food rows directly from verified summary, preserving paired CIs.
p=H/'table_cvpr_prospective.tex';s=p.read_text();s=s.replace('(a) SUN397: Full', '(a) New domains: Full').replace('Encoder &', 'Data / Encoder &',1)
s=s.replace('L/14 & +3.06','SUN / L/14 & +3.06').replace('B/32 & +3.76','SUN / B/32 & +3.76')
summary=json.loads((H/'prospective/summary.json').read_text());lines=[]
for en,name in [('l14','L/14'),('b32','B/32')]:
 r=next(r for r in summary['contrasts'] if r['block']=='external' and r['dataset']=='food' and r['encoder']==en and r['baseline']=='dense')
 def fmt(v): return f"{v['mean']:+.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"
 lines.append('Food-101 / '+name+' & '+' & '.join(fmt(r[k]) for k in ['capacity95','matched_actual_base','policy'])+r' \\')
s=s.replace('SUN / L/14 &','\n'.join(lines)+'\nSUN / L/14 &',1);p.write_text(s)
# Insert matched frozen, tuning-selected final novel readouts in the dual-view table.
p=H/'table_cvpr_dualview.tex';s=p.read_text().replace('llrrrr','llrrrrrr').replace('multicolumn{2}','multicolumn{3}').replace('{3-4}','{3-5}').replace('{5-6}','{6-8}')
s=s.replace('Average incr. & Direct novel (finite)','AIA & Tuned & Certified (n)')
lookup=json.loads((H/'tuned_no_fallback.json').read_text())['rows'];names={'Retained readout':'retained','Dense prototypes':'anchored_tp','Fixed temperature':'fixed_temperature','HyCal transfer':'hycal'};ds=None;result=[]
for line in s.splitlines():
 if any(n+' &' in line for n in names):
  c=line.removesuffix(r' \\').split(' & ')
  if c[0]:ds={'CIFAR-100':'cifar','Pets':'pets','ImageNet-R':'imagenetr'}[c[0]]
  m=names[c[1]]
  vals=[f"{next(r for r in lookup if r['dataset']==ds and r['encoder']==en and r['method']==m)['novel_mean']:.2f}" for en in ['l14','b32']]
  line=' & '.join(c[:3]+[vals[0],c[3],c[4],vals[1],c[5]])+r' \\'
 result.append(line)
p.write_text('\n'.join(result)+'\n')
# Verified primary-source references.
p=H/'references.bib';s=p.read_text();s+=r'''
@article{auckenthaler2000score,
 author={Auckenthaler, Roland and Carey, Michael and Lloyd-Thomas, Harvey},
 title={Score Normalization for Text-Independent Speaker Verification Systems},
 journal={Digital Signal Processing}, volume={10}, pages={42--54}, year={2000},
 doi={10.1006/dspr.1999.0360}
}
@article{mariethoz2005unified,
 author={Mari{\'e}thoz, Johnny and Bengio, Samy},
 title={A Unified Framework for Score Normalization Techniques Applied to Text Independent Speaker Verification},
 journal={IEEE Signal Processing Letters}, volume={12}, number={7}, year={2005}
}
@inproceedings{murugesan2024robust,
 author={Murugesan, Balamurali and Silva-Rodr{\'i}guez, Julio and Ben Ayed, Ismail and Dolz, Jose},
 title={Robust Calibration of Large Vision-Language Adapters},
 booktitle={European Conference on Computer Vision}, year={2024},
 doi={10.1007/978-3-031-72691-0_9}
}
''';p.write_text(s,encoding='utf-8')
# Add explicit supplementary note without inventing a self-reference risk evaluation.
p=H/'supplementary.tex';s=p.read_text(encoding='utf-8');needle='The resulting comparison tests matched access in our long-tailed, retention-constrained task.'
s=s.replace(needle,needle+' Main Table~\\ref{main-tab:recognition} also reports final novel accuracy at each frozen base-only tuning-selected offset, without fallback or a retention constraint. These values are reconstructed from the same stored prediction margins and class-ID tie rule as the certified policies; the accompanying \\nolinkurl{tuned_no_fallback.json} records partition values. They separate recognition from certification against the common Ridge reference. Candidate-dependent ranking changes and reference-floor infeasibility are distinct properties; this comparison does not estimate a causal share of rejection attributable to recovery.')
p.write_text(s,encoding='utf-8')
print('requested changes written')
