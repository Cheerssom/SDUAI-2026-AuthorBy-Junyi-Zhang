"""Integrate verified prospective evidence; never rerun model selection."""
from pathlib import Path
import json, shutil
H=Path(__file__).resolve().parent
R=H.parents[1]/'experiments/expansion_prospective_validation'
s=json.loads((R/'report/summary.json').read_text())
assert json.loads((R/'complete_all.json').read_text())
out=H/'prospective';out.mkdir(exist_ok=True)
for name in ['summary.json']:
    shutil.copy2(R/'report'/name,out/name)
for name in ['mechanism_capacity95.pdf','cardinality_cifar.pdf','cardinality_imagenetr.pdf','shift_noise.pdf','shift_blur.pdf']:
    shutil.copy2(R/'report'/name,out/name)
def replace_between(t,a,b,new):
    i=t.index(a);j=t.index(b,i);return t[:i]+new+'\n\n'+t[j:]
t=(H/'main.tex').read_text(encoding='utf-8')
t=replace_between(t,'Expanding a trained classifier',r'\end{abstract}',r'''Expanding a trained classifier requires admitting few-shot classes while retaining base accuracy. We propose reference-anchored asymmetric expansion: fixed base references preserve existing score differences, while query-dependent moments align learned, semantic and prototype evidence. A ridge mean identity exposes a closed-form group gate without learned router parameters; preserved base rankings yield a zero-recovery error structure for independent retention calibration. Cardinality-aware interpolation extends alignment to small reference banks. With the interpolation rule frozen before evaluation on new domains, the method exceeds dense prototypes on SUN397 by 7.79 and 10.14 percentage points in novel capacity at matched actual base accuracy with CLIP L/14 and B/32. Independently calibrated novel accuracy improves by 8.71 and 10.64 points, with both methods admitting all 20 final-session paths. On ImageNet-R, three prespecified query-moment permutations reduce capacity by 4.38--4.96 and 7.52--8.10 points, establishing the contribution of query-specific correspondence. Frozen source models also attain 82.99\%/54.87\% novel accuracy on ImageNet-Sketch without target calibration labels. These results connect invariant evidence alignment to useful recognition under explicit base-retention requirements.''')
t=replace_between(t,'Vocabulary invariance fixes base ordering;', 'Our contributions are:',r'''Vocabulary invariance fixes base ordering; the ridge identity $\mu_L(x)=1/|\mathcal B|$ exposes the dispersion-dependent gate. Relative to the same frozen readout, $G=0$ removes the recovery contribution to calibration variance at matched accuracies. Prospective evaluation fixes the interpolation rule before testing Food-101 and SUN397. On SUN397, actual-base-matched gains over dense prototypes reach 7.79 [5.57,10.22]/10.14 [7.88,12.57] points (L/14/B/32), with all ten partitions feasible. Query-moment permutations further isolate the information supplied by per-image alignment.''')
t=t.replace('Recognition and certified-admission gains after independent fusion optimization, across ridge/softmax readouts and ViT/CNN encoders, plus natural-shift transfer without target calibration labels.', 'Recognition and certified-admission gains on new semantic domains with a frozen design rule, supported by query-moment interventions, refitted base-cardinality studies and calibration-shift evaluation.')
t=t.replace('On Pets L/14, its capacity gains over trained logistic/MLP gates are 7.32 [4.01,10.28]/2.85 [0.80,4.78] points (Supplement V), supporting structural gating without fitting a pseudo-novel router.', 'Sections V and W compare this gate with base-only logistic and MLP routers; Section X tests which query statistics carry its recognition benefit.')
t=replace_between(t,r'\paragraph{Additional matched-base and gate controls}',r'\subsection{Useful admission',r'''\paragraph{Query correspondence and branch statistics}
Every moment intervention independently retunes fusion. On ImageNet-R, Full exceeds three prespecified shuffled-moment controls by 4.38--4.96 capacity points with L/14 and 7.52--8.10 with B/32; every paired interval excludes zero. Shuffling preserves the joint moment distribution but removes its correspondence with the query. Fixing prototype centering costs 1.70/1.67 points; fixing learned dispersion costs 0.03/1.41. Thus query correspondence and prototype centering contribute across encoders, with an additional learned-scale contribution on B/32 (Supplement X).

\paragraph{Prospective validation on new semantic domains}
We freeze $C=30$ before Food-101 and SUN397 evaluation and independently tune six readouts with equal selection budgets. On SUN397, Full exceeds dense prototypes by 7.79 [5.57,10.22]/10.14 [7.88,12.57] points at matched actual base accuracy and by 8.71/10.64 in direct novel accuracy. All methods admit and meet the target on all 20 final paths, separating these gains from fallback. Food-101 B/32 yields 1.56 [0.43,2.72] capacity points over None, with Full and dense prototypes comparable. Learned routers built on Full add 0.88--1.52 capacity points on SUN397, showing that reference alignment also supports an additional learned gate. Supplement W reports every readout, accuracy and paired interval.''')
t=t.replace('Supplement O tests noise and blur.', 'A progressive mismatch study locates the calibration boundary: at noise $\sigma=.12$, L/14 source-calibrated retention averages 93.04\%; independent target calibration raises it to 98.68\%, with novel accuracy changing from 21.90\% to 15.39\% (Supplement Z).')
t=t.replace('It improves retention-constrained recognition after independent tuning, transfers across heads and encoder families, and supports natural-shift recognition with source-only calibration.', 'Matched-base gains on the new SUN397 domain and query-moment interventions establish its recognition value beyond calibration fallback. The frozen design transfers across heads and encoder families, with independent calibration providing a controlled admission policy.')
(H/'main.tex').write_text(t,encoding='utf-8')
names={'full':'Full','none':'None','fixed':'Fixed','dense':'Dense','logistic':'Logistic','mlp':'MLP','pm_only':'Prototype mean','scale_only':'Scales only','minus_learned_scale':'Fixed learned scale','minus_text_scale':'Fixed text scale','retained':'Retained','anchored_tp':'Dense','reference':'Reference'}
def label(x):return names.get(x,x.replace('_',' '))
def val(x):return '--' if x['mean'] is None else f"{x['mean']:.2f}"
def ci(x):
    return '--' if x['mean'] is None else f"{x['mean']:.2f} [{x['ci95'][0]:.2f},{x['ci95'][1]:.2f}]"
def table(headers, rows, caption):
    n=len(headers)
    return '\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{'+'l'*n+'}\n\\caption{'+caption+'}\\\\\n\\toprule\n'+' & '.join(headers)+r' \\ \midrule\endfirsthead'+'\n'+r'\toprule '+' & '.join(headers)+r' \\ \midrule\endhead'+'\n'+''.join(' & '.join(map(str,r))+r' \\'+'\n' for r in rows)+r'\bottomrule\end{longtable}\endgroup'+'\n'
common=r'''\FloatBarrier
\section{Prospective evaluation on new semantic domains}\label{app:prospective}
The rule $\eta_B=\min(1,B/30)$ is frozen before new-domain outcomes. Food-101 uses its official train/test memberships, with 50 base and 50 novel classes and one excluded class per partition. SUN397 uses 195 base and 195 novel classes, excluding seven. Its task-specific class-wise splits reserve 20\% for testing (at least 20 images), 40 calibration, ten tuning and five support images per class before long-tailed fitting. Food reserves 40 calibration, 20 tuning and five support images from training; fitting uses at most 200 images per class. Exact byte duplicates are removed before role assignment. This is class-expansion evaluation rather than the standard SUN397 classification protocol.

Both frozen CLIP encoders use eight text templates, ten class assignments, two arrival orders, five arrivals and five-shot support. Each fusion readout independently selects among 36 coefficient pairs and 101 pseudo-expansion offsets (3,636 settings). Logistic and 16-unit MLP gates use the fixed Full group winners and 12 query features; 36 regularized models and 101 validation offsets give the same selection budget. They add a router to Full's representation. All fitting and selection use base-only data. The common reference is Full's closed-base predictor; cap-40 calibration uses the direct paired test with $\delta=.05$ allocated over five arrivals.

Tables report final-session means after averaging the two orders within each partition; AIA averages all arrivals. Finite/Met counts the 20 final partition/order paths. Capacities are test-label diagnostics. Matched-base contrasts use each comparator's realized direct-policy base accuracy and the tightest conservatively feasible Full frontier point without interpolation. Paired 95\% intervals use 20,000 partition bootstrap draws conditional on these images. The included \texttt{prospective/summary.json} provides all statistics, conditional policies, AUROCs and matching overshoot.

On SUN397, Full's matched-base capacity gains over Dense are 7.79 [5.57,10.22] and 10.14 [7.88,12.57] points, with direct-policy gains 8.71 and 10.64 points. All methods admit every final path, so these comparisons contain no fallback contribution. Food supports transfer of query alignment, with Full and Dense at comparable recognition levels. On SUN397, learned gates further improve Full's capacity by 0.88--1.52 points: the fixed reference representation supports both closed-form and trained routing.
'''
tex=common
for ds in ['food','sun']:
    rows=[r for r in s['rows'] if r['block']=='external' and r['dataset']==ds]
    tex+=table(['Encoder','Readout','AIA','Capacity','Base','Novel','Retention','Finite/Met'],[[r['encoder'].upper(),label(r['method'])]+[val(r[k]) for k in ['aia','capacity95','base','novel','retention']]+[f"{r['finite']}/{r['met']}"] for r in rows],ds.title()+r': absolute accuracies (\%). Finite/Met are counts out of 20; all capacity means use ten feasible partitions.')
    rows=[r for r in s['contrasts'] if r['block']=='external' and r['dataset']==ds]
    tex+=table(['Encoder','Baseline',r'$\Delta N_{.95}$',r'$\Delta N_{\rm matched}$',r'$\Delta N_{\rm policy}$'],[[r['encoder'].upper(),label(r['baseline'])]+[ci(r[k]) for k in ['capacity95','matched_actual_base','policy']] for r in rows],ds.title()+': Full minus comparator (pp), paired 95\% intervals; ten paired partitions in every cell. Policy gains also equal jointly admitted gains.')
tex+=r'''\FloatBarrier
\section{Query-moment correspondence and branch interventions}\label{app:momentintervention}
Each of eight variants independently selects fusion using the same 3,636-setting budget on CIFAR-100, Pets and ImageNet-R. Full uses stage-specific $C=30$. Prototype-mean-only retains query-dependent prototype centering with fixed scales; scale-only retains query-dependent scales with fixed prototype centering. Two further controls fix learned or text scale alone. The learned mean remains $1/B$ throughout.

Three prespecified seeds (3101--3103) jointly permute all branch moments between queries, separately within tuning, calibration and test roles. The mapping uses row identities rather than labels and remains fixed across fusion trials and arrivals. This preserves moment marginals and cross-branch dependence while breaking correspondence with the image. Shuffled batch predictors are interventions; their empirical selected policies do not receive the ordinary independent-query deployment guarantee.

Full's ImageNet-R capacity gains against all three shuffles have positive paired intervals: 4.38--4.96 points on L/14 and 7.52--8.10 on B/32. Prototype centering contributes on both encoders; fixing learned scale yields a supported capacity difference on B/32. The interventions identify complementary statistics rather than requiring every branch scale to be individually decisive.
'''
for ds in ['cifar','pets','imagenetr']:
    rows=[r for r in s['contrasts'] if r['block']=='mechanism' and r['dataset']==ds]
    tex+=table(['Encoder','Control',r'$\Delta N_{.95}$',r'$\Delta N_{\rm matched}$','Paired $n$'],[[r['encoder'].upper(),label(r['baseline'])]+[ci(r[k]) for k in ['capacity95','matched_actual_base']]+[r['matched_actual_base']['n']] for r in rows],ds+': Full minus intervention (pp). The paired count refers to matched-base feasibility; all common-floor contrasts use ten partitions.')
tex+=r'''\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{prospective/mechanism_capacity95.pdf}\caption{Independently tuned moment interventions: common-floor novel capacity.}\end{figure}
\FloatBarrier
\section{Refitted tasks across base-class cardinalities}\label{app:truecardinality}
We change the classification task itself: CIFAR-100 uses $B\in\{5,10,15,20,30,50\}$ with 20 novel classes; ImageNet-R uses $B\in\{5,10,15,20,30,50,100\}$ with 50 novel classes. Ten nested stratified base selections are disjoint from the fixed novel set. For every $B$, both the full-base and pseudo-base ridge heads are refitted and their regularization reselected. This preserves the ridge identity for the actual fitted catalog.

The prespecified constants $C\in\{15,20,30,40,60\}$ each receive independent fusion tuning; $C=30$ remains the main rule. All methods share the C30 closed-base reference at that task size. The curves characterize sensitivity without selecting a new constant from test outcomes. At small $B$, useful empirical capacity can coexist with sparse calibration coverage and reference fallback. Larger full and pseudo-base banks make the interpolation rules coincide when both exceed $C$.
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{prospective/cardinality_cifar.pdf}\caption{CIFAR-100: full classifier refitting across base cardinalities and five fixed interpolation constants.}\end{figure}
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{prospective/cardinality_imagenetr.pdf}\caption{ImageNet-R: capacity and selected-policy sensitivity to actual base-task size.}\end{figure}
'''
for ds in ['cifar','imagenetr']:
    rows=[r for r in s['rows'] if r['block']=='cardinality' and r['dataset']==ds]
    tex+=table(['Encoder','$B$','$C$','Capacity','Novel','Retention','Finite/Met'],[[r['encoder'].upper(),r['B'],r['method'][1:]]+[val(r[k]) for k in ['capacity95','novel','retention']]+[f"{r['finite']}/{r['met']}"] for r in rows],ds+r': complete cardinality sweep (\%). Finite/Met counts out of 20. Novel accuracy includes fallback; conditional values and paired intervals are in the accompanying summary.')
tex+=r'''\FloatBarrier
\section{Progressive calibration--deployment mismatch}\label{app:progressiveshift}
We freeze the original CIFAR retained and dense models, clean-tuning candidate sequences and fifth-arrival first-order tasks. Pixel noise has $\sigma\in\{.04,.08,.12,.20\}$ and blur radius is in $\{.75,1.25,2,3\}$, with clean controls. The same 20 test images per class and cap-40 independent calibration identities are used at every severity; noise uses common random numbers. Source calibration uses clean images, target calibration uses separately corrupted calibration images, and reference-only prediction supplies the fallback control.

At each severity the target is 95\% of the original reference's closed-base accuracy on that shifted distribution. The absolute reference accuracy is reported alongside retention, avoiding confusion between preserved relative accuracy and preserved source performance. At L/14 noise .12, source-calibrated mean retention is 93.04\%; target recalibration gives 98.68\%, with novel accuracy 21.90\% and 15.39\%, respectively. Target calibration restores the distribution correspondence required by the guarantee; source certificates retain their source-law scope.
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{prospective/shift_noise.pdf}\caption{Progressive noise: source calibration, independent target calibration and reference-only prediction.}\end{figure}
\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{prospective/shift_blur.pdf}\caption{Progressive blur under the same frozen models and candidate offsets.}\end{figure}
'''
for enc in ['l14','b32']:
    rows=[r for r in s['shift'] if r['encoder']==enc]
    tex+=table(['Shift','Readout','Cal.','Ref.','Base','Novel','Ret.','Viol.','Finite/Met'],[[r['kind'].replace('_',' '),label(r['method']),r['calibration']]+[val(r[k]) for k in ['reference','old','new','retention','violation_pp']]+[f"{round(r['finite']['mean']*10)}/{round(r['met']['mean']*10)}"] for r in rows],enc.upper()+r': absolute accuracies and retention (\%), mean positive target violation (pp), and final admission/target counts out of ten. Full intervals are supplied in the summary.')
tex+=r'''\paragraph{Reconstruction checks}
The four studies retain frozen selections and checkpointed predictions. Independent reconstruction covers 3,600 prediction archives, 20,200 method/session records, 101,818 threshold tests and 138,000 frontier checks, plus 720 shifted policy records. A separate NumPy score witness includes learned-gate inputs and predictions; its maximum margin discrepancy is $8.06\times10^{-5}$. These checks validate numerical reconstruction of the reported experiments.
'''
(H/'supplement_prospective.tex').write_text(tex,encoding='utf-8')
t=(H/'supplementary.tex').read_text(encoding='utf-8').replace('times,amsmath','longtable,times,amsmath')
t=t.replace(r'\input{supplement_gate_calibration}',r'\input{supplement_gate_calibration}'+'\n\n'+r'\input{supplement_prospective}')
t=t.replace('Sections A--H provide auxiliary', 'Sections W--Z report prospective new-domain validation, query-moment interventions, refitted base-cardinality tasks and progressive calibration shift. Sections A--H provide auxiliary')
(H/'supplementary.tex').write_text(t,encoding='utf-8')
p=H/'validate_cvpr.py';t=p.read_text().replace("MAIN_STEM='main_15'","MAIN_STEM='main_16'");p.write_text(t)
p=H/'build_delivery.py';t=p.read_text().replace("names.update(['gate_calibration/summary.json'", "names.add('prospective/summary.json')\nnames.update(['gate_calibration/summary.json'").replace('cvpr2027-20261028-targeted-five','cvpr2027-20261028-prospective');p.write_text(t)
p=H/'README.md';t=p.read_text().replace('main_15','main_16');t+='\nProspective revision integrates Food-101/SUN397, query-moment interventions, true base-cardinality refits and progressive calibration shift in Supplements W--Z. Full statistics are in prospective/summary.json.\n';p.write_text(t)
print('Integrated prospective evidence.')
