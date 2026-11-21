"""Present completed fixed-model diagnostics with their statistical scope."""
from pathlib import Path
import json,shutil
H=Path(__file__).resolve().parent
R=H.parents[1]/'experiments/expansion_followup_diagnostics'
S=json.loads((R/'summary.json').read_text());A=json.loads((R/'statistical_audit.json').read_text())
assert A['complete'] and json.loads((R/'complete.json').read_text())['complete']
snap=H.parent/'cvpr2027_revisions/20261028_before_followup_integration';snap.mkdir(parents=True,exist_ok=True)
for n in ['main.tex','supplementary.tex','validate_cvpr.py','build_delivery.py','README.md']:shutil.copy2(H/n,snap/n)
dest=H/'followup';dest.mkdir(exist_ok=True)
for n in ['summary.json','statistical_audit.json','coverage.pdf']:shutil.copy2(R/n,dest/n)
for enc in ['l14','b32']:shutil.copy2(R/f'router_inference/{enc}.json',dest/f'router_{enc}.json')
def ci(v):return f"{v['mean']:.2f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]"
def val(v):return '--' if v is None else f"{v['mean']:.2f}"
def table(headers,rows,caption):
 return r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{longtable}{'+'l'*len(headers)+'}\n'+r'\caption{'+caption+r'}\\\toprule'+'\n'+' & '.join(headers)+r'\\\midrule\endfirsthead'+'\n'+r'\toprule '+' & '.join(headers)+r'\\\midrule\endhead'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+'\n'+r'\bottomrule\end{longtable}\endgroup'+'\n'
t=r'''\FloatBarrier
\section{Calibration coverage, query groups and routing cost}\label{app:followup}
\subsection{Calibration coverage at small base cardinalities}
We fix the C30 models, fusion coefficients and tuning-derived candidate sequences from Section Y. ImageNet-R uses $B\in\{5,10,15,30\}$, both encoders, ten class partitions and both orders after the fifth arrival. Ten prespecified nested class-wise subsets use caps 5, 10, 20 and 40. Caps are upper limits: the existing independent pool supplies fewer observations for some classes. In the $B=5$ tasks, cap 10 gives 50 total queries, cap 20 gives 95--100 and cap 40 gives 128--200. No fitting, tuning or test images supplement these pools. The direct test retains $r=.95$ and the five-arrival confidence allocation.

For a fixed model, changing calibration coverage leaves its recognition frontier unchanged. At $B=5$, L/14 capacity remains 90.34\%; finite admission is 0\% at caps 10 and 20 and 10\% at cap 40. The corresponding B/32 counts are 0\%, 0\% and 20\%. A direct check compares the test against $C=R$, the unchanged closed-base prediction. The frozen Full model has the same closed-base predictions as the reference, so any novel admission satisfies $C_d\leq R$. At caps 10 and 20, even this maximal-retention correctness pattern cannot pass in any subset. Thus these budgets impose a statistical-power constraint independently of the model's empirical novel capacity.

At $B=10$, increasing cap 10 to 40 raises finite admission by 90 [70,100] percentage points with L/14 and 50 [20,80] with B/32. At $B=5$, the corresponding gains are 10 [0,30] and 20 [0,50]; these intervals include zero. Coverage supports useful admission, while the remaining capacity-to-policy gap also depends on candidate margins and reference accuracy.

Statistics average subset repetitions and arrival orders within each partition before 20,000 paired bootstrap draws. Repeated subsets are not independent resampling units. Conditional novel means average admitted policies within each partition and then the partitions with admission. Reference fallback contributes zero to unconditional novel accuracy and can satisfy the empirical base target. Complete intervals and effective partition counts are in \nolinkurl{followup/summary.json} and \nolinkurl{followup/statistical_audit.json}.
\begin{figure}[htbp]\centering\includegraphics[width=.95\linewidth]{followup/coverage.pdf}\caption{Fixed-model calibration coverage. Admission and novel accuracy vary with the per-class cap; test-label recognition capacity is unchanged within each fixed $B$ task.}\label{fig:followupcoverage}\end{figure}
'''
t+=table(['Encoder','$B$','Cap','Capacity','Finite','Novel','Cond.','Met'],[[r['encoder'].upper(),r['B'],r['cap']]+[val(r[k]) for k in ['capacity','finite','novel','conditional_novel','met']] for r in S['coverage'] if r['cap'] in [5,10,20,40]],r'Fixed-model coverage means (\%). Finite and Met are admission and empirical target rates; Cond. is admitted-only novel accuracy. Capacity uses ten partitions in every row. Conditional effective counts and paired intervals accompany the summary.')
t+=table(['Encoder','$B$','Cap change',r'$\Delta$ Finite (pp)',r'$\Delta$ Novel (pp)'],[[r['encoder'].upper(),r['B'],f"{r['low_cap']}--{r['high_cap']}",ci(r['finite']),ci(r['novel'])] for r in A['budget_effects']],r'Paired budget effects with 95\% intervals, using ten partition means per contrast.')
t+=r'''\paragraph{Equal total calibration size}
We also fix the total to 150: per-class counts are 30, 15, 10 and 5 for $B=5,10,15,30$. All four tasks have sufficient independent images in the same four partitions (0, 5, 6, 8); the comparison uses this common subset throughout. These are different fitted tasks, so the comparison controls calibration quantity without identifying a causal effect of $B$. The corresponding intervals are wide and do not establish a universal ordering across cardinalities.
'''
t+=table(['Encoder','$B$','Per class','Finite (\%)','Novel (\%)'],[[r['encoder'].upper(),r['B'],r['cap'],ci(r['finite']),ci(r['novel'])] for r in S['equal_total_joint_partitions']],r'Equal total of 150 calibration queries on the four jointly available partitions. Paired partition bootstrap intervals condition on this common set.')
t+=r'''\subsection{Where new-domain recognition gains arise}
Food-101 and SUN397 use the frozen models in Section W. For each final partition/order, both empirical frontiers are evaluated at Dense's direct policy's realized base accuracy. Queries are divided by whether the two readouts predict the same within-base class. These groups and the correction/destruction statistics are specified before examining this diagnostic; the analysis is descriptive and uses test labels for recognition evaluation, not policy selection.

On SUN397 novel queries with identical base winners, Full gains 7.74 [5.37,10.29] points with L/14 and 9.74 [7.18,12.40] with B/32. This group comprises approximately 74\% of novel queries and contributes 5.66/7.33 points to the total gain. The result locates recognition value in base--novel competition even when the within-base argmax agrees. Food's corresponding contrasts are 0.16 [$-0.29$,0.68] and 0.66 [$-2.06$,3.57] points. The grouping does not isolate the causal contribution of one branch statistic.

Accuracy within a group uses the original class-macro weights renormalized to its mass. Correction counts Full-correct/Dense-wrong outcomes; destruction counts the reverse. Their difference equals the accuracy contrast. Paired intervals resample partition means after averaging orders. Query and class counts below are ranges across the twenty partition/order cells, not numbers of independent observations. In Food, the disagreement group is empty in two partitions per encoder: its conditional statistics use the eight nonempty partitions. Group mass and its contribution to the overall gain still use all ten, assigning an empty group's contribution zero.
'''
names={'all':'All','base_winners_agree':'Agree','base_winners_disagree':'Disagree'}
t+=table(['Encoder','Data','Query','Group','Images','Classes','$n$',r'$\Delta$ Novel/Base (pp)'],[[r['encoder'].upper(),r['dataset'].title(),r['group'].title(),names[r['stratum']],f"{r['query_count_range'][0]}--{r['query_count_range'][1]}",f"{r['class_count_range'][0]}--{r['class_count_range'][1]}",r['gain']['n'],ci(r['gain'])] for r in A['groups']],r'Group sizes and Full minus Dense recognition accuracy with paired 95\% intervals. $n$ counts effective bootstrap partitions, excluding empty groups from conditional contrasts.')
t+=r'''\subsection{Closed-form and learned routing costs}
Timing uses SUN397 partition 0, both encoders, the same CPU with two compute threads, and batches of 512 frozen image features. Five warm-ups precede thirty measurements. Complete readout timing includes fixed-reference scoring and, for trained routers, the twelve-feature construction and gate forward pass. Image encoding, loading and label-dependent evaluation are excluded. A separate forward-only timing assumes the twelve router inputs are already available. P10--P90 describes empirical repetition variability rather than a confidence interval or a paired significance test.

Full introduces no additional router parameters or router fit. Logistic and the 16-unit MLP add 13 and 225 trainable parameters; their standardizers store 24 constants. Timing-only fitting uses the existing base-only pseudo-expansion data. Selected-setting fits have three repetitions; complete 36-model selection has one replay and is therefore reported without a repetition interval. Timing refits do not replace any accuracy model.
'''
t+=table(['Encoder','Router','Complete ms','P10--P90 ms'],[[r['encoder'].upper(),r['method'].title(),f"{r['median_ms']:.3f}",f"{r['p10_ms']:.3f}--{r['p90_ms']:.3f}"] for r in A['timing']],r'Cached-feature inference for 512 queries. All rows use thirty timing repetitions on the same hardware.')
rows=[]
for enc in ['l14','b32']:
 r=json.loads((dest/f'router_{enc}.json').read_text())
 for row in r['rows']:
  fit=row['selected_setting_fit_seconds'];fw=row['gate_forward']
  rows.append([enc.upper(),row['kind'].title(),row['trainable_router_parameters'],f"{fw['median_ms']:.3f}",f"{fw['p10_ms']:.3f}--{fw['p90_ms']:.3f}",f"{min(fit):.3f}--{max(fit):.3f}",f"{row['full_selection_seconds']:.3f}"])
t+=table(['Encoder','Router','Params','Gate ms','P10--P90','Fit s','Search s'],rows,r'Additional router costs. Fit gives the range across three selected-setting refits; Search times one full 36-model selection. Full has zero additional router fit. The routers are lightweight; these measurements characterize the trade-off rather than imply a large training burden.')
(H/'supplement_followup.tex').write_text(t,encoding='utf-8')
f=H/'main.tex';t=f.read_text();old='The frozen $C=30$ rule is used throughout prospective Food/SUN evaluation.'
t=t.replace(old,r'''Within SUN queries whose base winners agree, Full still adds 7.74 [5.37,10.29]/9.74 [7.18,12.40] novel points, locating gains in group competition (Supplement AA).''')
a=t.index('At fixed class-wise correctness rates, removing recovery');b=t.index(r'\subsection{Frozen-model',a)
t=t[:a]+r'''At fixed correctness rates, removing recovery raises CIFAR L/14 simulated acceptance from 2.8\% to 90.6\% (Supplement V). Its cap-10 study admits and meets targets in all 300 partition/subset cells on CIFAR and ImageNet-R. A fixed-model small-bank study separates capacity from calibration coverage: at $B=10$, increasing cap 10 to 40 raises admission by 90 [70,100]/50 [20,80] points (L/14/B/32). At $B=5$, cap 10/20 cannot certify even the unchanged reference despite 90.34\% L/14 capacity. Supplement AA reports coverage curves, group sizes and lightweight-router costs.

'''+t[b:];f.write_text(t)
f=H/'supplementary.tex';t=f.read_text().replace(r'\input{supplement_prospective}',r'\input{supplement_prospective}'+'\n\n'+r'\input{supplement_followup}');t=t.replace('Sections A--H provide auxiliary','Section AA analyzes small-bank calibration coverage, new-domain query groups and router costs. Sections A--H provide auxiliary');f.write_text(t)
for n in ['validate_cvpr.py','README.md']:
 f=H/n;f.write_text(f.read_text().replace('main_19','main_20'))
f=H/'build_delivery.py';t=f.read_text().replace("names.update(['prospective/summary.json','prospective_validation.json'])","names.update(['prospective/summary.json','prospective_validation.json'])\nnames.update(['followup/summary.json','followup/statistical_audit.json','followup/router_l14.json','followup/router_b32.json'])");t=t.replace('20261028-supplement-prose','20261028-followup-evidence');f.write_text(t)
shutil.copy2(R.parent/'expansion_followup_diagnostics_protocol.md',H/'experiments/expansion_followup_diagnostics_protocol.md')
print('Follow-up evidence integrated.')
