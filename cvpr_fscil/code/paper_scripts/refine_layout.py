from pathlib import Path
import re,json
H=Path(__file__).resolve().parent;K=H.parent/'kbs'
p=H/'main.tex';s=p.read_text(encoding='utf-8')
# A conference related-work section organized by problems, not venue.
a=s.index(r'\section{Related work}');b=s.index(r'\section{Reference-anchored',a)
s=s[:a]+r'''\section{Related work}
\paragraph{Recognition with unequal supervision}
Long-tailed learning addresses unequal class supervision through balanced fitting and losses~\cite{liu2019large,cao2019ldam,cui2019classbalanced}. CLIP~\cite{radford2021clip} provides image--text evidence; prompt adaptation~\cite{zhou2022coop,shi2024lift}, prototypes~\cite{snell2017prototypical,wang2019simpleshot} and semantic caches~\cite{zhang2022tipadapter} adapt pretrained features to labeled data. Incremental learning studies retention across arriving classes~\cite{rebuffi2017icarl,tao2020fewshot}. SimpleCIL~\cite{zhou2024simplecil} and RanPAC~\cite{mcdonnell2023ranpac} use prototype and analytic readouts; FeNeC~\cite{ksiazek2026fenec} uses clustered representatives. Our readout preserves fitted base discrimination while incorporating the different evidence available for novel classes.
\paragraph{Calibration and evidence alignment}
BiMC~\cite{chen2025bimc} and HyCal~\cite{lee2026hycal} calibrate multimodal or heterogeneous recognition metrics. We use fixed base references to separate query adaptation from vocabulary growth. Calibrated seen/unseen recognition~\cite{chao2016calibrated} supplies a related operating trade-off. Independently selecting the admission offset specializes risk-controlled selection~\cite{angelopoulos2021ltt} and bounded-mean betting~\cite{waudby2024betting} to class-macro base accuracy. The matched HyCal transfer tests recognition under the same encoder, support and data access; broader comparisons are in the supplement.

'''+s[b:]
# State the deployment control briefly; full statistical derivations remain J/N.
a=s.index(r'\subsection{Selecting an offset');b=s.index(r'\section{Experiments}',a)
s=s[:a]+r'''\subsection{Independent admission calibration}
\label{sec:riskcontrol}
Calibration selects the global offset after the readout is fixed. Let $R$ and $C_d$ indicate correctness of the original closed-base reference and expanded readout. We test $\mathbb E(C_d-rR)\geq0$ on independent base queries, giving each of $B$ classes equal weight. For $n_c$ queries in class $c$, set $w_i=1/(Bn_{c(i)})$ and $b=\max_i w_i$.

Adapting bounded-mean betting~\cite{waudby2024betting}, define
\begin{equation}
 E_\lambda=\prod_i[1+\lambda(w_i/b)(C_{d,i}-rR_i)],\quad 0\leq\lambda<1/r.
 \label{eq:directbet}
\end{equation}
Under independent correctness pairs and the null $\sum_iw_i\mathbb E(C_{d,i}-rR_i)\leq0$, this nonnegative product has expectation at most one. A fixed convex mixture $E$ yields the fixed-sample p-value $\min(1,1/E)$. This direct test credits recovery of reference errors and uses the same requirement for every readout; the proof and mixture are in Supplement N.

Threshold candidates are fixed on tuning data and tested from conservative to permissive. Following fixed-sequence testing~\cite{angelopoulos2021ltt}, we stop at the first failure and return the last accepted offset. If none passes, prediction reverts to the original reference with zero novel admission. Testing each arrival at $\delta/T$ bounds unsafe acceptance across $T=5$ prespecified arrivals by $\delta$. The guarantee is per fixed configuration under independent target class-conditional sampling, with calibration excluded from fitting and tuning.

We also evaluate the sufficient protection loss
\begin{equation}
 L_d=R(1-C_d)+(1-r)(1-R).
 \label{eq:protectionloss}
\end{equation}
Its requirement $\mathbb E L_d\leq1-r$ coincides with retention for the original retained readout because $G=0$. The weighted concentration bound and its sample-coverage condition are in Supplements J and L; all main cross-readout admission comparisons use the direct test.

'''+s[b:]
# Frontiers and selected points remain in Supplement P, with complete cohorts.
s=re.sub(r'\\begin\{figure\*\}\[t\]\s*\\centering\\includegraphics\[width=\\textwidth\]\{mechanism_operating_frontiers_p0.pdf\}.*?\\end\{figure\*\}\s*','',s,flags=re.S)
s=s.replace(r'linked through the operating trajectories in Fig.~\ref{fig:operatingMain}',r'linked through the operating trajectories in Supplement P')
s=s.replace(r'Equation~\eqref{eq:coveragefloor} gives',r'The protection bound in Supplement L gives')
# Compact tables retain every dataset/encoder and original value.
s=s.replace(r'\input{table_asymmetric_recognition}',r'\input{table_cvpr_recognition}')
s=s.replace(r'\input{table_asymmetric_admission}',r'\input{table_cvpr_admission}')
s=s.replace(r'\input{table_asymmetric_centering}',r'\input{table_cvpr_centering}')
s=s.replace('Base/Novel are final-session test percentages, including zero novel accuracy on reference fallback. Finite counts admission out of ten partitions; all empirical retention targets are met.',
            'Novel values are final-session test percentages, including zero on reference fallback. Finite gives Query/None admission counts out of ten. All empirical retention targets are met; absolute base accuracies are in Supplement P.')
p.write_text(s,encoding='utf-8')

for name in ['recognition','admission']:
    old=(K/f'table_asymmetric_{name}.tex').read_text()
    rows=[line for line in old.splitlines() if line.startswith(('Retained readout','Dense prototypes','Fixed temperature','HyCal transfer'))]
    assert len(rows)==8
    lines=[r'\begin{tabular}{@{}lrrrrrr@{}}',r'\toprule',
           r'& \multicolumn{3}{c}{CLIP ViT-L/14} & \multicolumn{3}{c}{CLIP ViT-B/32} \\',
           r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
           r'Readout & CIFAR-100 & Pets & ImageNet-R & CIFAR-100 & Pets & ImageNet-R \\',r'\midrule']
    for i in range(4):
        left=rows[i].removesuffix(r' \\');right=rows[i+4].removesuffix(r' \\').split(' & ',1)[1]
        lines.append(left+' & '+right+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (H/f'table_cvpr_{name}.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
rows=[];enc='L/14'
for line in (K/'table_asymmetric_centering.tex').read_text().splitlines():
    if 'ViT-B/32' in line:enc='B/32'
    if ' & Query &' in line:
        q=line.removesuffix(r' \\').split(' & ')
    elif ' & None &' in line:
        n=line.removesuffix(r' \\').split(' & ')
        rows.append(f'{enc} & {q[0]} & {q[4]} & {n[4]} & {q[5]} & {q[2][0:-3]}/{n[2][0:-3]}'+r' \\')
assert len(rows)==6
lines=[r'\begin{tabular}{@{}llrrrl@{}}',r'\toprule',r'Encoder & Dataset & Query novel (\%) & None novel (\%) & Paired gain (pp) [95\% CI] & Finite (Query/None) \\',r'\midrule']+rows+[r'\bottomrule',r'\end{tabular}']
(H/'table_cvpr_centering.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Conference narrative and compact tables prepared.')
