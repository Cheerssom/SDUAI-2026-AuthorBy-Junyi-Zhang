"""Present the matched recognition/admission views and new head control."""
from pathlib import Path
import re,json,shutil
H=Path(__file__).resolve().parent;ROOT=H.parents[1];K=H.parent/'kbs'
B=H.parent/'cvpr2027_revisions/20261027_before_softmax';B.mkdir(parents=True,exist_ok=True)
for n in ['main.tex','main_8.pdf','supplementary.tex','supplementary.pdf','validate_cvpr.py','README.md','build_delivery.py']:
    if not (B/n).exists():shutil.copy2(H/n,B/n)
def rows(p):return [l.removesuffix(r' \\').split(' & ') for l in p.read_text().splitlines() if l.startswith(('Retained readout','Dense prototypes','Fixed temperature','HyCal transfer'))]
a=rows(H/'table_cvpr_recognition.tex');b=rows(H/'table_cvpr_admission.tex')
lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',
       r'& & \multicolumn{2}{c}{CLIP ViT-L/14} & \multicolumn{2}{c}{CLIP ViT-B/32} \\',
       r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',
       r'Dataset & Readout & Average incr. & Direct novel (finite) & Average incr. & Direct novel (finite) \\',r'\midrule']
for j,d in enumerate(['CIFAR-100','Pets','ImageNet-R']):
    for i in range(4):
        lines.append((d if i==0 else '')+' & '+a[i][0]+' & '+' & '.join([a[i][j+1],b[i][j+1],a[i][j+4],b[i][j+4]])+r' \\')
    lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
(H/'table_cvpr_dualview.tex').write_text('\n'.join(lines)+'\n')
result=json.loads((H/'softmax_readout/summary.json').read_text())
lines=[r'\begin{tabular}{@{}llrr@{}}',r'\toprule',r'Encoder & Alignment & Capacity & Direct novel \\',r'\midrule']
for row in result['conditions']:
    lines.append(f"{row['encoder'].replace('l14','L/14').replace('b32','B/32')} & {row['mode'].capitalize()} & {row['capacity']['mean']:.2f} & {row['novel']['mean']:.2f}"+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular}'])
(H/'table_softmax_main.tex').write_text('\n'.join(lines)+'\n')
p=H/'main.tex';s=p.read_text()
blocks=list(re.finditer(r'\\begin\{table\*\}\[!t\].*?\\end\{table\*\}\s*',s,re.S))
m=next(m for m in blocks if 'table_cvpr_admission' in m[0]);s=s[:m.start()]+s[m.end():]
m=next(m for m in re.finditer(r'\\begin\{table\*\}\[!t\].*?\\end\{table\*\}',s,re.S) if 'table_cvpr_recognition' in m[0])
table=r'''\begin{table*}[!t]
\centering
\caption{Recognition and retention-constrained admission under matched access (\%). Average incr. is fixed-offset all-class macro accuracy averaged over five arrivals and two orders, mean $\pm$ sample standard deviation across ten partitions. Direct novel is fifth-arrival novel accuracy under the common 95\% retention test, first order; parentheses count finite admission out of ten, with zero novel accuracy on fallback. All methods share encoders, fitting/tuning data, five-shot support and cap-40 calibration. These are different evaluation views, not a before/after accuracy change. All empirical direct-test targets are met except one dense B/32 ImageNet-R case. Bold marks the largest mean within each view. Supplements J, M and N give absolute base accuracy and complete policy results.}
\label{tab:recognition}\label{tab:admission}
{\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}\input{table_cvpr_dualview}}
\end{table*}'''
s=s[:m.start()]+table+s[m.end():]
anchor=r'\paragraph{Recognition at shared retention requirements}'
block=r'''\paragraph{Transfer to a different base classifier}
Replacing ridge regression with a class-weighted softmax classifier preserves the alignment effect on ImageNet-R. With fusion independently selected for every scheme, query centering improves common-target novel capacity over no centering by 3.46 [2.17,4.70] points with L/14 and 6.09 [4.30,8.00] with B/32. Average incremental gains are 1.30 and 1.58 points. Table~\ref{tab:softmax} separates these frontiers from direct-calibrated policies; Supplement S reports all configurations and fitting details.

\begin{table}[t]
\centering
\caption{Softmax base readout, ImageNet-R (\%). Capacity averages both orders; direct novel uses order 0. All ten partitions admit expansion and meet the test target for every row. Reference targets and calibration images match the original ridge study. Query/None/Fixed schemes select fusion independently.}
\label{tab:softmax}
{\small\setlength{\tabcolsep}{5pt}\input{table_softmax_main}}
\end{table}

'''
assert anchor in s;s=s.replace(anchor,block+anchor)
s=s.replace('Table~\\ref{tab:recognition} evaluates joint base/novel recognition across five arrivals.','Table~\\ref{tab:recognition} juxtaposes incremental recognition and final-session admission under the same base requirement.')
s=s.replace('The table reports finite admission so that fallback remains visible.','Finite admission counts accompany the policy accuracies.')
p.write_text(s)
p=H/'supplementary.tex';s=p.read_text()
pos=s.index(r'\FloatBarrier'+'\n'+r'\begingroup',s.index(r'\label{app:sketchvalidation}'))
addition=r'''\FloatBarrier
\section{Transfer to a cross-entropy base readout}
\label{app:softmaxreadout}
\subsection{Locked protocol and data access}
This study replaces the ridge-fitted base head with an affine multiclass softmax classifier on the same frozen ImageNet-R features. Both CLIP encoders, ten partitions, two orders, five arrivals, fitting/tuning/calibration/test pools and five-shot supports remain fixed. The protocol is \nolinkurl{experiments/expansion_softmax_readout_protocol.md}. No new encoder training or data extraction is involved.

The head minimizes class-uniform inverse-frequency-weighted cross-entropy plus $\lambda\|W\|_F^2/2$ on slopes, with an unregularized intercept. Zero initialization and 300 full-batch Adam steps use learning rate .05, betas .9/.999 and epsilon $10^{-8}$. Base tuning macro accuracy selects $\lambda\in\{10^{-5},10^{-4},10^{-3},10^{-2}\}$ separately for full-base and pseudo-base fits. Every 50 steps, an atomic checkpoint saves weights and optimizer state. Query centering/query scaling, no centering/query scaling and fixed tuning moments each independently select their two fusion coefficients and offset using the same 3,636-setting base-only pseudo-expansion grid as Section Q. Selection is frozen before calibration/test evaluation.

All schemes use the original ridge readout's reference correctness, absolute retention target and fallback, copied from the frozen Section Q archives. Thus cross-head recovery is allowed. The ridge mean identity does not apply to the softmax head; vocabulary invariance still holds within each fixed selected readout. Both certificates use the same cap-40 calibration images and five-arrival allocation. Main policies use the direct paired accuracy test.

\subsection{Recognition frontiers and selected policies}
\begin{table}[htbp]
\centering
\caption{Cross-entropy readout control on ImageNet-R. All accuracies are percentages. AIA averages five arrivals and both orders within each partition. Capacity averages both orders at the original common 95\% target; every partition is feasible. Direct novel is final-session order-0 accuracy, including fallback. Finite/Met counts admission and empirical target attainment out of ten. Full means, conditional means and paired contrasts are in the accompanying summary.}
{\small\setlength{\tabcolsep}{5pt}\input{table_softmax_readout}}
\end{table}

Query centering increases capacity over uncentered scaling by 3.46 [2.17,4.70] points with L/14 and 6.09 [4.30,8.00] with B/32. AIA gains are 1.30 [0.86,1.77] and 1.58 [1.22,1.98] points. Relative to fixed moments, capacity gains are 2.03 [1.41,2.59] and 1.18 [$-0.19$,2.47] points. Direct-policy gains over no centering are 1.92 [$-6.16$,7.60] and 8.97 [4.38,15.27] points. Fixed moments select final novel accuracies 79.04\%/48.11\%, while query alignment selects 78.03\%/46.35\%. Every configuration admits expansion and meets the empirical final target in all ten partitions. These results support transfer of the recognition/frontier effect across fitting losses; they do not impose a universal ordering on finite-sample selected policies.

Intervals are paired conditional 95\% bootstrap intervals over ten partition means, averaging orders first. The same fixed images and encoders underlie all contrasts. Common-floor frontier minus selected-policy novel accuracy averages 6.62/10.27 points for query alignment, 5.08/13.16 for no centering and 3.57/7.33 for fixed moments (L/14/B/32), making the capacity-to-policy difference explicit.

The 4,800 metric records, 200 prediction archives and 5,672 threshold tests are reconstructed by an independent direct-product implementation of the paired p-value. A restart preserves hashes and timestamps of 604 checkpoint files; a separate sampled witness recomputes score margins and predictions from saved weights and original features. Optimizer continuation also matches a seeded uninterrupted fit exactly. These are deterministic reconstruction checks, not a fresh independent semantic review. No test outcome selects the fitting or fusion protocol.

'''
s=s[:pos]+addition+s[pos:];s=s.replace('Sections A--H provide auxiliary readout','Section S tests a cross-entropy base readout. Sections A--H provide auxiliary readout')
p.write_text(s)
# Bundle the newly referenced protocol under its neutral name.
(H/'experiments/expansion_softmax_readout_protocol.md').write_bytes((ROOT/'experiments/expansion_softmax_readout_protocol.md').read_bytes())
print('Integrated dual-view primary table and the new base-head control.')
