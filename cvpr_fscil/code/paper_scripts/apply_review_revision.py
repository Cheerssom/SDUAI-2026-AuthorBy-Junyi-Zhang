"""Apply the current presentation revision; no experiment is rerun."""
from pathlib import Path
import re,shutil,json,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
B=H.parent/'cvpr2027_revisions/20261027_before_review'
B.mkdir(parents=True,exist_ok=True)
for n in ['main.tex','main.pdf','supplementary.tex','supplementary.pdf','validate_cvpr.py','README.md','build_delivery.py','validation.json','freeze_manifest.json']:
    if not (B/n).exists():shutil.copy2(H/n,B/n)
def terms(s):
    for old,new in [('Reference-anchored tables','Reference-anchored readouts'),('Reference-anchored table','Reference-anchored readout'),
                    ('reference-anchored tables','reference-anchored readouts'),('reference-anchored table','reference-anchored readout'),
                    ('Candidate-normalized table','Candidate-normalized readout'),('candidate-normalized table','candidate-normalized readout'),
                    ('retained tables','retained readouts'),('retained table','retained readout'),('proposed table','proposed readout'),
                    ('heterogeneous table','asymmetric readout'),('candidate-table expansion','asymmetric-readout expansion')]:s=s.replace(old,new)
    return s
for p in H.glob('*.tex'):
    s=terms(p.read_text(encoding='utf-8')).replace('kbs_','expansion_').replace('reviewer_revision','controlled_ablation')
    p.write_text(s,encoding='utf-8')
for p in list(H.glob('*kbs_*')):
    if p.suffix=='.tex':p.rename(p.with_name(p.name.replace('kbs_','expansion_')))
p=H/'supplementary.tex';s=p.read_text(encoding='utf-8')
s=s.replace(r'\renewcommand{\thefigure}{S\arabic{figure}}',r'\renewcommand{\thefigure}{S\arabic{figure}}'+'\n'+r'\renewcommand{\theequation}{S\arabic{equation}}')
p.write_text(s,encoding='utf-8')
p=H/'paired_accuracy_proof.tex';s=p.read_text(encoding='utf-8')
s=s.replace('% Proof source for the reviewer-revision direct calibration procedure.','% Direct paired accuracy calibration proof.')
s=s.replace(r'\subsection{Direct paired accuracy calibration}',r'\subsection{Direct paired accuracy calibration}'+'\n'+r'We specialize the bounded-mean betting construction of Waudby-Smith and Ramdas~\cite{waudby2024betting} to paired correctness with class-macro weights.')
p.write_text(s,encoding='utf-8')
p=H/'main.tex';s=p.read_text(encoding='utf-8')
old='The system selects the novel group when $M(x)+d>0$ and the base group when it is negative, with the fixed class-ID rule resolving ties. Query scales make branch responses comparable; query centers adjust their relative group evidence. The scalar $d$ then shifts this adaptive decision to the desired retention level. The mechanism uses the existing readout responses and requires no auxiliary router or task label. Its empirical signature is improved ranking of base versus novel queries together with greater correctly classified novel admission.'
new=r'''The system selects the novel group when $M(x)+d>0$ and the base group when it is negative, with the fixed class-ID rule resolving ties. This is an \textbf{implicit query-adaptive group gate}: each query is assessed through its class-specific learned response and its novel-prototype evidence relative to the same base references. A high standardized base response favors the base group; a high standardized novel response favors admission. Query scales set the relative influence of these branches, while query centers change their group balance. The global $d$ then selects the retention level.

The learned contribution depends on the full ratio $(q_L-1/B)/\sigma_L$, linking the group decision to both the winning response and its reference dispersion. For example, two-class responses $(.9,.1)$ and $(.51,.49)$ both standardize to $(1,-1)$ despite their different dispersions. The offset term in Eq.~\eqref{eq:ridgeoffset} therefore acts together with the scaled class scores. This algebra identifies a gate obtained directly from evidence alignment, without an auxiliary routing network or task oracle. The improved base--novel ranking and controlled centering results below test its recognition effect.'''
assert old in s;s=s.replace(old,new)
old='The controlled source tables above and the full paired results retain dataset-specific comparisons, including the strong prototype performance on Pets. Auxiliary FeNeC and analytic-readout comparisons are in Sections A--H.'
new=r'''Pets illustrates the distinction between recognition and retention-constrained admission. Dense prototypes have the highest unconstrained average incremental accuracy in Table~\ref{tab:recognition}. Under the common 95\% retention test, the retained readout instead attains 87.52\% versus 71.64\% novel accuracy with L/14 and 54.58\% versus 13.25\% with B/32 (Table~\ref{tab:admission}): gains of \textbf{15.88} and \textbf{41.33} points. These final-session policy comparisons include fallback. Auxiliary FeNeC and analytic-readout comparisons are in Supplements A--H.'''
assert old in s;s=s.replace(old,new)
s=s.replace('Capacity and selected-policy performance are linked through the operating trajectories in Supplement P.',r'Figure~\ref{fig:sourceOperating} links recognition trajectories to independently selected policies on the original frozen source models.')
fig=r'''\begin{figure*}[!t]
\centering\includegraphics[width=\textwidth]{source_operating_l14.pdf}
\caption{Source-domain capacity and calibrated admission, L/14, prespecified partition 0, fifth arrival. Lines are empirical recognition trajectories; circles use the direct test and triangles protection. Hollow markers indicate reference fallback, and dotted lines the common minimum base target. Calibration selects the points independently of the test-label trajectories. All partitions and both encoders are reported in Supplement P.}
\label{fig:sourceOperating}
\end{figure*}

'''
s=s.replace(r'\subsection{Useful admission under a base-accuracy requirement}',fig+r'\subsection{Useful admission under a base-accuracy requirement}')
s=s.replace('The protection bound in Supplement L gives the necessary effective count 89.78 for the protection bound.','The protection bound requires effective count 89.78 under this five-arrival setting (Supplement L).')
s=s.replace(r'\section*{LLM Usage Statement}',r'\clearpage'+'\n'+r'\section*{LLM Usage Statement}')
p.write_text(s,encoding='utf-8')
print('Text, equations, terminology and protocol references revised.')
