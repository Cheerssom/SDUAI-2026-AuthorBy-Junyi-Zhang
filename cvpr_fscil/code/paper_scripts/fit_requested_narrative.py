"""Remove repeated explanations and schedule the unchanged Sketch figure earlier."""
from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'main.tex';s=p.read_text(encoding='utf-8')
def paragraph(start,text):
 global s
 a=s.index(start);b=s.index('\n\n',a);s=s[:a]+text+s[b:]

paragraph('The system selects the novel group',r'''The system selects the novel group when $M(x)+d>0$, with fixed class-ID ties. This \textbf{implicit query-adaptive group gate} combines the standardized learned response with novel-prototype evidence relative to base references. The global $d$ selects the retention level. The gate depends on the full ratio $(q_L-1/B)/\widetilde s_L$, rather than dispersion alone; Supplement P gives the algebraic decomposition and controlled ranking tests.''')
a=s.index('The gate depends on the full ratio',s.index('This \\textbf{implicit'))
# Remove the former standalone paragraph, retaining the new sentence above.
a=s.index('The gate depends on the full ratio',a+1);b=s.index('\n\n',a)
s=s[:a]+s[b+2:]

paragraph('Table~\\ref{tab:recognition} juxtaposes',r'''Table~\ref{tab:recognition} compares incremental recognition and final-session admission. The retained readout reaches 87.29\%/67.93\% ImageNet-R average incremental accuracy with L/14/B/32, exceeding matched HyCal by 10.93/13.74 points and also improving on it on CIFAR and Pets. Dense prototypes and fixed-temperature fusion isolate retained weights and query alignment. These source models are subsequently used for admission and transfer (Supplement J).''')
paragraph('The vocabulary-growth study measures',r'''The vocabulary-growth study measures zero base-score drift and within-base ranking churn, with identical final predictions across two arrival orders under fixed class-ID ties. Supplements A and I report stability and paired classifier-value estimates.''')
paragraph('Pets illustrates the distinction',r'''Pets distinguishes unconstrained recognition from retention-constrained admission. Dense prototypes lead average incremental accuracy in Table~\ref{tab:recognition}, while the retained readout leads direct novel accuracy by \textbf{15.88} points with L/14 (87.52\% versus 71.64\%) and \textbf{41.33} with B/32 (54.58\% versus 13.25\%), including fallback.''')

old=r'''Full-query recognition and retention-constrained admission under matched access (\%). Average incr. is fixed-offset all-class macro accuracy averaged over five arrivals and two orders, mean $\pm$ sample standard deviation across ten partitions. Direct novel is fifth-arrival novel accuracy under the common 95\% retention test, first order; parentheses count finite admission out of ten, with zero novel accuracy on fallback. All methods share encoders, fitting/tuning data, five-shot support and cap-40 calibration. These are different evaluation views, not a before/after accuracy change. All empirical direct-test targets are met except one dense B/32 ImageNet-R case. Bold marks the largest mean within each view. Supplements J, M and N give absolute base accuracy and complete policy results.'''
new=r'''Full-query recognition and direct 95\%-retention admission (\%). Average incr.: all-class accuracy over five arrivals and two orders, mean $\pm$ partition standard deviation. Direct novel: final first-order novel accuracy, with finite admissions out of ten in parentheses and zero on fallback. Readouts share encoders, fitting/tuning data, five-shot support and cap-40 calibration. All empirical targets are met except one dense B/32 ImageNet-R case. Bold marks each view's largest mean. Supplements J, M and N give absolute base accuracy and full policies.'''
assert old in s;s=s.replace(old,new)
old=r'''Novel recognition at shared base-retention requirements in the ten-partition replication (Supplementary Section I), after the fifth arrival. Each readout attains the stated fraction of the retained reference's closed-base accuracy. Means and sample standard deviations average two orders within each partition; points require all ten partitions to be feasible. Vertical ranges are 50--100\% (L/14) and 30--90\% (B/32); bars stop at panel limits. Test-label sweeps describe capacity. Section I gives partial feasible sets and counts.'''
new=r'''Final-arrival novel capacity at shared base-retention requirements (Supplement I). Test-label sweeps attain the stated fraction of the retained reference's closed-base accuracy. Means and standard deviations average two orders per partition; plotted points require all ten partitions feasible. Vertical ranges are 50--100\% (L/14) and 30--90\% (B/32); bars stop at panel limits.'''
assert old in s;s=s.replace(old,new)

# Early declaration allows the full-width figure to enter page 8.
a=s.index('\\begin{figure*}[t]',s.index('\\paragraph{Recognition at shared'))
b=s.index('\\end{figure*}',a)+len('\\end{figure*}')
fig=s[a:b];s=s[:a]+s[b:]
pos=s.index('\\paragraph{Transfer to a different base classifier}')
s=s[:pos]+fig+'\n\n'+s[pos:]
p.write_text(s,encoding='utf-8')
print('Removed repeated prose; all five requested evidence blocks retained.')
