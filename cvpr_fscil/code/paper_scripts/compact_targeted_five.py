from pathlib import Path
p=Path(__file__).resolve().parent/'main.tex';s=p.read_text(encoding='utf-8')
def para(start,end,new):
 global s
 a=s.index(start);b=s.index(end,a);s=s[:a]+new+'\n\n'+s[b:]
para('These properties connect algebraically.', 'Our contributions are:',r'''Vocabulary invariance fixes base ordering; the ridge identity $\mu_L(x)=1/|\mathcal B|$ exposes the dispersion-dependent gate. Relative to the same frozen readout, $G=0$ removes the recovery contribution to calibration variance at matched accuracies. On Pets L/14, the gate exceeds trained logistic/MLP routers by 7.32/2.85 capacity points and 21.75/15.25 direct-admission points (Supplement V). After independent fusion tuning, actual-base-matched ImageNet-R capacity gains over dense prototypes are 3.35 [1.92,4.98]/17.86 [11.88,24.60] points (L/14/B/32; ten/six feasible partitions).''')
para(r'\paragraph{Independent fusion optimization}',r'\paragraph{Transfer to a different base classifier}',r'''\paragraph{Independent fusion optimization}
Each readout selects fusion and offset from 3,636 base-only settings. On ImageNet-R, query alignment adds 3.60 [2.55,4.74]/5.29 [3.74,7.25] capacity points over no centering and 2.84 [2.42,3.28]/4.65 [3.57,5.71] over fixed moments (L/14/B/32), averaging both orders. First-order direct policies give CIFAR L/14 71.14\% versus 59.61\%/59.75\% (None/Fixed) and 34.60\% (dense), a 36.54-point dense contrast. ImageNet-R gives 81.66\% versus 76.42\% (None) and 76.35\% (dense) with L/14; B/32 gives 52.44\% versus 45.01\%/46.25\% (None/Fixed) and 19.97\% (dense). Supplement Q reports all retuned comparisons.''')
para(r'\paragraph{Transfer to a different base classifier}',r'\paragraph{Cardinality-aware alignment on Pets}',r'''\paragraph{Transfer to a different base classifier}
With independently tuned class-weighted softmax readouts, query centering adds 3.46 [2.17,4.70]/6.09 [4.30,8.00] ImageNet-R capacity points over no centering (L/14/B/32; Table~\ref{tab:softmax}). Average incremental gains are 1.30/1.58 points (Supplement S).''')
para(r'\paragraph{Transfer across backbones and pretraining families}',r'\paragraph{Additional matched-base and gate controls}',r'''\paragraph{Transfer across backbones and pretraining families}
On ImageNet-R, convolutional CLIP RN50 gives 45.17\% capacity and 41.40\% direct novel accuracy, with 10/10 admissions and targets met. Capacity gains over None/Fixed are 4.23 [2.80,5.56]/3.25 [1.82,4.75] points. SigLIP-B/16~\cite{zhai2023siglip} gives 85.68\% direct novel accuracy with 10/10 admissions and a 1.34 [0.82,1.89]-point capacity gain over None (Supplement T).''')
start=s.index(r'\begin{figure*}[t]',s.index(r'\paragraph{Additional matched-base'))
end=s.index(r'\end{figure*}',start)+len(r'\end{figure*}')
fig=s[start:end].replace(r'\begin{figure*}[t]',r'\begin{figure*}[!t]')
s=s[:start]+s[end:]
pos=s.index(r'\paragraph{Transfer across backbones')
s=s[:pos]+fig+'\n\n'+s[pos:]
p.write_text(s,encoding='utf-8')
