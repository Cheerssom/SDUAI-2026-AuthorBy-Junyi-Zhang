"""Apply the five author-requested narrative edits to the frozen main_11 source."""
from pathlib import Path
import re

H = Path(__file__).resolve().parent
p = H / 'main.tex'
s = p.read_text(encoding='utf-8')

def paragraph(start, replacement):
    global s
    a = s.index(start)
    b = s.index('\n\n', a)
    s = s[:a] + replacement + s[b:]

paragraph('Expanding a trained classifier requires', r'''Expanding a trained classifier requires admitting few-shot classes while retaining base accuracy. We propose reference-anchored asymmetric class expansion, coupling vocabulary-invariant evidence alignment with a query-adaptive admission margin and a preserved base-error structure. Fixed references prevent vocabulary-induced score drift; a ridge mean identity exposes the dispersion-dependent group gate; preserving base rankings eliminates error recovery and its contribution to paired calibration variance at matched accuracies. Independent calibration turns this margin into a retention-constrained policy. After independent fusion optimization, query alignment adds 3.60 and 5.29 points of ImageNet-R novel capacity at 95\% retention with CLIP L/14 and B/32, while calibrated policy gains over dense prototypes reach 36.54 points on CIFAR-100 L/14. Cardinality-aware interpolation supports small reference banks, and RN50 validates convolutional-backbone transfer. With zero target-domain calibration labels, frozen source models attain 82.99\% and 54.87\% novel accuracy on ImageNet-Sketch, admitting expansion and meeting empirical retention targets in all ten partitions per encoder. Target calibration further raises accuracy to 86.82\% and 58.96\%.''')

paragraph('We propose \\emph{reference-anchored', r'''We propose \emph{reference-anchored asymmetric class expansion}, aligning the base--novel decision margin for certifiable expansion. The design couples three properties: a fixed score frame preserves stored discrimination under vocabulary growth; query-dependent reference moments align heterogeneous evidence; and the preserved base ranking constrains the error structure seen by calibration. A cardinality-aware interpolation extends the same alignment operator to small reference banks.''')
paragraph('The mechanism connects stored', r'''These properties connect algebraically. Vocabulary-growth invariance fixes the base ordering (Proposition~\ref{prop:invariance}). The ridge identity $\mu_L(x)=1/|\mathcal B|$ then exposes a dispersion-dependent admission gate without a routing network. Relative to the same frozen base readout, ranking preservation makes recovery $G=0$: new base errors go to novel labels, so protection and retention coincide. At matched correctness rates, this structure removes the $2rG$ contribution to paired-score variance (Supplement P). Independent calibration selects the global offset on this aligned margin. Experiments separate its recognition frontier from the selected policy and test source-only transfer as well as target recalibration.''')
a=s.index('Our contributions are:')
b=s.index('\\end{itemize}',a)+len('\\end{itemize}')
s=s[:a]+r'''Our contributions are:
\begin{itemize}
\item Vocabulary-invariant asymmetric alignment that preserves fitted base discrimination while admitting semantic and few-shot novel evidence, with moment interpolation for small reference banks.
\item An algebraic link from the ridge mean invariant to query-adaptive group gating and from preserved rankings to a zero-recovery error structure for retention calibration.
\item Recognition and certified-admission gains after independent fusion optimization, across ridge/softmax readouts and ViT/CNN encoders, plus natural-shift transfer without target calibration labels.
\end{itemize}'''+s[b:]

s=s.replace(r'\section{Reference-anchored asymmetric class expansion}'+'\n', r'''\section{Reference-anchored asymmetric class expansion}
The readout shapes the base--novel margin that calibration will threshold. Fixed references make its score frame invariant to vocabulary growth; query alignment sets its sample-dependent balance; preserving the base ordering gives calibration a one-way error structure. We derive these coupled properties below.
''',1)

paragraph('We also optimize the fusion coefficients', r'''Each readout independently selects fusion and offset from 3,636 base-only tuning settings. On ImageNet-R, query alignment increases 95\%-retention capacity over no centering by 3.60 [2.55,4.74] points with L/14 and 5.29 [3.74,7.25] with B/32; gains over fixed moments are 2.84 [2.42,3.28] and 4.65 [3.57,5.71]. These frontiers average both orders. Independently calibrated first-order policies also retain substantial gains: CIFAR L/14 reaches 71.14\% versus 59.61\%/59.75\% for None/Fixed and 34.60\% for dense prototypes. ImageNet-R gives 81.66\% versus 76.42\% (None) and 76.35\% (dense) with L/14, and 52.44\% versus 45.01\%/46.25\% (None/Fixed) and 19.97\% (dense) with B/32. Thus gains over None are 5.24--11.53 points and over dense prototypes 5.31--36.54 points in these conditions, after independent optimization (Supplement Q).''')

paragraph('Replacing ridge regression with', r'''Replacing ridge regression with a class-weighted softmax classifier preserves the alignment effect on ImageNet-R (Table~\ref{tab:softmax}). With fusion independently selected for each scheme, query centering increases common-target novel capacity over no centering by 3.46 [2.17,4.70] points with L/14 and 6.09 [4.30,8.00] with B/32. Average incremental gains are 1.30 and 1.58 points. Supplement S gives the fitting details and complete policies.''')

paragraph('The stage-consistent rule uses', r'''Supplement U tests small-bank alignment under the 3,636-setting quantile search, separately from the compact-grid models in Tables~\ref{tab:recognition}--\ref{tab:centeringMain}; Pets has 15 base and seven pseudo-base classes. Against the retuned Query capacity of 87.59\%, stage-wise $\eta_B=\min(1,B/30)$ raises L/14 capacity to 92.49\%, a gain of 4.89 [1.52,8.91] points; freezing the catalog weight $\eta=.5$ throughout tuning and deployment gives 83.81\% direct novel accuracy with 10/10 admissions, versus retuned Query's 68.67\% (8/10).''')
a=s.index('\\begin{table}[t]',s.index('\\paragraph{Cardinality-aware alignment'))
b=s.index('\\end{table}',a)+len('\\end{table}')
s=s[:a]+r'''\begin{table}[t]
\centering
\caption{Softmax base readout, ImageNet-R (\%). Capacity averages both orders; direct novel uses order 0. All ten partitions admit expansion and meet the test target for every row. Reference targets and calibration images match the original ridge study. Query/None/Fixed schemes select fusion independently.}
\label{tab:softmax}
{\small\setlength{\tabcolsep}{5pt}\input{table_softmax_main}}
\end{table}'''+s[b:]

paragraph('On CIFAR L/14, the retained readout reaches', r'''Figure~\ref{fig:sourceOperating} shows why the operating region matters. On ImageNet-R L/14, matching both empirical frontiers at the dense policy's stricter realized base accuracy yields a 3.10 [1.37,4.88]-point recognition advantage. At 99\% retention, the gap is 6.06 [1.50,12.35] points on five jointly feasible partitions (Supplement P). The 95\% minimum lies in the flatter region, where the gap is 0.29 [$-0.55$,1.16]. At this common minimum, retained/dense policy headrooms are 2.32/6.88 points: the 0.29-point capacity gap plus the 4.56-point headroom difference equals the 4.85-point policy gain. Preserved rankings give $G=0$, removing the $2rG$ variance term when correctness rates match; Supplement P derives the class-stratified identity. On CIFAR L/14, the selected-policy gain reaches 35.40 points (72.00\% versus 36.60\%), including finite admission in 10/10 versus 6/10 partitions.''')

paragraph('For the original retained readout,', r'''For the original retained readout, $G=0$ makes protection coincide with retention. Calibration uses held-out base images, with no extra novel labels beyond the five-shot support. In the nested-cap protection study (Supplement L), five images per base class suffice for final-session admission on CIFAR L/14 and ImageNet-R with both encoders; CIFAR B/32 reaches full admission at ten. Every one of the ten partitions admits expansion for each of three subset seeds (30/30 cells), with all empirical test targets met. At cap 10, ImageNet-R L/14 reaches 80.09\% novel accuracy versus 80.74\% at cap 40. Cap 40 admits all six dataset/encoder conditions in 10/10 partitions; L/14 novel accuracies are 72.76\%, 88.51\% and 80.74\% on CIFAR, Pets and ImageNet-R. Supplements L and N give both certificates and complete arrival paths.''')

s=s.replace(r'\subsection{Frozen-model transfer through target calibration}',r'\subsection{Frozen-model transfer without target calibration labels}')
paragraph('The transfer study evaluates', r'''The original ImageNet-R readouts transfer to matching ImageNet-Sketch classes~\cite{wang2019robustglobal} with the encoder, classifier, fusion, five-shot source prototypes and candidate offsets frozen. With \textbf{zero target-domain calibration labels}, source-calibrated retained readouts achieve 82.99\% and 54.87\% novel accuracy with L/14 and B/32. All ten partitions admit expansion and meet empirical retention targets throughout all five arrivals. Dense B/32 reaches 11.19\% and HyCal B/32 0.00\% final novel accuracy (Supplement R).''')
paragraph('With direct target calibration,', r'''Optional target recalibration uses independent labeled base queries and matches source calibration's per-class counts; only the scalar offset changes. Novel accuracy rises to 86.82\% at 90.76\% base accuracy with L/14 and 58.96\% at 78.87\% with B/32, with all ten paths admitted and targets met (Fig.~\ref{fig:sketchMain}). Dense L/14 reaches 83.34\% with 10/10 final admissions; dense B/32 reaches 8.35\% including fallback, with 3/10 admissions. Identity-disjoint pools and complete policies are in Supplement R.''')
paragraph('Source calibration also meets', r'''Source-only transfer establishes useful recognition on this natural shift; target calibration provides additional admission capacity. The CIFAR noise/blur study separately tests calibration--query correspondence: matched-shift calibration gives 10/10 finite admissions and test targets for both encoders, whereas clean calibration meets 5/10 and 1/10 L/14 targets (Supplement O). Population confidence concerns the class-conditional distribution represented by the calibration sample.''')
paragraph('Reference-anchored class expansion aligns', r'''Reference anchoring couples vocabulary-invariant asymmetric evidence, a query-adaptive margin and preserved base-error structure. The resulting readout improves retention-constrained recognition and certified admission after independent tuning, transfers across fitted heads and encoder families, and supports frozen-model natural-shift recognition with source-only calibration.''')
p.write_text(s,encoding='utf-8')

v=H/'validate_cvpr.py'
v.write_text(v.read_text().replace("MAIN_STEM='main_11'", "MAIN_STEM='main_12'"),encoding='utf-8')
b=H/'build_delivery.py'
b.write_text(b.read_text().replace('cvpr2027-20261027-cardinality-rn50','cvpr2027-20261027-requested-narrative'),encoding='utf-8')
print('Applied the five requested edits; experimental files unchanged.')
