"""Reframe main evidence around one Full model and stated references."""
from pathlib import Path
import json,shutil
H=Path(__file__).resolve().parent;R=H.parents[1]
old=(H/'main.tex').read_text();method=old[old.index('\\section{Reference-anchored'):old.index('\\begin{table*}')]
method=method.replace('Three requirements guide the score operator; independent calibration selects $d$.','Fixed base references define the score operator; independent calibration selects $d$.')
a=method.index('\\subsection{Structural requirements');b=method.index('\\subsection{Constructing',a)
method=method[:a]+r'''\subsection{Vocabulary-growth invariance}
For active classes $\mathcal C$, define $\Delta_{\mathcal C}(x;c,c')=s_{\mathcal C}(x,c)-s_{\mathcal C}(x,c')$. Our design preserves every existing difference under appending classes: $\Delta_{\mathcal C}=\Delta_{\mathcal C\cup\mathcal N}$ at fixed evidence and parameters. Query adaptation can still vary with $x$.

Candidate-dependent normalization need not preserve a fused ranking. Text scores $(0,2)$ standardize to $(-1,1)$; adding learned contributions $(.25,-.25)$ makes class 2 win. Appending a text score of 100 changes the old standardized scores to $(-.728,-.685)$, making class 1 win after the same fusion (Fig.~\ref{fig:framework}). Fixed references retain the original moments and ranking. This is an existence counterexample, not a claim that every candidate-normalized prediction changes.

'''+method[b:]
method=method.replace('To satisfy Condition 1 without fixing the query response,','To preserve differences while adapting to the query,').replace('satisfying Condition 1.','preserving existing differences.').replace('(Condition 3).','.').replace('Use the same frozen fusion for the closed-base reference and expanded readout. Condition 1 gives','Use the same frozen fusion for the closed-base reference and expanded readout. Fixed within-base predictions give')
method=method.replace('Zero recovery and minimum-variance coupling','Zero-recovery coupling')
a=method.index('At fixed correctness rates $A_C');b=method.index('\\end{corollary}',a)
method=method[:a]+r'''At fixed class-conditional correctness rates, $G=0$ removes the recovery-dependent term $2rG$ from $\operatorname{Var}(C_d-rR)$.
'''+method[b:]
method=method.replace('Independently retuned readouts use a common original reference in experiments, relative to which recovery is allowed.','This also holds for a fixed prototype readout relative to its own closed-base predictor. A common reference can yield a different error coupling; we evaluate both contracts.')
method=method.replace('Deriving the query-adaptive admission gate','Ridge identity and closed-form admission margin').replace('Condition 3 is realized by combining','The admission margin combines').replace('The full standardized responses implement an \\textbf{implicit query-adaptive group gate}; the scalar $d$ selects its retention level. This closed-form gate adds no router parameters and preserves the base ordering. Router choice interacts with the available base vocabulary: Pets and CIFAR provide only 7 and 20 pseudo-base classes, whereas SUN397 provides 97. Closed-form and learned gates share anchored evidence; Section~4.3 compares their observed operating regimes.','The standardized responses define a closed-form admission margin with no added router parameters; $d$ selects its retention level. Learned routers can use the same anchored evidence. Section~4.3 compares their roles.')
method=method.replace('coefficients stay fixed across arrivals (Supplement K).','coefficients stay fixed across arrivals. Each fusion readout searches the same $6\\times6$ coefficient grid and 101 offset quantiles (3,636 settings).')
method=method.replace('correctness of the original closed-base reference and expanded readout.','correctness of a prespecified closed-base reference and expanded readout.').replace('reference with zero novel admission.','chosen reference with zero novel admission.').replace('across $T=5$ prespecified arrivals by $\\delta$.','across $T$ prespecified arrivals by $\\delta$ ($T=5$ in the main expansion studies and 10 on CUB).')
intro=r'''% Anonymous CVPR 2027 manuscript; official style unchanged.
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
\input{preamble}
\title{Reference-Anchored Asymmetric Class Expansion with Query-Adaptive Evidence Alignment}
\author{Anonymous authors}
\begin{document}
\maketitle
\begin{abstract}
Expanding a trained classifier with few-shot classes requires aligning evidence of different origins while retaining base accuracy. Candidate-dependent normalization can reverse existing fused rankings when classes are appended. We anchor each branch to fixed base references, preserving existing scores while adapting to each query, and derive a closed-form admission margin from a ridge mean identity. On SUN397, novel capacity improves over dense prototypes by 7.79/10.14 percentage points at matched base accuracy (CLIP L/14, B/32), and over a class-mean cache by 11.44/6.67 points at the common 95\% base floor. Breaking query--moment correspondence costs 4.38--4.96/7.52--8.10 points. On Food-101, Full and dense-prototype capacities are comparable. Under a common retention requirement, frozen-CLIP transfers reach 73.15\%/47.11\% novel accuracy on CUB-200, admitting all three calibration splits. Frozen source readouts also attain 82.99\%/54.87\% on ImageNet-Sketch without target calibration labels. Reporting common and method-specific retention references reveals how the chosen contract changes method rankings.
\end{abstract}
\section{Introduction}
Growing vocabularies bring unequal evidence: base classes have a fitted classifier, while arriving classes provide names and a few images. Recognition must combine learned, semantic and support evidence under an explicit base-accuracy requirement. Their score frames should adapt to the query and remain stable as the vocabulary grows.

\begin{figure}[!ht]
\centering\includegraphics[width=\columnwidth]{principle_framework.pdf}
\caption{Design principle. (a) Appending an unrelated text score changes candidate-based moments and flips an existing fused ranking. (b) Fixed base cohorts preserve the score frame while their responses adapt to each query. The admission offset $d$ controls base--novel competition.}
\label{fig:framework}
\end{figure}

Lightweight expansion is supported by pretrained representations~\cite{radford2021clip}, prototype and analytic readouts~\cite{zhou2024simplecil,mcdonnell2023ranpac}, and multimodal calibration~\cite{chen2025bimc,lee2026hycal}. A remaining design question is which population should define a branch's score origin and scale. Candidate-dependent moments change when classes are appended. For example, old text scores $(0,2)$ fused with learned contributions $(.25,-.25)$ favor class 2 after standardization; appending a text score of 100 reverses the old fused ranking (Fig.~\ref{fig:framework}). No existing weight changes.

We use a fixed base-class reference bank for each branch. Its members stay fixed, but their responses yield query-dependent moments. This couples vocabulary-growth invariance with per-query adaptation. A ridge mean identity exposes the resulting admission margin without adding router parameters; cardinality-aware interpolation stabilizes moments for small reference banks. Independent calibration then selects the novel-class offset against a stated base reference.

Our contributions are fourfold:
\begin{itemize}
\item A fixed-reference design principle that preserves existing score differences under vocabulary growth while adapting heterogeneous evidence to each query.
\item Recognition evidence from SUN397 at matched base accuracy, cache comparisons, query-moment permutations, and CUB-200 under a common retention requirement.
\item A retention-constrained admission protocol, evaluated across encoders and in source-frozen, zero-target-label transfer to ImageNet-Sketch.
\item A reference-sensitivity analysis showing that common and method-specific accuracy contracts can change method rankings; both are reported.
\end{itemize}

\section{Related work}
\paragraph{Unequal supervision and class expansion}
Long-tailed losses and fitting address unequal supervision~\cite{liu2019large,cao2019ldam,cui2019classbalanced}. CLIP adaptation uses prompts~\cite{zhou2022coop,shi2024lift}, prototypes~\cite{snell2017prototypical,wang2019simpleshot} and caches~\cite{zhang2022tipadapter}. Incremental recognition studies retention across arrivals~\cite{rebuffi2017icarl,tao2020fewshot}; SimpleCIL, RanPAC and FeNeC develop lightweight readouts~\cite{zhou2024simplecil,mcdonnell2023ranpac,ksiazek2026fenec}. We study long-tailed base fitting and asymmetric five-shot expansion, and additionally evaluate the official CUB class/support sequence with matched frozen-feature transfers.
\paragraph{Score normalization and calibration}
T-norm uses query-dependent cohort statistics; Z-norm estimates target-dependent score distributions~\cite{auckenthaler2000score,mariethoz2005unified}. A fixed-cohort T-norm already shares our unshrunk normalization operation. Our study connects this operation to vocabulary growth, heterogeneous fusion, the ridge mean identity and retention contracts. Vision--language logit rescaling~\cite{murugesan2024robust}, BiMC~\cite{chen2025bimc} and HyCal~\cite{lee2026hycal} address related alignment problems. The novel offset is a calibrated-stacking decision variable~\cite{chao2016calibrated}; independent selection uses risk-controlled testing~\cite{angelopoulos2021ltt,waudby2024betting}.

'''
exp=r'''\section{Experiments}
\subsection{Protocol and reference contracts}
\label{sec:tenpartitions}
We use frozen CLIP L/14 and B/32, eight text templates and five novel supports. CIFAR-100~\cite{krizhevsky2009cifar}, Pets~\cite{parkhi2012cats} and ImageNet-R~\cite{hendrycks2021manyfaces} have 40/40, 15/15 and 100/100 base/novel classes; Food-101 and SUN397 have 50/50 and 195/195. Ten class partitions share images, with two five-arrival orders and long-tailed base fitting. \textbf{Full} always denotes $\eta_B=\min(1,B/30)$ at both pseudo and actual base stages, with 3,636 independent base-only fusion settings. Dense replaces learned base evidence by dense prototypes and has its own fusion and stacking offset. None removes centering; Fixed uses base-tuning moments. The cache control independently searches 36 affinity/mixing pairs and 101 offsets.

A \textbf{common reference} fixes the same closed-base Full predictor for every method; a \textbf{self reference} uses each method's own frozen closed-base predictor. Both test $A_B\geq.95A_{\rm ref}$ with identical calibration images. Self-reference floors can differ in absolute accuracy. We report novel/base accuracy, finite admission and target attainment; fallback has zero novel accuracy. Intervals on the expansion studies bootstrap ten partition means after averaging orders, conditional on the fixed images.

CUB-200 follows the official 100-base plus ten 10-class arrivals and five-shot support indices. The standard track uses all 3,000 official base training images. The certification track reserves five tuning and ten calibration images per base class before fitting, identically for all methods, with three split seeds. The CLIP-based SimpleCIL, RanPAC, BiMC and HyCal comparisons are matched transfers; they do not reproduce each paper's original backbone/training setting. CUB allocates confidence across ten arrivals. The supplement's reading index locates implementation, access and verification details.

\begin{table*}[!t]
\centering
\caption{Recognition and certified admission with the same Full definition and fusion budget. AIA is unconstrained average incremental accuracy; Base/Novel are final certified accuracies under the common reference, with finite counts in parentheses. SUN/Food: macro means over ten partitions and two orders, denominator 20. CUB: standard-track AIA is Top-1 over the official class sequence; certified Base/Novel average three separate held-out training splits, denominator 3. These tracks have different fitting access. All displayed final common-reference targets are empirically met.}
\label{tab:recognition}\label{tab:admission}
{\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.08}\input{table_cvpr_primary}}
\end{table*}

\subsection{Recognition under a shared base requirement}
SUN397 provides both recognition and selected-policy gains (Table~\ref{tab:recognition}). At Dense's realized base accuracy, Full adds 7.79 [5.57, 10.22]/10.14 [7.88, 12.57] novel-capacity points (L/14/B/32), with all partitions feasible and no fallback. Against the class-mean cache, gains at the common 95\% floor are 11.44 [8.73, 14.82]/6.67 [5.42, 7.84] points. A separate conservative matching of the cache frontier to Full's achieved base accuracy yields 14.29 [11.68, 17.51]/15.64 [10.56, 21.79] points on seven/ten feasible partitions; the two matching directions are not interchangeable.

On CUB, Full attains 73.15\%/47.11\% final novel accuracy under the common requirement, with 3/3 finite admissions and targets met. Table~\ref{tab:recognition} also shows the unconstrained results: BiMC leads L/14 AIA, while Full has the largest B/32 AIA in this matched-transfer study. Three calibration splits on one official class order give limited precision; the supplement reports split values and conditional intervals. On ImageNet-R, independent fusion tuning gives 3.60 [2.55, 4.74]/5.29 [3.74, 7.25] capacity points over None. These capacities are test-label frontier measurements, distinct from independently selected policies.

\subsection{Which components explain the gains?}
\begin{figure*}[!t]
\centering\includegraphics[width=.90\textwidth]{query_shuffle_main.pdf}
\caption{Query--moment correspondence on ImageNet-R. Each permutation preserves the moment distributions but breaks correspondence to images. All three paired capacity contrasts favor Full; error bars are 95\% intervals over ten partitions. Each control independently reselects fusion under the same budget.}
\label{fig:capacity}
\end{figure*}

The three prespecified moment permutations cost 4.38--4.96/7.52--8.10 capacity points (Fig.~\ref{fig:capacity}), supporting the contribution of query-specific alignment. Fixing prototype centering costs 1.70/1.67 points; fixing learned dispersion costs 0.03/1.41. Thus centering and scaling contribute differently across encoders. The combined ImageNet-R margin AUROC rises from .924 to .977 (L/14) when query centering is added on fixed original readouts.

\paragraph{Shared experts and learned routers}
To isolate routing from classification, both diagnostic gates emit the same frozen Ridge base winner and the same novel winner, and each retunes its margin independently. On SUN397, the prototype-score gate exceeds the learned-score gate by 1.02 [0.42, 1.60]/1.17 [0.52, 1.89] capacity points. This places the larger original Full--Dense differences in the joint choice of aligned evidence and classification experts; it does not identify a unique contribution of the closed-form gate. Both shared-expert gates have $G=0$.

Learned Logistic/MLP routers use the same anchored evidence and improve SUN capacity by 0.88--1.52 points, with 13/225 router parameters. The closed-form margin provides competitive recognition with no added router parameters or routing fit; the same certification procedure applies to both. Vocabulary size and domain vary together in these comparisons.

\subsection{Admission and reference sensitivity}
\label{sec:operational}\label{sec:riskexperiment}\label{sec:budgetexperiment}
Table~\ref{tab:references} reports both contracts for every headline deployment setting. On SUN, switching Dense from the common reference to its own reference raises novel accuracy from 55.01\%/44.36\% to 65.63\%/57.23\%; Full remains at 63.72\%/55.00\%. The ranking changes because the underlying accuracy requirement changes. Common-reference results compare a shared deployment contract; self-reference results measure preservation of each method's starting point.

The same effect explains part of the source-domain policy gaps. On CIFAR L/14, independently retuned Dense moves from 34.60\% novel accuracy (12/20 admissions) to 72.82\% (20/20) under its own reference. HyCal also recovers useful admission with its true closed-vocabulary predictor: 52.79\%/33.88\% novel accuracy on CIFAR (20/20 each). Its measured internal ranking changes are 0.012--0.793 points across source conditions. The transferred components are numerically checked against author code; these scores are not a reproduction of HyCal's published setting.

\begin{table*}[!t]
\centering
\caption{Reference sensitivity: final novel accuracy (\%) and finite admissions, common/self reference. SUN uses Full; Sketch uses the frozen unshrunk source readouts ($\eta=1$) and source calibration only; CUB uses the separate certification track. Denominators are 20, 10 and 3, respectively. Self-reference changes the absolute base target; complete base accuracies, reference floors, target attainment and intervals are in the supplement. All final targets are met except Sketch B/32 HyCal under self-reference (9/10).}
\label{tab:references}
{\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.08}\input{table_cvpr_references}}
\end{table*}

At fixed correctness rates, zero recovery removes $2rG$ from paired variance. It is shared by any invariant within-base predictor relative to itself, including Dense. Small-bank coverage studies separate recognition from statistical power: at $B=5$, cap 10/20 cannot certify even the unchanged reference despite about 90.34\% L/14 oracle capacity. The reference tests therefore complement, rather than replace, matched-base recognition comparisons.

\subsection{Frozen-model transfer and other encoders}
\label{sec:shift}
The source ImageNet-R readouts transfer to matching ImageNet-Sketch classes~\cite{wang2019robustglobal}, with encoder, classifier, five-shot prototypes and fusion frozen. This transfer study uses the unshrunk source setting ($\eta=1$), separately labeled from Full. With zero target calibration labels, the retained source model attains 82.99\%/54.87\% novel accuracy under the common source-reference contract, admitting and meeting test targets in all ten partitions across all five arrivals. Target recalibration gives 86.82\%/58.96\% by changing only the offset (Fig.~\ref{fig:sketchMain}). Table~\ref{tab:references} gives the corresponding self-reference results.

\begin{figure*}[!t]
\centering\includegraphics[width=.88\textwidth]{sketch_operating_frontiers.pdf}
\caption{Source-frozen transfer on Sketch, prespecified partition 0. Open/filled circles use source/target calibration; crosses indicate fallback. Dotted lines are the common base targets. Test-label frontiers describe recognition; calibration selects the operating points independently. All partitions and both reference contracts are reported.}
\label{fig:sketchMain}
\end{figure*}

Convolutional CLIP RN50 reaches 45.17\% capacity and 41.40\% common-reference novel accuracy on ImageNet-R, with 10/10 admissions and targets met. Capacity gains over None/Fixed are 4.23 [2.80, 5.56]/3.25 [1.82, 4.75] points. SigLIP-B/16~\cite{zhai2023siglip} gives 85.68\% common-reference novel accuracy and a 1.34 [0.82, 1.89]-point capacity gain over None. Softmax-head controls further test the alignment mechanism beyond ridge fitting (Supplement S).

\subsection{Scope and mixed results}
Food-101 Full and Dense capacities differ by $-0.04$ [$-0.17$, .10]/$-0.21$ [$-.83$, .38] points. These intervals support comparable observed performance, not a universal benefit from retaining a linear head. Fixed-cohort T-norm is exactly the unshrunk operator; cardinality interpolation stabilizes small banks, with mixed effects across tasks (Table~\ref{tab:tnorm}). Constants 10/20/30/50 are prespecified sensitivity controls, with 30 fixed as the main rule. Learned routers can improve recognition on SUN, and the shared-expert control does not favor our particular margin over prototype routing. CUB covers one official class sequence and three certification splits. Finally, retention guarantees concern the calibration distribution; zero-label transfer is empirical evidence, not certification under arbitrary shift.

\begin{table}[!t]
\centering
\caption{Explicit fixed-cohort T-norm control: Full minus T-norm on Pets (pp), common 95\% base target. L/14 and B/32 share the same features, heads, supports and search budget. Paired intervals and all six conditions, including CIFAR's negative mean differences and ImageNet-R's exact equality, are reported in Supplement AB.}
\label{tab:tnorm}
{\small\setlength{\tabcolsep}{3pt}\input{table_cvpr_tnorm_compact}}
\end{table}

\section{Conclusion}
Fixed base references make query-adaptive evidence alignment compatible with vocabulary growth. Matched-base recognition gains, moment-permutation controls and frozen-source transfer support this design. Independent admission testing turns aligned scores into policies under an explicit retention contract, whose choice is essential to interpreting comparisons.
\label{end:maintext}
\section*{LLM Usage Statement}
OpenAI Codex assisted literature discovery, manuscript drafting and structural revision, experiment and verification code, and LaTeX tables. This included code generation and substantial language editing. The authors are responsible for all claims, citations, results and the final submission.
\clearpage
{\small\bibliographystyle{ieeenat_fullname}\bibliography{references}}
\end{document}
'''
(H/'main.tex').write_text(intro+method+exp,encoding='utf-8')
print('rewritten main')
