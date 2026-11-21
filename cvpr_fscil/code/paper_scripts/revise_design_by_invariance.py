"""Reorganize Section 3 around its objective and structural requirements."""
from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'main.tex';s=p.read_text(encoding='utf-8')
start=s.index(r'\section{Reference-anchored asymmetric class expansion}')
stop=s.index(r'\subsection{Independent admission calibration}',start)
old=s[start:stop]
construction=old[old.index('A frozen encoder produces') if 'A frozen encoder produces' in old else 0:]
# Extract the existing construction and ridge proof; retain executable formulas.
a=old.index('Base classes');b=old.index('For each evidence source')
evidence=old[a:b]
evidence=evidence[evidence.index('A frozen encoder'):].strip()
a=old.index('For each evidence source');b=old.index(r'\paragraph{Why the reference must be fixed}')
operator=old[a:b].strip()
pa=operator.index('The cardinality-aware variant recomputes')
pb=operator.index(r'\begin{align}',pa)
operator=operator[:pa]+r'''The bank size is stage-specific during pseudo-base tuning; Supplement U also tests a frozen catalog weight. Banks, tuning moments and weights stay fixed across actual arrivals. With coefficients $a,u$ and novel offset $d$, asymmetric fusion is
'''+operator[pb:]
operator=operator.replace('Both variants use the aligned evidence operator','The resulting reference-anchored operator is')
operator=operator.replace('using the aligned moments in Eq.~\\eqref{eq:anchored}. Centering adjusts the group margin; scaling also changes within-group fusion.','This offset and the scaled class responses jointly determine admission (Condition 3).')

a=old.index(r'\begin{proposition}[Vocabulary-growth invariance]')
b=old.index(r'\subsection{Query-adaptive base--novel admission}',a)
invariance=old[a:b]
invariance=invariance[:invariance.index('This property concerns')].strip()
invariance=invariance.replace('With fixed input, encoder, existing evidence, reference banks, coefficients and tie-breaking, appending classes preserves every existing score and its within-vocabulary ordering.', 'With fixed input, encoder, existing evidence, reference banks, moments, coefficients and tie-breaking, the anchored operator preserves every existing score and pairwise difference as classes are appended, satisfying Condition 1.')

a=old.index(r'\begin{proposition}[Ridge identity')
b=old.index(r'\subsection{The retention envelope}',a)
gate=old[a:b].strip()
gate=gate.replace(r'\begin{proposition}[Ridge identity and adaptive admission margin]',r'\begin{proposition}[Ridge identity and adaptive admission margin]'+'\n'+r'\label{prop:ridgegate}')
gate=gate.replace('The learned score contributes its deviation from the common ridge mean, $(q_L(x,b)-1/B)/\\widetilde s_L(x)$. The novel prototype contributes its similarity relative to base visual references, $(q_P(x,n)-\\widetilde\\mu_P(x))/\\widetilde s_P(x)$. With the common text-centering term removed, their competition is', 'Condition 3 is realized by combining deviations from the ridge mean with prototype evidence relative to base references. Removing common text centering gives')
ga=gate.index('The system selects the novel group');gb=gate.index('\n\n',ga)
gate=gate[:ga]+r'''Novel admission occurs when $M(x)+d>0$, with fixed class-ID ties. The full standardized responses implement an \textbf{implicit query-adaptive group gate}; the scalar $d$ selects its retention level. Thus reference dispersion modulates the branch contributions without a routing network, while Proposition~\ref{prop:invariance} preserves the base ordering.''' +gate[gb:]
ga=gate.index('Each new class needs');gate=gate[:ga]+r'''After encoding, each arrival requires $O(kD)$ prototype construction and $O(D)$ storage per class; scoring $K$ candidates costs $O(KD)$.'''

opening=r'''\section{Reference-anchored asymmetric class expansion}
Base classes $\mathcal B$ have a fitted classifier; arriving classes $\mathcal N$ provide names and $k$ support images. Our objective is maximal novel recognition under a shared base-accuracy requirement:
\begin{equation}
N_r=\max_d A_N(d)\quad\text{subject to}\quad
A_B(d)\geq rA_{B,\mathrm{ref}}^{\mathrm{closed}}.
\label{eq:retentioncriterion}
\end{equation}
Here $A_B,A_N$ are macro accuracies, $0<r<1$, and $A_{B,\mathrm{ref}}^{\mathrm{closed}}$ is the closed-base reference accuracy. We design the score operator around three structural requirements, then construct the margin on which independent calibration selects $d$.

\subsection{Structural requirements for expansion}
\paragraph{Condition 1: Intra-base order invariance}
Let $\Delta_{\mathcal C}(x;c,c')=s_{\mathcal C}(x,c)-s_{\mathcal C}(x,c')$. Appending classes must preserve $\Delta_{\mathcal C}=\Delta_{\mathcal C\cup\mathcal N}$ for every existing pair, at fixed parameters. This prevents vocabulary growth from changing stored discrimination. Candidate-dependent branch scaling can violate this requirement: old text scores $(0,2)$ fused with $(.25,-.25)$ reverse their ranking after candidate standardization when a text score of $100$ is appended (Supplement K).

\paragraph{Condition 2: Zero-recovery calibration structure}
For base-query correctness $R$ before expansion and $C_d$ afterward, require $C_d\leq R$, hence $G=\Pr(R=0,C_d=1)=0$. This aligns protection with retention and yields the minimum-variance correctness coupling at matched class-conditional accuracies (Corollary~\ref{cor:zerorecovery}).

\paragraph{Condition 3: Query-adaptive group gating}
The base--novel margin must adapt to each query's learned responses and reference dispersion without an auxiliary router, while satisfying Conditions 1--2. Fixed reference banks permit this adaptation independently of vocabulary growth.

\subsection{Constructing the invariant asymmetric operator}
'''

corollary=r'''\begin{corollary}[Zero recovery and minimum-variance coupling]
\label{cor:zerorecovery}
Use the same frozen fusion for the closed-base reference and expanded readout. Condition 1 gives $C_d\leq R$, so $G=0$ and
\begin{equation}
A_B(d)=A_{\mathrm{ref}}-D,\qquad D=\Pr(R=1,C_d=0).
\label{eq:retainederrors}
\end{equation}
At fixed correctness rates $A_C\leq A_R$, this coupling minimizes $\operatorname{Var}(C_d-rR)$ among joint Bernoulli laws with those rates. The result applies to the stratified calibration mean when rates are fixed within each class.
\end{corollary}
\begin{proof}
The expanded winner is the unchanged base winner or a novel label; neither can correct a base error. Since $\mathbb E(RC_d)=A_C-G$, the variance equals $A_C+r^2A_R-2r(A_C-G)-(A_C-rA_R)^2$. At fixed rates its sole coupling-dependent term is $2rG\geq0$, minimized at zero. The protection loss in Eq.~\eqref{eq:protectionloss} then satisfies $\mathbb E L_d-(1-r)=rA_R-A_C$. Supplement P gives the class-wise form.
\end{proof}
Independently retuned readouts use a common original reference in experiments, relative to which recovery is allowed.

\subsection{Deriving the query-adaptive admission gate}
'''

tuning=r'''

\paragraph{Fitting and selecting the retention level}
Base-only pseudo-expansion divides classes into pseudo-base and pseudo-novel groups. Held-out base queries select $a,u,d$ using
\begin{equation}
H=2A_BA_N/(A_B+A_N),
\label{eq:harmonic}
\end{equation}
with zero when both accuracies vanish. Actual novel query and test labels are excluded; coefficients stay fixed across arrivals (Supplement K).

For fixed group winners $h_B,h_N$, increasing $d$ switches each query once from $h_B$ to $h_N$. Hence $A_B(d)$ is nonincreasing and $A_N(d)$ nondecreasing; the largest feasible event solves Eq.~\eqref{eq:retentioncriterion}. Test-label sweeps measure capacity; independent calibration chooses deployable offsets.

'''
new=opening+evidence+'\n\nTo satisfy Condition 1 without fixing the query response, we anchor branch moments to base references.\n'+operator+'\n\n'+invariance+'\n\n'+corollary+gate+tuning
s=s[:start]+new+s[stop:]
p.write_text(s,encoding='utf-8')
for name,old,new in [('validate_cvpr.py',"MAIN_STEM='main_12'","MAIN_STEM='main_13'"),('build_delivery.py','cvpr2027-20261027-requested-narrative','cvpr2027-20261027-design-by-invariance')]:
 p=H/name;p.write_text(p.read_text(encoding='utf-8').replace(old,new),encoding='utf-8')
print('Reordered Section 3: objective, conditions, construction, proofs, calibration.')
