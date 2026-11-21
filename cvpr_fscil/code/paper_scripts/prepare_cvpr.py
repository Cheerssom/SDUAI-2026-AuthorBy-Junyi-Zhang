"""One-time migration from the frozen KBS source to a separate CVPR project."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile
H=Path(__file__).resolve().parent; K=H.parent/'kbs'
O=H/'template_provenance/official_2027'
title='Reference-Anchored Asymmetric Class Expansion with Query-Adaptive Evidence Alignment'
with zipfile.ZipFile(K/'candidate_tables_kbs_source.zip') as z:
    for n in z.namelist():
        if n.endswith(('.pdf','.tex','.bib')) and n not in ['main.tex','supplementary.tex']:
            (H/n).write_bytes(z.read(n))
for n in ['cvpr.sty','ieeenat_fullname.bst']:
    shutil.copy2(O/n,H/n)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prov={'date':'2026-10-27','official_guidelines':'https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines',
      'official_template_url':'https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip',
      'user_template_year':2026,'active_template_year':2027,
      'official_style_sha256':sha(O/'cvpr.sty'),'official_bst_sha256':sha(O/'ieeenat_fullname.bst'),
      'official_zip_sha256':sha(H/'template_provenance/official_2027.zip'),
      'source_kbs_main_sha256':sha(K/'main.tex'),'source_kbs_supplement_sha256':sha(K/'supplementary.tex')}
(H/'template_provenance.json').write_text(json.dumps(prov,indent=2)+'\n',encoding='utf-8')
preamble=r'''\usepackage[T1]{fontenc}
\usepackage{amsthm}
\usepackage{xurl}
\usepackage{placeins}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{theorem}{Theorem}
\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=cvprblue]{hyperref}
\hypersetup{pdfauthor={},pdftitle={Reference-Anchored Asymmetric Class Expansion with Query-Adaptive Evidence Alignment}}
\def\confName{CVPR}
\def\confYear{2027}
\input{numbers}
\input{retention_numbers}
% Diagnostic threshold only: expose overflow hidden by the stock style.
\hfuzz=0pt
'''
(H/'preamble.tex').write_text(preamble,encoding='utf-8')
source=(K/'main.tex').read_text(encoding='utf-8')
abstract=source.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0]
body=source[source.index(r'\section{Introduction}'):source.index(r'\section*{CRediT')]
# Preserve all scientific content initially; tailor layout for two columns.
body=body.replace('KBS approaches develop','Related approaches develop')
body=body.replace(r'\centering\setlength{\belowcaptionskip}{8pt}',r'\centering')
body=body.replace('[!htbp]','[t]')
body=re.sub(r'\\begin\{figure\}\[t\](.*?)\\end\{figure\}',
            lambda m:r'\begin{figure*}[t]'+m[1]+r'\end{figure*}',body,flags=re.S)
body=re.sub(r'\\begin\{table\}\[t\](.*?)\\end\{table\}',
            lambda m:r'\begin{table*}[t]'+m[1]+r'\end{table*}',body,flags=re.S)
body=body.replace(r'\includegraphics[width=.98\textwidth]',r'\includegraphics[width=\textwidth]')
main=r'''% CVPR 2027 review manuscript; official style is unmodified.
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
\input{preamble}
\title{'''+title+r'''}
\author{Anonymous authors}
\begin{document}
\maketitle
\begin{abstract}
'''+abstract+r'''
\end{abstract}
'''+body+r'''
\label{end:maintext}
\clearpage
\section*{LLM Usage Statement}
OpenAI Codex assisted literature discovery, drafting and revision throughout the manuscript, experiment and verification scripts, and LaTeX and table preparation. Its use included code generation and substantial language and structural editing. The authors remain responsible for the scientific claims, citations, experimental results and final submission.
{\small
\bibliographystyle{ieeenat_fullname}
\bibliography{references}
}
\end{document}
'''
(H/'main.tex').write_text(main,encoding='utf-8')
supp=(K/'supplementary.tex').read_text(encoding='utf-8')
supp=supp[supp.index(r'\appendix'):]
supp=supp.replace(r'\bibliographystyle{elsarticle-num}',r'\bibliographystyle{ieeenat_fullname}')
head=r'''% Anonymous supplementary evidence; single-column layout permitted by CVPR.
\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{times,amsmath,amssymb,amsthm,booktabs,graphicx,placeins,float}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdfauthor={},pdftitle={Supplementary Material: Reference-Anchored Asymmetric Class Expansion with Query-Adaptive Evidence Alignment}}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\input{numbers}
\input{retention_numbers}
\usepackage{xr}\externaldocument[main-]{main}
\begin{document}
\title{Supplementary Material: '''+title+r'''}
\author{Anonymous CVPR 2027 submission}
\date{}
\maketitle
'''
(H/'supplementary.tex').write_text(head+supp,encoding='utf-8')
print('Created separate CVPR main and anonymous supplementary sources.')
