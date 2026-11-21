from pathlib import Path
import re
p=Path(__file__).resolve().parent/'main.tex';s=p.read_text(encoding='utf-8')
def move(s,env,needle,anchor):
    blocks=list(re.finditer(r'\\begin\{'+re.escape(env)+r'\}\[t\].*?\\end\{'+re.escape(env)+r'\}\s*',s,re.S))
    m=next(m for m in blocks if needle in m[0]); block=m[0]
    s=s[:m.start()]+s[m.end():];pos=s.index(anchor)
    return s[:pos]+block+'\n'+s[pos:]
s=move(s,'table*','table_cvpr_recognition',r'\section{Experiments}')
s=move(s,'figure*','matched_capacity_main.pdf',r'\section{Experiments}')
s=move(s,'table*','table_cvpr_centering',r'\subsection{Query-adaptive alignment improves')
s=move(s,'figure*','sketch_operating_frontiers.pdf',r'\subsection{Query-adaptive alignment improves')
s=move(s,'table*','table_cvpr_admission',r'\subsection{Useful admission')
s=s.replace(r'\begin{figure*}[t]',r'\begin{figure*}[!t]').replace(r'\begin{table*}[t]',r'\begin{table*}[!t]')
s=s.replace('\\FloatBarrier\n\\section{Conclusion}',r'\section{Conclusion}')
p.write_text(s,encoding='utf-8')
