from pathlib import Path
import re
H=Path(__file__).resolve().parent
p=H/'main.tex';s=p.read_text(encoding='utf-8')
def relocate(text,asset,anchor):
    pattern=r'\\begin\{figure\*\}\[t\]\s*\\centering\\includegraphics\[width=\\textwidth\]\{'+re.escape(asset)+r'\}.*?\\end\{figure\*\}\s*'
    m=re.search(pattern,text,re.S); assert m,asset
    block=m[0];text=text[:m.start()]+text[m.end():]
    i=text.index(anchor)
    return text[:i]+block+'\n'+text[i:]
s=relocate(s,'framework.pdf',r'Pretrained representations make')
s=relocate(s,'stability.pdf',r'\begin{proposition}[Vocabulary-growth invariance]')
s=s.replace(r'\includegraphics[width=\textwidth]{matched_capacity_main.pdf}',r'\includegraphics[width=.80\textwidth]{matched_capacity_main.pdf}')
s=s.replace(r'\includegraphics[width=\textwidth]{sketch_operating_frontiers.pdf}',r'\includegraphics[width=.90\textwidth]{sketch_operating_frontiers.pdf}')
# Keep a complete conclusion together after pending results.
s=s.replace(r'\section{Conclusion}',r'\FloatBarrier'+'\n'+r'\section{Conclusion}')
p.write_text(s,encoding='utf-8')
p=H/'supplementary.tex';s=p.read_text(encoding='utf-8')
s=s.replace(r'\usepackage{xurl}',r'\usepackage{xurl,microtype}')
s=s.replace(r'\%/',r'\%/\allowbreak ')
s=s.replace('dataset/encoder/partition runs','runs over datasets, encoders and partitions')
s=s.replace('fallback-inclusive/conditional novel accuracy','fallback-inclusive and admission-conditioned novel accuracy')
s=s.replace(r'\texttt{revision\_summary.csv}',r'\nolinkurl{revision_summary.csv}')
p.write_text(s,encoding='utf-8')
print('Adjusted floats and supplementary line breaks.')
