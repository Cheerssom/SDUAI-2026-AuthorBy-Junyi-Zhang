"""Render prespecified paired contrasts from verified experiment summaries."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent

def interval(row):
    if row['mean'] is None:return '--'
    return f"{row['mean']:+.2f} [{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}]"

d=json.loads((H/'cross_encoder/summary.json').read_text())
names={'dense_query':'Dense/Query','retained_none':'Retained/None','dense_none':'Dense/None',
       'retained_fixed':'Retained/Fixed','dense_fixed':'Dense/Fixed','hycal':'HyCal transfer'}
lines=[r'\begin{tabular}{@{}lrrrc@{}}',r'\toprule',
       r'Comparator & AIA gain & Capacity gain & Policy gain & $n$ \\',r'\midrule']
for row in d['comparisons']:
    lines.append(' & '.join([names[row['comparator']],interval(row['aia_gain']),interval(row['capacity_gain']),
                             interval(row['policy_gain']),str(len(row['joint_parts']))])+r' \\')
lines += [r'\bottomrule',r'\end{tabular}']
(H/'table_cross_encoder_pairs.tex').write_text('\n'.join(lines)+'\n')
d=json.loads((H/'shrinkage/summary.json').read_text())
lines=[r'\begin{tabular}{@{}lllrrc@{}}',r'\toprule',
       r'Encoder & Selection & Comparator & Capacity gain & Policy gain & $n$ \\',r'\midrule']
for row in d['comparisons']:
    if row['dataset']!='pets' or row['method']=='half':continue
    enc={'l14':'L/14','b32':'B/32'}[row['encoder']]
    method={'adaptive_eta':r'Tuned $\eta$','adaptive_scheme':'Tuned scheme'}[row['method']]
    lines.append(' & '.join([enc,method,row['comparator'].title(),interval(row['capacity_gain']),
                             interval(row['policy_gain']),str(len(row['joint_parts']))])+r' \\')
lines += [r'\bottomrule',r'\end{tabular}']
(H/'table_shrinkage_pairs.tex').write_text('\n'.join(lines)+'\n')
print('Paired contrast tables complete.')
