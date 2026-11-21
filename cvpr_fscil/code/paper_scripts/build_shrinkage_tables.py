from pathlib import Path
import json
H=Path(__file__).resolve().parent;D=json.loads((H/'shrinkage/summary.json').read_text())
names={'query':'Query','none':'None','fixed':'Fixed','half':r'$\eta=.5$','adaptive_eta':r'Tuned $\eta$','adaptive_scheme':'Tuned scheme'}
for ds in ['pets','cifar','imagenetr']:
    lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Encoder & Scheme & AIA & Capacity ($n$) & Base & Novel (cond.) & Finite/Met \\',r'\midrule']
    for enc in ['l14','b32']:
        for m in names:
            r=next(r for r in D['conditions'] if (r['encoder'],r['dataset'],r['method'])==(enc,ds,m))
            c=r['capacity'];cv='--' if c['mean'] is None else f"{c['mean']:.2f}"
            cond=r['conditional_novel']['mean'];cs='--' if cond is None else f'{cond:.2f}'
            lines.append(f"{enc.replace('l14','L/14').replace('b32','B/32')} & {names[m]} & {r['aia']['mean']:.2f} & {cv} ({c['n']}) & {r['base']['mean']:.2f} & {r['novel']['mean']:.2f} ({cs}) & {r['finite']}/{r['test_met']}"+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
    (H/f'table_shrinkage_{ds}.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Encoder & Dataset & $\eta=1$ & $\eta=.5$ & $\eta=0$ & None \\',r'\midrule']
for r in D['conditions']:
    if r['method']!='adaptive_scheme':continue
    c=r['chosen_schemes'];ds={'pets':'Pets','cifar':'CIFAR-100','imagenetr':'ImageNet-R'}[r['dataset']];lines.append(f"{r['encoder'].replace('l14','L/14').replace('b32','B/32')} & {ds} & {c.get('query',0)} & {c.get('half',0)} & {c.get('fixed',0)} & {c.get('none',0)}"+r' \\')
lines.extend([r'\bottomrule',r'\end{tabular}']);(H/'table_shrinkage_choices.tex').write_text('\n'.join(lines)+'\n')
print('All shrinkage schemes and selections tabled.')
