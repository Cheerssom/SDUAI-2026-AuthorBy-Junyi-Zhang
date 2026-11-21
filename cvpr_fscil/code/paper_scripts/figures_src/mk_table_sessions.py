"""Session-wise top-1 table (miniImageNet, CLIP ViT-B/16) in HyCal's style. Usage: python mk_table_sessions.py <summary.json> [zh]"""
import json, sys
S = json.load(open(sys.argv[1]))['standard']; zh = len(sys.argv) > 2
M = [('ZeroShot_CLIP', 'CLIP zero-shot'), ('SimpleCIL_transfer', 'SimpleCIL'), ('TEEN_transfer', 'TEEN'), ('FeCAM_transfer', 'FeCAM'),
     ('RanPAC_frozen_transfer', 'RanPAC'), ('GDA_transfer', 'GDA'), ('KLDA_transfer', 'KLDA'), ('HyCal_transfer', 'HyCal'),
     ('BiMC_transfer', 'BiMC'), ('dense', 'Dense'), ('full_gda', '\\textbf{\\textsc{AnchorCE} (Ours)}')]
rows = [[S[f'mini/b16/{m}']['session_top1'][t] for t in range(9)] + [S[f'mini/b16/{m}']['Avg'], S[f'mini/b16/{m}']['PD']] for m, _ in M]
cells = [[f'{v:.2f}' for v in r] for r in rows]
for j in range(11):
    vals = [r[j] for k, r in enumerate(rows) if k]  # zero-shot is a reference row, not ranked
    u = sorted({round(v, 2) for v in vals}, reverse=(j != 10))
    for k, r in enumerate(rows):
        if not k: continue
        v = round(r[j], 2)
        if v == u[0]: cells[k][j] = f'\\textbf{{{cells[k][j]}}}'
        elif v == u[1]: cells[k][j] = f'\\underline{{{cells[k][j]}}}'
head = '各会话准确率（\\%）' if zh else 'Accuracy in each session (\\%)'
L = ['\\begin{tabular}{@{}lccccccccc@{\\hspace{8pt}}cc@{}}', '\\toprule',
     '\\rowcolor{gray!15}\\multicolumn{12}{c}{\\textbf{miniImageNet, CLIP ViT-B/16}} \\\\', '\\midrule',
     f'& \\multicolumn{{9}}{{c}}{{{head}$\\uparrow$}} & & \\\\', '\\cmidrule(lr){2-10}',
     ('方法' if zh else 'Method') + ' & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & Avg$\\uparrow$ & PD$\\downarrow$ \\\\', '\\midrule']
for k, ((m, lab), c) in enumerate(zip(M, cells)):
    L.append(lab + ' & ' + ' & '.join(c) + ' \\\\')
    if k == 0: L.append('\\midrule')
L += ['\\bottomrule', '\\end{tabular}']
print('\n'.join(L))
