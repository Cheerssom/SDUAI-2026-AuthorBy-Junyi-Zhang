"""Results figure for main_34 (fig:results): four panels in one row.

(a) miniImageNet and (b) CIFAR-100 session-wise top-1 accuracy, CLIP ViT-B/16;
(c) unconstrained average accuracy versus certified new-class accuracy (B/16, mean over the
    three standard benchmarks) for every readout;
(d) certified new-class accuracy averaged over the three benchmarks, per encoder.

Input : ../../../experiments/autodl_fecam_replay2_20261108/summary.json
Output: ../fig_results.pdf (vector) and ../fig_results.png (preview)
Run   : python make_fig_results.py
Requires matplotlib only. Colors are the validated categorical palette (slots 1-8, fixed
entity order); run validate_palette.js as described in README_ZH.md before finalising.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parents[2] / 'experiments' / 'autodl_round4_20261109' / 'extracted' / 'work' / 'summary.json'
OUT = HERE.parent / 'fig_results_grid'

# Fixed entity -> color (validated palette, slots 1..8 in order); color follows the entity in every panel.
ENT = [  # key, label, color, marker
    ('ZeroShot_CLIP', 'CLIP zero-shot', '#2a78d6', 'v'),
    ('GDA_transfer', 'GDA', '#eb6834', 's'),
    ('FeCAM_transfer', 'FeCAM', '#1baf7a', 'D'),
    ('RanPAC_frozen_transfer', 'RanPAC', '#eda100', 'P'),
    ('dense', 'Dense', '#e87ba4', '^'),
    ('BiMC_transfer', 'BiMC', '#008300', 'o'),
    ('KLDA_transfer', 'KLDA', '#6250d6', 'X'),
    ('full_gda', 'AnchorCE (Ours)', '#e34948', '*'),
]
OTHERS = [('HyCal_transfer', 'HyCal', '#898781', 'h'), ('TEEN_transfer', 'TEEN', '#898781', '<'),
          ('SimpleCIL_transfer', 'SimpleCIL', '#898781', '>')]   # entities beyond 8 fold into muted gray
COLOR = {k: c for k, _, c, _ in ENT + OTHERS}
LABEL = {k: l for k, l, _, _ in ENT + OTHERS}
MARK = {k: m for k, _, _, m in ENT + OTHERS}
OURS = 'full_gda'
INK, INK2, GRID, AXIS = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.2, 'axes.labelsize': 7, 'axes.titlesize': 7.6,
    'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'legend.fontsize': 7,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.labelcolor': INK, 'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

S = json.loads(SUMMARY.read_text())
std, cert, avg = S['standard'], S['certified'], S['average']


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def sessions(ax, ds, title):
    keys = ['ZeroShot_CLIP', 'GDA_transfer', 'FeCAM_transfer', 'dense', 'BiMC_transfer', OURS]
    for k in keys:
        y = std[f'{ds}/b16/{k}']['session_top1']
        ours = k == OURS
        ax.plot(range(len(y)), y, color=COLOR[k], marker=MARK[k], markersize=5.5 if ours else 3.2,
                linewidth=1.8 if ours else 1.0, markeredgecolor='white', markeredgewidth=0.4,
                zorder=5 if ours else 3, label=LABEL[k])
    ax.set_xlabel('Incremental session')
    ax.set_ylabel('Top-1 accuracy (%)')
    ax.set_xticks(range(0, len(y), 2))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_title(title, color=INK)
    style(ax)


def tradeoff(ax):
    keys = [k for k, *_ in ENT] + [k for k, *_ in OTHERS]
    # (dx, dy, ha) label offsets in data units, set per point to avoid collisions
    pos = {'full_gda': (0.0, 8.0, 'center'), 'GDA_transfer': (0.0, -8.5, 'center'), 'FeCAM_transfer': (-0.5, 0.0, 'right'),
           'KLDA_transfer': (-0.5, 0.0, 'right'), 'BiMC_transfer': (0.35, 5.0, 'left'), 'dense': (-0.35, -7.5, 'right'),
           'RanPAC_frozen_transfer': (0.0, -8.0, 'center'), 'HyCal_transfer': (0.35, 4.0, 'left'),
           'TEEN_transfer': (0.55, 0.0, 'left'), 'SimpleCIL_transfer': None, 'ZeroShot_CLIP': (0.0, 7.5, 'center')}
    text = {'TEEN_transfer': 'TEEN, SimpleCIL'}
    for k in keys:
        a = avg[f'b16/{k}']
        x, y = a['Avg'], a['certified_common']
        ours = k == OURS
        ax.scatter(x, y, s=110 if ours else 28, color=COLOR[k], marker=MARK[k], edgecolor='white',
                   linewidth=0.5, zorder=5 if ours else 3)
        if pos[k] is None:
            continue
        dx, dy, ha = pos[k]
        ax.text(x + dx, y + dy, text.get(k, LABEL[k].replace(' (Ours)', '')), fontsize=5.2, ha=ha, va='center',
                color=INK if ours else INK2, fontweight='bold' if ours else 'normal', zorder=6)
    ax.set_xlabel('Average accuracy (%)')
    ax.set_ylabel('Certified acc. (%)')
    ax.set_title('Accuracy vs. certification', color=INK)
    ax.set_xlim(70.0, 84.2)
    ax.set_ylim(-14, 82)
    style(ax)


def encoders(ax):
    keys = ['BiMC_transfer', 'FeCAM_transfer', 'GDA_transfer', OURS]
    encs = [('b16', 'B/16'), ('l14', 'L/14'), ('b32', 'B/32')]
    w = 0.19
    for j, k in enumerate(keys):
        vals = [avg[f'{e}/{k}']['certified_common'] for e, _ in encs]
        xs = [i + (j - 1.5) * (w + 0.02) for i in range(len(encs))]
        ax.bar(xs, vals, width=w, color=COLOR[k], edgecolor='white', linewidth=0.6,
               label=LABEL[k], zorder=3)
        if k == OURS:
            for x, v in zip(xs, vals):
                ax.text(x, v + 1.2, f'{v:.1f}', ha='center', va='bottom', fontsize=6, color=INK,
                        fontweight='bold')
    ax.set_xticks(range(len(encs)))
    ax.set_xticklabels([n for _, n in encs])
    ax.set_ylabel('Certified acc. (%)')
    ax.set_title('Across encoders', color=INK)
    ax.set_ylim(0, 92)
    style(ax)
    ax.grid(False, axis='x')


fig, axs = plt.subplots(2, 2, figsize=(3.9, 4.9), gridspec_kw=dict(wspace=0.40, hspace=0.62))
axes = axs.ravel()
sessions(axes[0], 'mini', 'miniImageNet')
sessions(axes[1], 'cifar100', 'CIFAR-100')
tradeoff(axes[2])
encoders(axes[3])
for ax, t in zip(axes, 'abcd'):
    ax.text(0.5, -0.24, f'({t})', transform=ax.transAxes, ha='center', va='top', fontsize=7)
from matplotlib.lines import Line2D
legend_keys = ['ZeroShot_CLIP', 'GDA_transfer', 'FeCAM_transfer', 'RanPAC_frozen_transfer', 'KLDA_transfer', 'dense', 'BiMC_transfer', OURS]
h = [Line2D([0], [0], color=COLOR[k], marker=MARK[k], markersize=5 if k == OURS else 3.5,
            linewidth=1.6 if k == OURS else 0.9, markeredgecolor='white', markeredgewidth=0.4) for k in legend_keys]
fig.legend(h, [LABEL[k] for k in legend_keys], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4,
           frameon=False, handlelength=1.4, columnspacing=0.9, fontsize=6.4)
fig.savefig(str(OUT) + '.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(str(OUT) + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
print('saved', str(OUT) + '.pdf')
