"""Problem-setting figure: the alignment-invariance dilemma seen through one base query.

Visual grammar follows HyCal's Fig. 1 (bold panel titles, boxed labels, light-gray canvas, red emphasis),
but the content is ours: base-class score bars of one base query at session 0 and at session t.
(a) standard FSCIL: model updated, judged by average accuracy only;
(b) active-class calibration (BiMC, HyCal): base scores move -> recovery -> retention test fails;
(c) frozen prototypes (SimpleCIL, TEEN): base scores fixed, trained classifier discarded;
(d) AnchorCE: base-anchored alignment -> base scores fixed with all evidence -> certified.
The earlier tile-stack version is kept in make_fig_paradigm_tiles_v1.py.
Output: ../fig_paradigm.pdf / .png
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Arc

OUT = Path(__file__).resolve().parent.parent / 'fig_paradigm'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
INK, GRAYBG, ARROW = '#111111', '#f1f1f1', '#8c8c8c'
BASE, BASE_E = '#f6c453', '#a8781c'
FADE, FADE_E = '#efe6cf', '#b9ad8d'
NEW, NEW_E = '#8fcf88', '#3f7f3a'
RED, GREEN, BLUE = '#d62728', '#1a7f37', '#1d4e89'

B0 = [0.95, 0.62, 0.40, 0.28]           # base scores at session 0 (true class is the second bar)
X0, X1, BW, GAP, YB = 0.62, 2.72, 0.24, 0.07, 1.02


def label(ax, x, y, s, ec=INK, color=INK):
    ax.text(x, y, s, fontsize=6.2, fontweight='bold', ha='left', va='center', color=color, zorder=6,
            bbox=dict(boxstyle='square,pad=0.2', fc='white', ec=ec, lw=0.7))


def strip(ax, x, y, w):
    ax.add_patch(Rectangle((x, y - 0.16), 1.9, 0.32, fc='#fbf2d7', ec='#c2b285', lw=0.4, zorder=2))
    ax.text(x + 0.08, y, 'base', fontsize=5.6, ha='left', va='center', color='#7a5a12', zorder=3)
    for i, m in enumerate(['o', 's', '^', 'D']):
        ax.plot(x + 0.68 + i * 0.3, y, marker=m, ms=3.8, color='#e07b00', mec='white', mew=0.3, zorder=3)
    ax.add_patch(Rectangle((x + 1.9, y - 0.16), w - 1.9, 0.32, fc='#e3f3df', ec='#c2b285', lw=0.4, zorder=2))
    ax.text(x + 1.98, y, r'+$K$-shot new', fontsize=5.6, ha='left', va='center', color='#2e6b28', zorder=3)
    for i, m in enumerate(['*', 'P']):
        ax.plot(x + 3.05 + i * 0.28, y, marker=m, ms=4.0, color='#2e8b2e', mec='white', mew=0.3, zorder=3)


def bars(ax, x, heights, fc=BASE, ec=BASE_E, ghost=None, new=None, true_idx=1):
    for i, h in enumerate(heights):
        xi = x + i * (BW + GAP)
        if ghost is not None:
            ax.add_patch(Rectangle((xi, YB), BW, ghost[i], fc='none', ec='#9a9a9a', lw=0.6, ls=(0, (1.6, 1.2)), zorder=5))
        ax.add_patch(Rectangle((xi, YB), BW, h, fc=fc, ec=ec, lw=0.5, zorder=4))
        if i == true_idx:
            ax.text(xi + BW / 2, YB - 0.09, '$y$', fontsize=5.6, ha='center', va='center', color=INK)
    n = len(heights)
    if new:
        for j, h in enumerate(new):
            xi = x + (n + j) * (BW + GAP) + 0.06
            ax.add_patch(Rectangle((xi, YB), BW, h, fc=NEW, ec=NEW_E, lw=0.5, zorder=4))
        n += len(new)
    ax.plot([x - 0.06, x + n * (BW + GAP) + 0.06], [YB, YB], color=INK, lw=0.6, zorder=6)


def padlock(ax, x, y, color=BLUE, s=0.11):
    ax.add_patch(Rectangle((x - s, y - s), 2 * s, 1.5 * s, fc=color, ec=color, zorder=8))
    ax.add_patch(Arc((x, y + 0.5 * s), 1.3 * s, 2.4 * s, theta1=0, theta2=180, color=color, lw=1.0, zorder=8))


def panel(ax, title, readout, readout_color=INK):
    ax.set_xlim(0, 5.0); ax.set_ylim(0, 4.3); ax.axis('off')
    ax.text(2.5, 4.16, title, fontsize=7.4, fontweight='bold', ha='center', va='center', color=INK)
    label(ax, 0.2, 3.7, 'Session $t$')
    strip(ax, 1.3, 3.7, 3.55)
    label(ax, 0.2, 3.24, readout, ec=readout_color, color=readout_color)
    ax.add_patch(FancyBboxPatch((0.2, 0.12), 4.65, 2.8, boxstyle='round,pad=0.02,rounding_size=0.12', fc=GRAYBG, ec='none', zorder=1))
    ax.text(X0 + 0.55, 0.7, 'session 0', fontsize=5.8, color=ARROW, ha='center', va='center')
    ax.text(X1 + 0.85, 0.7, 'session $t$', fontsize=5.8, color=ARROW, ha='center', va='center')
    ax.add_patch(FancyArrowPatch((X0 + 1.28, 1.45), (X1 - 0.16, 1.45), arrowstyle='-|>', mutation_scale=7, color=ARROW, lw=0.9, zorder=5))
    ax.text(0.34, 2.74, 'base scores of one base query', fontsize=5.6, style='italic', color='#555555', ha='left', va='center')


def outcome(ax, text, color):
    ax.text(2.52, 0.4, text, fontsize=5.9, color=color, ha='center', va='center', zorder=9, linespacing=1.08)


fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), gridspec_kw=dict(wspace=0.05))
plt.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)

# (a) standard FSCIL: model updated each session, judged by average accuracy
ax = axes[0]
panel(ax, '(a) Standard FSCIL', 'Model updated each session')
bars(ax, X0, B0)
bars(ax, X1, [0.70, 0.80, 0.52, 0.20], ghost=B0, new=[0.55, 0.35])
ax.text(4.42, 2.38, 'Avg.\nacc.', fontsize=5.9, fontweight='bold', ha='center', va='center', color=INK, linespacing=1.0,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=INK, lw=0.6))
outcome(ax, 'judged by average accuracy:\nno guarantee for the deployed model', '#444444')

# (b) active-class calibration: base scores move with the classes that arrive
ax = axes[1]
panel(ax, '(b) Active-class calibration', 'BiMC, HyCal')
bars(ax, X0, B0)
bars(ax, X1, [0.58, 0.86, 0.46, 0.18], ghost=B0, new=[0.62, 0.30], fc='#f3a39e', ec='#a83a33')
ax.plot(X0 + BW / 2, YB + B0[0] + 0.1, marker='v', ms=3.6, color=INK, zorder=9)
ax.text(X0 + BW / 2 + 0.1, YB + B0[0] + 0.13, 'wrong', fontsize=5.4, color=INK, ha='left', va='center')
ax.plot(X1 + (BW + GAP) + BW / 2, YB + 0.86 + 0.1, marker='v', ms=3.6, color=RED, zorder=9)
ax.text(X1 + (BW + GAP) + BW / 2 + 0.1, YB + 0.86 + 0.13, 'repaired', fontsize=5.4, color=RED, ha='left', va='center')
ax.text(X1 + 0.15, 2.55, 'recovery $G>0$', fontsize=5.8, color=RED, ha='left', va='center')
outcome(ax, 'aligned evidence, moving base scores:\nnoisy retention test $\\rightarrow$ fallback', RED)

# (c) frozen prototypes: base scores fixed, trained classifier set aside
ax = axes[2]
panel(ax, '(c) Frozen prototypes', 'SimpleCIL, TEEN')
P0 = [0.55, 0.48, 0.42, 0.36]
bars(ax, X0, P0, fc=FADE, ec=FADE_E)
bars(ax, X1, P0, fc=FADE, ec=FADE_E, new=[0.50, 0.30])
padlock(ax, X1 + 0.3, 2.0)
ax.text(X1 + 0.5, 2.02, 'fixed', fontsize=5.8, color=BLUE, ha='left', va='center')
ax.text(X0 - 0.02, 2.3, 'trained classifier', fontsize=5.7, color='#777777', ha='left', va='center')
ax.plot([X0 - 0.04, X0 + 1.08], [2.3, 2.3], color=RED, lw=0.9)
outcome(ax, 'invariant but weaker evidence:\nlower base accuracy to protect', '#444444')

# (d) AnchorCE: aligned evidence and invariant base scores
ax = axes[3]
panel(ax, '(d) AnchorCE (ours)', 'Base-anchored alignment', readout_color=RED)
bars(ax, X0, B0)
bars(ax, X1, B0, new=[0.50, 0.30])
padlock(ax, X1 + 0.3, 2.2)
ax.text(X1 + 0.5, 2.22, 'invariant, $G=0$', fontsize=5.8, color=BLUE, ha='left', va='center')
ax.text(0.34, 2.48, 'classifier + text + prototypes + covariance', fontsize=5.3, color=INK, ha='left', va='center')
outcome(ax, 'aligned evidence, invariant base scores:\ncertified ' + r'$A_B\geq r\,A_{\mathrm{ref}}$', GREEN)

fig.savefig(str(OUT) + '.pdf', bbox_inches='tight', pad_inches=0.01)
fig.savefig(str(OUT) + '.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
print('saved', OUT)
