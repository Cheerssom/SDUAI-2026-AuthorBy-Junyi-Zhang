"""Problem-setting figure for AnchorCE, drawn from scratch (no external artwork).

Top: data arrival (base session with many shots, then 5-shot sessions).
(a) Expansion and the alignment-invariance dilemma: the deployed predictor grows session by session;
    active-class normalization can flip base rankings, fixed base references keep them.
(b) Retention certification on held-out base images: A_{B,t} >= r A_ref with confidence >= 1 - delta.
(c) Deployment: certified -> admit new classes; otherwise fall back to the deployed predictor.
Output: ../fig_setting.pdf / .png
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Arc

OUT = Path(__file__).resolve().parent.parent / 'fig_setting'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
INK, MUTE, PANEL = '#111111', '#555555', '#f3f3f1'
C = {'base': ('#f6c453', '#d39b2c', '#fdf1cf', '#b5791a'),     # top, side, band, icon
     's1': ('#9ed39a', '#5fa35a', '#e5f3e2', '#2f7a2b'),
     's2': ('#8ec5ee', '#4a8fc9', '#e3effa', '#1f5f99')}
RED, GREEN, BLUE = '#c0392b', '#2e7d32', '#1d4e89'
DX0, DY0, W0, H0 = 0.20, 0.13, 0.36, 0.05
EDGE = '#5b4f3a'


def tile(ax, x, y, key, layers=1, icon=None, faded=False, z=3, k=1.0):
    W, H, DX, DY = W0 * k, H0 * k, DX0 * k, DY0 * k
    top, side, _, ic = C[key]
    if faded:
        top, side, ic = '#f7f2e6', '#ddd5c2', '#c9bfa8'
    h = H * layers
    ax.add_patch(Polygon([(x, y - h), (x + W, y - h), (x + W, y), (x, y)], fc=side, ec=EDGE, lw=0.3, zorder=z))
    ax.add_patch(Polygon([(x + W, y - h), (x + W + DX, y - h + DY), (x + W + DX, y + DY), (x + W, y)], fc=side, ec=EDGE,
                         lw=0.3, zorder=z))
    for j in range(1, layers):
        ax.plot([x, x + W, x + W + DX], [y - j * H, y - j * H, y - j * H + DY], color=EDGE, lw=0.2, alpha=0.6, zorder=z)
    ax.add_patch(Polygon([(x, y), (x + W, y), (x + W + DX, y + DY), (x + DX, y + DY)], fc=top, ec=EDGE, lw=0.35, zorder=z + .1))
    if icon:
        ax.plot(x + W / 2 + DX / 2, y + DY / 2, marker=icon, ms=3.4 * k, color=ic, mec='white', mew=0.3, zorder=z + .2)


def group(ax, x, y, items, z=3, k=1.0):
    """items: list of (key, icon, layers, faded) placed left to right."""
    for i, (key, icon, layers, faded) in enumerate(items):
        tile(ax, x + i * (W0 * k + 0.03), y, key, layers=layers, icon=icon, faded=faded, z=z, k=k)


def chip(ax, x, y, key, icon, r=0.13):
    _, _, band, ic = C[key]
    ax.add_patch(Circle((x, y), r, fc='white', ec='#333333', lw=0.6, zorder=5))
    ax.add_patch(Circle((x, y), r * 0.82, fc=band, ec='none', zorder=5))
    ax.plot(x, y, marker=icon, ms=4.2, color=ic, mec='white', mew=0.3, zorder=6)


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.0, ls='-', ms=7):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=7))


def minibars(ax, x, y, hs, colors, w=0.09, gap=0.03, scale=0.42):
    for i, (h, c) in enumerate(zip(hs, colors)):
        ax.add_patch(Rectangle((x + i * (w + gap), y), w, h * scale, fc=c, ec='none', zorder=6))
    ax.plot([x - 0.02, x + len(hs) * (w + gap)], [y, y], color=INK, lw=0.5, zorder=6)


def padlock(ax, x, y, s=0.055, color=BLUE):
    ax.add_patch(Rectangle((x - s, y - s), 2 * s, 1.6 * s, fc=color, ec=color, zorder=8))
    ax.add_patch(Arc((x, y + 0.6 * s), 1.3 * s, 2.2 * s, theta1=0, theta2=180, color=color, lw=0.9, zorder=8))


fig = plt.figure(figsize=(7.0, 2.55))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 5.1); ax.axis('off')

# ---- top: data arrival ------------------------------------------------------------------------
for x0, key, title, sub, n, icons in [(0.1, 'base', 'Base session', 'many shots per class', 7, 'osDoso^'),
                                       (4.8, 's1', 'Session 1', '5 shots per class', 5, 'v*v*v'),
                                       (9.5, 's2', 'Session 2', '5 shots per class', 5, 'PhPhP')]:
    _, _, band, _ = C[key]
    ax.add_patch(FancyBboxPatch((x0, 4.2), 4.4, 0.78, boxstyle='round,pad=0.02,rounding_size=0.14', fc=band, ec='none', zorder=1))
    ax.text(x0 + 0.2, 4.72, title, fontsize=8.2, fontweight='bold', ha='left', va='center', color=INK)
    ax.text(x0 + 0.2, 4.43, sub, fontsize=6.6, style='italic', ha='left', va='center', color=MUTE)
    for i in range(n):
        chip(ax, x0 + 2.05 + i * 0.3, 4.58, key, icons[i])
ax.plot([0.1, 13.9], [4.03, 4.03], color='#666666', lw=1.0, ls=(0, (5, 4)))

# ---- panels ---------------------------------------------------------------------------------
P = [(0.1, '(a) Expansion and the alignment dilemma'), (4.8, '(b) Retention certification'), (9.5, '(c) Deployment')]
for x0, t in P:
    ax.text(x0 + 2.2, 3.72, t, fontsize=8.2, fontweight='bold', ha='center', va='center', color=INK)
    ax.add_patch(FancyBboxPatch((x0, 0.08), 4.4, 3.38, boxstyle='round,pad=0.02,rounding_size=0.14', fc=PANEL, ec='none', zorder=1))

# (a) the deployed predictor grows; two ways to fuse evidence
group(ax, 0.42, 3.0, [('base', 'o', 5, False), ('base', 's', 5, False)])
ax.text(0.86, 2.5, 'Deployed predictor', fontsize=6.4, fontweight='bold', ha='center', va='center')
arrow(ax, (1.55, 2.9), (1.95, 2.9), color=RED)
group(ax, 2.0, 3.0, [('base', None, 3, True), ('s1', 'v', 1, False)])
ax.text(2.42, 2.5, 'Session 1', fontsize=6.4, fontweight='bold', ha='center', va='center')
arrow(ax, (3.05, 2.9), (3.4, 2.9), color=RED)
group(ax, 3.42, 3.0, [('base', None, 3, True), ('s2', 'P', 1, False)])
ax.text(3.82, 2.5, 'Session 2', fontsize=6.4, fontweight='bold', ha='center', va='center')
ax.plot([0.35, 4.25], [2.22, 2.22], color='#c8c8c8', lw=0.6)
# row 1: active-class normalization flips base rankings
minibars(ax, 0.45, 1.5, [0.9, 1.0, 0.5], ['#f6c453', '#f6c453', '#f6c453'])
arrow(ax, (0.83, 1.66), (1.05, 1.66), color=MUTE, lw=0.7, ms=5)
minibars(ax, 1.1, 1.5, [1.0, 0.8, 0.5], [RED, '#f6c453', '#f6c453'])
ax.text(1.55, 1.83, 'Active-class normalization', fontsize=6.5, fontweight='bold', ha='left', va='center')
ax.text(1.55, 1.58, 'base rankings can flip (recovery)', fontsize=6.3, color=RED, ha='left', va='center')
# row 2: fixed base references keep them
minibars(ax, 0.45, 0.62, [0.9, 1.0, 0.5], ['#f6c453'] * 3)
arrow(ax, (0.83, 0.78), (1.05, 0.78), color=MUTE, lw=0.7, ms=5)
minibars(ax, 1.1, 0.62, [0.9, 1.0, 0.5], ['#f6c453'] * 3)
padlock(ax, 1.26, 1.2)
ax.text(1.55, 0.95, 'Fixed base references (ours)', fontsize=6.5, fontweight='bold', ha='left', va='center')
ax.text(1.55, 0.70, 'scores aligned, rankings preserved', fontsize=6.3, color=BLUE, ha='left', va='center')

# (b) retention certification on held-out base images
ax.text(7.0, 3.2, 'Held-out base images', fontsize=7.0, fontweight='bold', ha='center', va='center')
for i, ic in enumerate('osDo'):
    chip(ax, 6.55 + i * 0.3, 2.82, 'base', ic, r=0.12)
arrow(ax, (7.0, 2.6), (7.0, 2.3), color='#4a7f6a', lw=0.9)
ax.add_patch(FancyBboxPatch((5.3, 0.55), 3.4, 1.65, boxstyle='round,pad=0.02,rounding_size=0.1', fc='white', ec='#a9b8b0',
                            lw=0.9, zorder=4))
ax.text(7.0, 1.72, r'$A_{B,t}\ \geq\ r\,A_{\mathrm{ref}}$', fontsize=11, ha='center', va='center', zorder=5)
ax.text(7.0, 1.2, r'with confidence $\geq 1-\delta$', fontsize=6.8, fontweight='bold', ha='center', va='center', zorder=5)
ax.text(7.0, 0.92, 'jointly over prespecified sessions', fontsize=6.3, style='italic', ha='center', va='center', color=MUTE, zorder=5)
ax.text(7.0, 0.32, 'zero recovery $\\Rightarrow$ fewest calibration images', fontsize=6.3, color=BLUE, ha='center', va='center')

# (c) deployment outcomes
group(ax, 9.75, 3.0, [('base', None, 3, True), ('s1', 'v', 1, False), ('s2', 'P', 1, False)], k=1.35)
ax.text(12.05, 2.98, 'Certified', fontsize=8.0, fontweight='bold', ha='left', va='center', color=GREEN)
ax.text(12.05, 2.68, 'admit new classes', fontsize=6.6, style='italic', ha='left', va='center')
ax.plot([9.8, 13.6], [2.1, 2.1], color='#c8c8c8', lw=0.6)
group(ax, 9.95, 1.45, [('base', 'o', 5, False), ('base', 's', 5, False)], k=1.35)
ax.text(12.05, 1.25, 'Fallback', fontsize=8.0, fontweight='bold', ha='left', va='center', color=INK)
ax.text(12.05, 0.95, 'keep deployed\npredictor', fontsize=6.6, style='italic', ha='left', va='center')
arrow(ax, (8.75, 1.6), (9.7, 2.75), color=GREEN, rad=-0.3, lw=1.2)
arrow(ax, (8.75, 1.15), (9.85, 1.1), color=MUTE, rad=0.2, lw=0.9, ls=(0, (2, 1.5)))
ax.text(9.0, 2.55, 'pass', fontsize=6.6, fontweight='bold', color=GREEN, ha='right', va='center')
ax.text(9.25, 0.82, 'fail', fontsize=6.6, fontweight='bold', color=MUTE, ha='center', va='center')

fig.savefig(str(OUT) + '.pdf', bbox_inches='tight', pad_inches=0.01)
fig.savefig(str(OUT) + '.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
print('saved', OUT)
