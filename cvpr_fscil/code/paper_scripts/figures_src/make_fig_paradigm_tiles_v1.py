"""Paradigm overview (problem-setting figure) in the visual style of HyCal's Figure 1.

(a) CIL, (b) FSCIL, (c) training-free FSCIL on frozen CLIP, (d) retention-certified FSCIL (ours).
Isometric tile stacks are sessions (stack height ~ labeled images per class); bold boxed labels;
light-gray canvas with a vertical session arrow; red marks what retention-certified FSCIL adds.
Output: ../fig_paradigm.pdf / .png
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / 'fig_paradigm'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
INK, GRAYBG, ARROW, EDGE = '#111111', '#f1f1f1', '#8c8c8c', '#6b5a35'
BASE, BASE_D = '#f6c453', '#d39b2c'      # current-session base tiles
PAST, PAST_D = '#f8eed6', '#dccfae'      # earlier sessions (faded)
NEW, NEW_D = '#a6d8a0', '#62a35c'        # new classes, standard settings
RNEW, RNEW_D = '#f5a29c', '#d0625a'      # new classes admitted under certification
RED, GREEN, BLUE = '#d62728', '#1a7f37', '#1d4e89'
DX, DY, W, H = 0.24, 0.16, 0.48, 0.065   # tile depth, width and per-layer height


def tile(ax, x, y, top=BASE, side=BASE_D, layers=1, icon=None, icol='#e07b00', z=3):
    """Isometric stack of `layers` tiles; (x, y) is the top-left front corner of the top face."""
    h = H * layers
    ax.add_patch(Polygon([(x, y - h), (x + W, y - h), (x + W, y), (x, y)], fc=side, ec=EDGE, lw=0.35, zorder=z))
    ax.add_patch(Polygon([(x + W, y - h), (x + W + DX, y - h + DY), (x + W + DX, y + DY), (x + W, y)], fc=side, ec=EDGE,
                         lw=0.35, zorder=z))
    for k in range(1, layers):
        ax.plot([x, x + W, x + W + DX], [y - k * H, y - k * H, y - k * H + DY], color=EDGE, lw=0.22, alpha=0.65, zorder=z)
    ax.add_patch(Polygon([(x, y), (x + W, y), (x + W + DX, y + DY), (x + DX, y + DY)], fc=top, ec=EDGE, lw=0.4, zorder=z + .1))
    if icon:
        ax.plot(x + W / 2 + DX / 2, y + DY / 2, marker=icon, ms=3.6, color=icol, mec='white', mew=0.3, zorder=z + .2)


def session(ax, x, y, n_past, new=None, layers_new=5, top=BASE, side=BASE_D, icol='#e07b00', past_layers=5):
    for i in range(n_past):
        tile(ax, x + i * (W + 0.03), y, top=PAST, side=PAST_D, layers=past_layers)
    if new:
        tile(ax, x + n_past * (W + 0.03), y, top=top, side=side, layers=layers_new, icon=new, icol=icol)


def label(ax, x, y, s):
    ax.text(x, y, s, fontsize=6.4, fontweight='bold', ha='left', va='center', color=INK, zorder=6,
            bbox=dict(boxstyle='square,pad=0.2', fc='white', ec=INK, lw=0.7))


def strip(ax, x, y, w):
    ax.add_patch(Rectangle((x, y - 0.16), w, 0.32, fc='#fbf2d7', ec='#c2b285', lw=0.4, zorder=2))
    items = ['o', '...', 's', '...', '^', '...', 'D', '...', 'v']
    step = w / (len(items) + 1)
    for i, m in enumerate(items):
        if m == '...':
            ax.text(x + step * (i + 1), y, r'$\cdots$', fontsize=5.5, ha='center', va='center', color=INK, zorder=3)
        else:
            ax.plot(x + step * (i + 1), y, marker=m, ms=4.0, color='#e07b00', mec='white', mew=0.3, zorder=3)


def arrow(ax, p0, p1, txt=None, color=INK, rad=-0.45, tx=None, ty=None, lw=0.8, ls='-', fs=6.2, ha='left'):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=6.5,
                                 color=color, lw=lw, linestyle=ls, zorder=7))
    if txt:
        ax.text(tx, ty, txt, fontsize=fs, color=color, ha=ha, va='center', zorder=8)


def panel(ax, title, train):
    ax.set_xlim(0, 5.0); ax.set_ylim(0, 4.3); ax.axis('off')
    ax.text(2.5, 4.16, title, fontsize=7.4, fontweight='bold', ha='center', va='center', color=INK)
    label(ax, 0.2, 3.7, 'Data')
    strip(ax, 0.9, 3.7, 3.95)
    label(ax, 0.2, 3.24, train)
    x0, y0, w, h = 0.2, 0.12, 4.65, 2.8
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=GRAYBG, ec='none', zorder=1))
    ax.annotate('', xy=(x0 + 0.17, y0 + 0.1), xytext=(x0 + 0.17, y0 + h - 0.12),
                arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=0.9, mutation_scale=7), zorder=2)
    ax.text(x0 + 0.06, y0 + h * 0.42, 'Session', rotation=90, fontsize=6.0, color=ARROW, fontweight='bold',
            ha='center', va='center', zorder=2, bbox=dict(fc=GRAYBG, ec='none', pad=0.6))


R0, R1, R2 = (0.62, 2.62), (1.42, 1.86), (2.22, 1.10)   # top-left corners of the three session rows

fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), gridspec_kw=dict(wspace=0.05))
plt.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)

# (a) CIL: every session brings many labeled images per class
for ax, title, few, train, newtop, newside, icol in [
        (axes[0], '(a) Class-incremental learning', False, 'Train phase', BASE, BASE_D, '#e07b00'),
        (axes[1], '(b) Few-shot CIL', True, 'Train phase', BASE, BASE_D, '#e07b00'),
        (axes[2], '(c) Training-free FSCIL', True, 'Frozen CLIP', NEW, NEW_D, '#2e6b28')]:
    panel(ax, title, train)
    ln = 1 if few else 5
    session(ax, *R0, 0); tile(ax, R0[0], R0[1], layers=5, icon='o'); tile(ax, R0[0] + W + 0.03, R0[1], layers=5, icon='s')
    session(ax, *R1, 2, new='^', layers_new=ln, top=newtop, side=newside, icol=icol)
    session(ax, *R2, 3, new='D', layers_new=ln, top=newtop, side=newside, icol=icol)
    lab = r'$+K\times|C|$' if few else r'$+\sum^{C}N$'
    arrow(ax, (R0[0] + 2 * W + DX + 0.12, R0[1] + 0.02), (R1[0] + 2 * (W + 0.03) + W / 2 + 0.08, R1[1] + DY + 0.08),
          lab, tx=R1[0] + 2 * (W + 0.03) + W / 2 + 0.22, ty=R0[1] - 0.05)
    arrow(ax, (R1[0] + 3 * (W + 0.03) + DX + 0.08, R1[1] - 0.02), (R2[0] + 3 * (W + 0.03) + W / 2 + 0.08, R2[1] + DY + 0.08),
          lab, tx=R2[0] + 3 * (W + 0.03) + W / 2 + 0.2, ty=R1[1] - 0.29)
axes[2].text(1.86, 3.24, '❄', fontsize=7.2, color='#3a7bd5', ha='left', va='center', family='DejaVu Sans')
axes[2].text(0.5, 0.3, 'evaluated by average accuracy', fontsize=6.0, style='italic', color='#444444', ha='left', va='center', zorder=8)

# (d) Retention-certified FSCIL (ours): each upgrade of the deployed model must pass a retention test
ax = axes[3]
panel(ax, '(d) Retention-certified FSCIL (ours)', 'Deployed model')
tile(ax, R0[0], R0[1], layers=5, icon='o'); tile(ax, R0[0] + W + 0.03, R0[1], layers=5, icon='s')
gx, gy, gw, gh = 2.12, 1.92, 2.62, 0.42
ax.add_patch(FancyBboxPatch((gx, gy - gh / 2), gw, gh, boxstyle='round,pad=0.02,rounding_size=0.06', fc='white', ec=RED,
                            lw=0.9, zorder=8))
ax.text(gx + gw / 2, gy, r'Retention test: $A_B\geq r\,A_{\mathrm{ref}}$', fontsize=5.9, color=RED, ha='center',
        va='center', zorder=9)
arrow(ax, (R0[0] + 2 * W + DX + 0.1, R0[1] + 0.02), (gx + 0.45, gy + gh / 2 + 0.03), r'$+K\times|C|$', color=RED, rad=-0.45,
      tx=gx + 0.62, ty=R0[1] - 0.02, lw=1.0)
session(ax, 2.62, 1.12, 2, new='^', layers_new=1, top=RNEW, side=RNEW_D, icol='#9b1c1c', past_layers=5)
arrow(ax, (gx + gw - 0.25, gy - gh / 2 - 0.02), (2.62 + 2 * (W + 0.03) + W / 2 + 0.12, 1.12 + DY + 0.08), 'pass: admit',
      color=GREEN, rad=-0.4, tx=gx + gw - 0.32, ty=1.38, lw=1.3, fs=5.9, ha='right')
arrow(ax, (gx + 0.25, gy - gh / 2 - 0.02), (1.25, 1.42), color=INK, rad=0.3, lw=0.7, ls=(0, (2, 1.5)))
ax.text(0.5, 1.12, 'fail: keep the\ndeployed model', fontsize=5.9, color=INK, ha='left', va='center', zorder=9,
        linespacing=1.05)
ax.text(0.5, 0.3, r'base accuracy certified w.p. $\geq 1-\delta$', fontsize=6.0, style='italic', color=RED, ha='left',
        va='center', zorder=8)

fig.savefig(str(OUT) + '.pdf', bbox_inches='tight', pad_inches=0.01)
fig.savefig(str(OUT) + '.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
print('saved', OUT)
