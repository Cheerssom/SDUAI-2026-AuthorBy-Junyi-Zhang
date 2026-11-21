"""Draft layout for Fig. 1 (overview of AnchorCE), addressing the advisor's comments.

Layout follows HyCal (session strip + fit-once / per-session blocks, explicit output),
BiMC (colored regions with titles per evidence source) and QKD (in-figure legend).
This is a layout draft for the author to refine; it does not replace framework_v9.pdf.
Output: ../fig1_draft.pdf / .png
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
HERE = Path(__file__).resolve().parent
BASE, NEW, INK, GREY = '#e07a2e', '#2a78d6', '#1d1d1b', '#6b6a66'
FROZ = '#2f8fd8'
REG = {'L': '#fbe3cf', 'G': '#e9e1f5', 'T': '#e2efd9', 'P': '#dcebf7'}
EDGE = {'L': '#d9935c', 'G': '#9a7cc9', 'T': '#7fae62', 'P': '#6a9fd0'}

fig = plt.figure(figsize=(14.0, 5.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(-3, 57)
ax.axis('off')


def box(x0, y0, x1, y1, text='', fc='white', ec=GREY, lw=1.0, fs=9, ls='-', r=0.8, bold=False, color=INK, italic=False, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f'round,pad=0,rounding_size={r}',
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', fontsize=fs, color=color,
                fontweight='bold' if bold else 'normal', fontstyle='italic' if italic else 'normal', zorder=z + 1)


def arrow(p, q, color=BASE, lw=1.3, rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', zorder=4, shrinkA=0, shrinkB=0))


def flake(x, y, s=10):
    ax.text(x, y, '❄', fontsize=s, color=FROZ, ha='center', va='center', family='DejaVu Sans', zorder=6)


def poly(xs, ys, color=BASE, lw=1.3):
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=4, solid_capstyle='round')
    arrow((xs[-2], ys[-2]), (xs[-1], ys[-1]), color=color, lw=lw)


# ---------------- session strip ----------------
ax.text(1.5, 53.5, 'Sessions', fontsize=9.5, fontweight='bold', va='center')
sess = [('Session 0: base classes $\\mathcal{B}$', BASE, 12, 40), ('Session 1: new $\\mathcal{N}_1$', NEW, 44, 66),
        ('Session $t$: new $\\mathcal{N}_t$', NEW, 76, 98), ('Session $T$', NEW, 108, 124)]
for i, (t, c, a, b) in enumerate(sess):
    box(a, 51.2, b, 55.8, t, fc='white', ec=c, lw=1.2, fs=8.5)
    if i:
        prev = sess[i - 1][3]
        arrow((prev + 0.6, 53.5), (a - 0.6, 53.5), color=GREY, lw=1.0)
ax.text(70.6, 53.5, '…', fontsize=11, ha='center', va='center', color=GREY)
ax.text(103, 53.5, '…', fontsize=11, ha='center', va='center', color=GREY)
ax.text(126, 53.5, 'every upgrade must certify base retention', fontsize=8.5, fontstyle='italic', va='center', color=GREY)

# ---------------- panel frames ----------------
box(1, 1.5, 101, 49, fc='none', ec='#c9c7bf', lw=1.0, ls=(0, (4, 3)), r=1.5, z=1)
box(103, 1.5, 159, 49, fc='none', ec='#c9c7bf', lw=1.0, ls=(0, (4, 3)), r=1.5, z=1)

# (a) Session 0 block: fit once
box(2.5, 27, 28, 47.5, fc='#f6f5f1', ec='#d8d6cd', lw=0.8, r=1.0, z=1)
ax.text(3.5, 46, 'Session 0: fit once', fontsize=8.5, fontweight='bold', va='center', color=BASE)
box(3.5, 30, 12.5, 43, 'Base\ntraining set\n(features\n+ labels)', fs=8)
box(16, 38.7, 26.8, 44.4, 'Ridge classifier\n$W_{\\mathcal{B}},\\,b_{\\mathcal{B}}$', fc=REG['L'], ec=EDGE['L'], fs=8)
box(16, 29.2, 26.8, 34.9, 'Shared\ncovariance $\\Sigma$', fc=REG['G'], ec=EDGE['G'], fs=8)
flake(26.4, 43.8, 8)
flake(26.4, 34.3, 8)
arrow((12.5, 39.5), (16, 41.5), color=GREY, lw=1.0)
arrow((12.5, 33.5), (16, 32.0), color=GREY, lw=1.0)
ax.text(13.0, 41.6, 'fit', fontsize=7, color=GREY, fontstyle='italic')
ax.text(14.2, 30.5, 'estimate', fontsize=6.6, ha='center', color=GREY, fontstyle='italic')

# (a) Session t block: inputs and encoders
box(2.5, 3, 28, 25.5, fc='#f6f5f1', ec='#d8d6cd', lw=0.8, r=1.0, z=1)
ax.text(3.5, 24, 'Session $t$: score each query', fontsize=8.5, fontweight='bold', va='center', color=NEW)
rows = [(18.5, 'Query image $x$', 'Visual\nencoder'), (11.5, 'Class names\n"a photo of a {c}"', 'Text\nencoder'),
        (4.8, '$k$-shot images\nof new classes', 'Visual\nencoder')]
for y, inp, enc in rows:
    box(3.5, y - 0.2, 13.5, y + 4.6, inp, fs=7.6)
    box(16, y - 0.2, 25.5, y + 4.6, enc, fc='#ecebe6', ec='#a9a79f', fs=7.8)
    flake(25.1, y + 4.0, 8)
    arrow((13.5, y + 2.2), (16, y + 2.2), color=GREY, lw=1.0)

# ---------------- four evidence sources (colored regions) ----------------
src = [('L', 39.5, 'Retained classifier', '$q_L(x,b)$'), ('G', 31.5, 'Covariance', '$q_G(x,c)$'),
       ('T', 23.5, 'Text anchors', '$q_T(x,c)$'), ('P', 15.5, 'Few-shot prototypes', '$q_P(x,n)$')]
for k, y, title, q in src:
    box(31, y, 51.5, y + 6.6, fc=REG[k], ec=EDGE[k], lw=1.1, r=0.9)
    ax.text(32, y + 5.2, title, fontsize=8.2, fontweight='bold', va='center')
    ax.text(50.8, y + 5.1, q, fontsize=10, ha='right', va='center')
# token rows: base (orange) / new (blue)
for k, y in [('T', 23.5), ('P', 15.5)]:
    for i in range(4):
        ax.add_patch(Rectangle((32.2 + 1.7 * i, y + 1.1), 1.3, 2.2, fc=BASE, ec='white', lw=0.5, zorder=5))
    for i in range(2):
        ax.add_patch(Rectangle((39.6 + 1.7 * i, y + 1.1), 1.3, 2.2, fc=NEW, ec='white', lw=0.5, zorder=5))
for (y, lb, ln) in [(23.5, '$t_b$', '$t_n$'), (15.5, '$p_b$', '$p_n$')]:
    ax.text(43.6, y + 2.2, lb, fontsize=7.6, color=BASE, va='center')
    ax.text(45.6, y + 2.2, '/', fontsize=7.6, color=GREY, va='center')
    ax.text(46.4, y + 2.2, ln, fontsize=7.6, color=NEW, va='center')
for i in range(4):
    ax.add_patch(Rectangle((32.2 + 1.7 * i, 40.6), 1.3, 2.2, fc=BASE, ec='white', lw=0.5, zorder=5))
ax.text(39.2, 41.7, '$w_b$', fontsize=7.6, color=BASE, va='center')
ax.text(32, 33.0, '$v^\\top\\Sigma^{-1}m_c-\\frac{1}{2}m_c^\\top\\Sigma^{-1}m_c$', fontsize=7.6, va='center')


# connections into sources
arrow((26.8, 41.5), (31, 41.5), color=EDGE['L'], lw=1.4)
arrow((26.8, 32.0), (31, 33.5), color=EDGE['G'], lw=1.4)
arrow((25.5, 13.7), (31, 26.0), color=EDGE['T'], lw=1.2, rad=0.1)
arrow((25.5, 7.0), (31, 17.6), color=EDGE['P'], lw=1.2, rad=0.15)
# query feature v bus
ax.plot([25.5, 28.6], [20.7, 20.7], color=INK, lw=1.0, zorder=4)
ax.plot([28.6, 28.6], [18.6, 46.0], color=INK, lw=1.0, zorder=4)
for y in (44.5, 36.6, 28.6, 18.6):
    ax.plot([28.6, 31], [y, y], color=INK, lw=1.0, zorder=4)
    ax.add_patch(plt.Circle((28.6, y), 0.35, color=INK, zorder=5))
ax.text(29.0, 47.2, 'query feature $v=f(x)$', fontsize=7.4, va='center')

# ---------------- alignment, margin, output ----------------
box(55, 21, 77, 46.4, fc='#eef4ea', ec='#5f8f4a', lw=1.3, r=1.0)
ax.text(66, 44.3, 'Base-anchored alignment', fontsize=9, fontweight='bold', ha='center', va='center')
box(56.5, 35.2, 75.5, 41.6, 'fixed base reference banks\n$\\mathcal{R}_e$: $W_{\\mathcal{B}},\\,T_{\\mathcal{B}},\\,P_{\\mathcal{B}},\\,G_{\\mathcal{B}}$', fc='white', ec='#5f8f4a', fs=7.8)
flake(75.0, 41.0, 8)
ax.text(66, 30.2, '$\\hat q_e(x,c)=\\dfrac{q_e(x,c)-\\mu_e(x)}{s_e(x)}$', fontsize=10, ha='center', va='center')
ax.text(66, 24.0, 'arriving classes never change\na base score  (Eq. 4)', fontsize=7.4, ha='center', va='center', color=GREY, fontstyle='italic')
for y in (42.0, 34.6, 26.8, 18.8):
    arrow((51.5, y), (55, min(max(y, 23), 44)), color=GREY, lw=1.0)
box(55, 6, 77, 16.5, 'Closed-form admission\n$M(x)+d>0$\n(Eq. 9)', fc='#fdf3e2', ec='#c98a2a', lw=1.3, fs=8.6)
arrow((66, 21), (66, 16.5), color=INK, lw=1.3)
box(81, 6, 99.5, 16.5, 'Prediction\nbest base class  |  new class', fc='white', ec=INK, lw=1.5, fs=8.6)
arrow((77, 11.2), (81, 11.2), color=INK, lw=1.3)

# legend (QKD-style)
lx, ly = 81.5, 41.5
ax.text(lx, ly + 4.4, 'Legend', fontsize=8.2, fontweight='bold')
ax.add_patch(Rectangle((lx, ly + 1.4), 1.6, 1.6, fc=BASE, ec='none'))
ax.text(lx + 2.4, ly + 2.2, 'base classes', fontsize=7.8, va='center')
ax.add_patch(Rectangle((lx, ly - 1.6), 1.6, 1.6, fc=NEW, ec='none'))
ax.text(lx + 2.4, ly - 0.8, 'new classes', fontsize=7.8, va='center')
flake(lx + 0.8, ly - 3.9, 9)
ax.text(lx + 2.4, ly - 3.9, 'frozen', fontsize=7.8, va='center')
ax.text(lx, ly - 7.6, '$q_e(x,c)$: score of query $x$ for\nclass $c$ from source $e$;\n$b$: base class, $n$: new class', fontsize=7.4,
        va='top', color=GREY, linespacing=1.3)

# ---------------- (b) certifying the upgrade ----------------
poly([99.5, 101.8, 101.8, 105], [11.2, 11.2, 43.2, 43.2], color=INK, lw=1.3)
box(105, 40, 157.5, 46.6, 'Compare deployed vs. upgraded predictions\non base calibration images', fc='white', ec=INK, lw=1.1, fs=8.6)
box(105, 27.2, 157.5, 37.6, fc='#f7f1f6', ec='#b5839f', lw=1.1)
ax.text(106.3, 35.4, '✗', fontsize=11, color='#c43c3c', va='center', family='DejaVu Sans')
ax.text(108.8, 35.4, 'correct $\\rightarrow$ wrong: lost prediction $D$, the only cost to retention', fontsize=8.1, va='center')
ax.text(106.3, 31.0, '∅', fontsize=11, color='#2f7d32', va='center', family='DejaVu Sans')
ax.text(108.8, 31.0, 'wrong $\\rightarrow$ correct: impossible, recovery $G=0$', fontsize=8.1, va='center')
ax.text(108.8, 28.6, 'because base scores never change (Cor. 1)', fontsize=7.4, va='center', color=GREY, fontstyle='italic')
arrow((131, 40), (131, 37.6), color=INK, lw=1.2)
box(105, 18.5, 129.5, 24.8, 'Max. growth of the\nbetting test (Prop. 3)', fc='#fbe7e7', ec='#c46262', lw=1.1, fs=8.3)
box(133, 18.5, 157.5, 24.8, 'Certified offset $\\hat d$\n(Eq. 11)', fc='#e5f0fb', ec='#4f8fd0', lw=1.1, fs=8.3)
arrow((117, 27.2), (117, 24.8), color=INK, lw=1.2)
arrow((129.5, 21.6), (133, 21.6), color=INK, lw=1.2)
# evidence inset
ax.text(131, 14.6, 'Evidence: fixed-accuracy study (acceptance, 40 images/class)', fontsize=7.6, ha='center',
        va='center', fontstyle='italic', color=GREY)
ins = fig.add_axes([0.705, 0.10, 0.25, 0.11])
ins.barh([1, 0], [2.8, 90.6], color=['#b9b7af', '#c0392b'], height=0.6)
ins.set_yticks([1, 0])
ins.set_yticklabels(['$G>0$ (max.)', '$G=0$'], fontsize=8)
ins.set_xlim(0, 108)
ins.set_xticks([])
for s in ('top', 'right', 'bottom'):
    ins.spines[s].set_visible(False)
ins.text(2.8 + 1.5, 1, '2.8%', va='center', fontsize=8)
ins.text(90.6 + 1.5, 0, '90.6%', va='center', fontsize=8, fontweight='bold', color='#c0392b')

# panel captions
ax.text(51, -1.2, '(a) Scoring an upgrade: base-anchored evidence alignment', fontsize=10, ha='center',
        fontstyle='italic', fontweight='bold', color='#a8322d')
ax.text(131, -1.2, '(b) Certifying the upgrade: certifiability by design', fontsize=10, ha='center',
        fontstyle='italic', fontweight='bold', color='#a8322d')

out = HERE.parent / 'fig1_draft'
fig.savefig(str(out) + '.pdf', bbox_inches='tight', pad_inches=0.03)
fig.savefig(str(out) + '.png', dpi=200, bbox_inches='tight', pad_inches=0.03)
print('saved', out)
