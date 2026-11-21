"""Shared style for the paper's data figures (CVPR two-column, Times-like serif)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Validated categorical palette (dataviz reference instance), fixed entity order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
# Sequential single-hue ramp (light -> dark) for magnitudes such as the recovery rate.
SEQ_BLUE = ['#b7d3f2', '#86b4e8', '#5591d9', '#2a6cc0', '#1b4a8a']

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.6,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.labelcolor': INK, 'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def style(ax, xgrid=True):
    ax.grid(True, color=GRID, linewidth=0.5)
    if not xgrid:
        ax.grid(False, axis='x')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def tag(axes, y=-0.33):
    for ax, t in zip(axes, 'abcdefgh'):
        ax.text(0.5, y, f'({t})', transform=ax.transAxes, ha='center', va='top', fontsize=7.5)


def save(fig, out):
    fig.savefig(str(out) + '.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(str(out) + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    print('saved', str(out) + '.pdf')
