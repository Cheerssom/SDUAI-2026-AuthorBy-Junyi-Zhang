"""Split results figures for main_34 (each one row, sitting next to a table):
fig_results_cd: (a) accuracy vs certification, (b) certified accuracy across encoders  -> beside Tables 2-3
fig_sessions:   (a) miniImageNet, (b) CIFAR-100 session-wise top-1                     -> beside Table 4
Reuses the panel functions and palette of make_fig_results.py."""
from pathlib import Path
src = (Path(__file__).with_name('make_fig_results.py')).read_text()
exec(src[:src.index('fig, axes = plt.subplots')])
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 6.8, 'axes.titlesize': 7.4, 'xtick.labelsize': 6, 'ytick.labelsize': 6})

fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.85), gridspec_kw=dict(wspace=0.45))
tradeoff(axes[0]); encoders(axes[1])
keys = ['BiMC_transfer', 'FeCAM_transfer', 'GDA_transfer', OURS]
axes[1].legend(handles=[Patch(facecolor=COLOR[k], label=LABEL[k].replace(' (Ours)', '')) for k in keys], frameon=False,
               fontsize=5.2, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, handlelength=0.8, columnspacing=0.5, handletextpad=0.3)
axes[1].set_ylim(0, 100)
for ax, t in zip(axes, 'ab'):
    ax.text(0.5, -0.30, f'({t})', transform=ax.transAxes, ha='center', va='top', fontsize=7)
fig.savefig(str(HERE.parent / 'fig_results_cd.pdf'), bbox_inches='tight', pad_inches=0.02)
fig.savefig(str(HERE.parent / 'fig_results_cd.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)

fig, axes = plt.subplots(1, 2, figsize=(4.75, 0.98), gridspec_kw=dict(wspace=0.22))
sessions(axes[0], 'mini', ''); sessions(axes[1], 'cifar100', '')
for ax, t in zip(axes, ['(a) miniImageNet', '(b) CIFAR-100']):
    ax.set_title(''); ax.text(0.03, 0.06, t, transform=ax.transAxes, fontsize=7, fontweight='bold', ha='left', va='bottom', color=INK)
for ax in axes:
    ax.set_xlabel('Incremental session', labelpad=3); ax.set_ylabel('Top-1 (%)', labelpad=1); ax.tick_params(pad=1)
axes[0].yaxis.set_major_locator(MultipleLocator(5)); axes[1].yaxis.set_major_locator(MultipleLocator(10))
for ax, t in zip(axes, ''):
    ax.text(0.5, -0.30, f'({t})', transform=ax.transAxes, ha='center', va='top', fontsize=7)
lk = ['ZeroShot_CLIP', 'GDA_transfer', 'FeCAM_transfer', 'dense', 'BiMC_transfer', OURS]
h = [Line2D([0], [0], color=COLOR[k], marker=MARK[k], markersize=5 if k == OURS else 3.5, linewidth=1.6 if k == OURS else 0.9,
            markeredgecolor='white', markeredgewidth=0.4) for k in lk]
fig.legend(h, [LABEL[k] for k in lk], loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=6, frameon=False,
           handlelength=1.4, columnspacing=0.8, fontsize=6)
fig.savefig(str(HERE.parent / 'fig_sessions.pdf'), bbox_inches='tight', pad_inches=0.02)
fig.savefig(str(HERE.parent / 'fig_sessions.png'), dpi=300, bbox_inches='tight', pad_inches=0.02)
print('saved fig_results_cd / fig_sessions')
