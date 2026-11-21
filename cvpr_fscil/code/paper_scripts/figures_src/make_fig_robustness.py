"""Robustness figure for main_33 (fig:robustness): the gain comes from per-query alignment.

(a, b) Certified new-class gain of AnchorCE over the uncentered and the fixed-moment readouts on
       SUN397 under four reference predictors (paired 95% intervals; Supplement R).
(c) ImageNet-R capacity lost when query moments are permuted across queries (three seeds;
    main Table prospective b).
(d) Oracle capacity across bank sizes for interpolation constants C in {10,20,30,50}, ViT-L/14
    (Supplement L.1, tab:constantsensitivity).

Inputs: experiments/expansion_independent_reference_replay/summary.json,
        experiments/expansion_dense_reference_replay/summary.json; (c) and (d) values are the
        published table entries reproduced below.
Output: ../fig_robustness.pdf / .png
"""
from pathlib import Path
import json
from figstyle import plt, style, tag, save, RED, INK, INK2, SEQ_BLUE

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / 'experiments'
IND = json.loads((EXP / 'expansion_independent_reference_replay' / 'summary.json').read_text())['contrasts']
DEN = json.loads((EXP / 'expansion_dense_reference_replay' / 'summary.json').read_text())['contrasts']
LIGHT = '#f1a3a2'   # lighter tint of the ours-red for the second encoder


def contrast(enc, ref, comp):
    pool = DEN if ref == 'common_dense' else IND
    c = next(c for c in pool if c['encoder'] == enc and c['reference'] == ref and c['contrast'] == f'full - {comp}')
    return c['delta'], c['ci']


REFS = [('common_full', 'Own'), ('common_dense', 'Dense'), ('zeroshot', 'Zero-shot'), ('probe', 'Probe')]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85), gridspec_kw=dict(wspace=0.45))
for ax, comp, title in [(axes[0], 'none', 'Gain over no centering'), (axes[1], 'fixed', 'Gain over fixed moments')]:
    w = 0.36
    for j, (enc, col, lab) in enumerate([('l14', RED, 'L/14'), ('b32', LIGHT, 'B/32')]):
        xs = [i + (j - 0.5) * (w + 0.04) for i in range(len(REFS))]
        vals = [contrast(enc, r, comp) for r, _ in REFS]
        ax.bar(xs, [v for v, _ in vals], width=w, color=col, edgecolor='white', linewidth=0.6, zorder=3, label=lab)
        ax.errorbar(xs, [v for v, _ in vals], yerr=[[v - ci[0] for v, ci in vals], [ci[1] - v for v, ci in vals]],
                    fmt='none', ecolor=INK2, elinewidth=0.7, capsize=1.6, zorder=4)
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks(range(len(REFS)))
    ax.set_xticklabels([r[1] for r in REFS], fontsize=6, rotation=25, ha='right', rotation_mode='anchor')
    ax.set_xlabel('Reference predictor')
    ax.set_ylabel('Certified gain (pts)')
    ax.set_title(title, color=INK)
    ax.set_ylim(0, 10.5)
    ax.legend(frameon=False, fontsize=5.8, loc='upper right', handlelength=1.0)
    style(ax, xgrid=False)

# (c) permutation drop, values from main Table (b): seed, L/14 [lo,hi], B/32 [lo,hi]
PERM = [('3101', 4.96, (3.90, 6.19), 7.56, (6.13, 8.96)), ('3102', 4.38, (3.17, 5.71), 8.10, (6.27, 9.92)),
        ('3103', 4.64, (3.73, 5.60), 7.52, (5.53, 9.64))]
ax = axes[2]
w = 0.36
for j, (col, lab, vi, ci) in enumerate([(RED, 'L/14', 1, 2), (LIGHT, 'B/32', 3, 4)]):
    xs = [i + (j - 0.5) * (w + 0.04) for i in range(len(PERM))]
    v = [p[vi] for p in PERM]
    ax.bar(xs, v, width=w, color=col, edgecolor='white', linewidth=0.6, zorder=3, label=lab)
    ax.errorbar(xs, v, yerr=[[x - p[ci][0] for x, p in zip(v, PERM)], [p[ci][1] - x for x, p in zip(v, PERM)]],
                fmt='none', ecolor=INK2, elinewidth=0.7, capsize=1.6, zorder=4)
ax.set_xticks(range(len(PERM)))
ax.set_xticklabels([p[0] for p in PERM], fontsize=5.8)
ax.set_xlabel('Permutation seed')
ax.set_ylabel('Capacity lost (pts)')
ax.set_title('Breaking query correspondence', color=INK)
ax.set_ylim(0, 11)
ax.legend(frameon=False, fontsize=5.8, loc='upper left', handlelength=1.0, ncol=2)
style(ax, xgrid=False)

# (d) interpolation-constant stability, L/14 (tab:constantsensitivity)
B = [5, 10, 15, 20, 30, 50, 100]
CAP = {10: [90.41, 90.92, 90.43, 88.66, 88.76, 89.51, 88.54], 20: [90.37, 90.54, 90.13, 90.18, 88.90, 89.51, 88.54],
       30: [90.34, 89.97, 89.92, 90.07, 89.63, 89.50, 88.54], 50: [90.32, 89.82, 89.82, 89.76, 90.31, 89.49, 88.54]}
ax = axes[3]
for i, C in enumerate([10, 20, 50]):
    ax.plot(B, CAP[C], color=SEQ_BLUE[i + 1], marker='o', markersize=3, linewidth=1.0, markeredgecolor='white',
            markeredgewidth=0.4, label=f'C = {C}')
ax.plot(B, CAP[30], color=RED, marker='*', markersize=6, linewidth=1.8, markeredgecolor='white', markeredgewidth=0.4,
        label='C = 30 (default)', zorder=5)
ax.set_xscale('log')
ax.set_xticks([5, 10, 20, 50, 100])
ax.set_xticklabels(['5', '10', '20', '50', '100'], fontsize=5.8)
ax.minorticks_off()
ax.set_ylim(80, 95)
ax.set_yticks([80, 85, 90, 95])
ax.set_xlabel('Base classes in the bank')
ax.set_ylabel('Oracle capacity (%)')
ax.set_title('Interpolation constant', color=INK)
ax.legend(frameon=False, fontsize=5.6, loc='lower left', handlelength=1.4, ncol=2, columnspacing=0.8)
style(ax)

tag(axes, y=-0.36)
save(fig, HERE.parent / 'fig_robustness')
