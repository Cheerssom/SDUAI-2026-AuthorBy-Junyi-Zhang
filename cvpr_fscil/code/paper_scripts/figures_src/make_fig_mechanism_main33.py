"""Mechanism figure for main_33 (fig:mechanism): recovery governs certification.

(a) Retention-test acceptance vs calibration images per class, one curve per recovery fraction
    (fixed class-wise accuracies, CIFAR-100 L/14 profile; Supplement M.2 study).
(b) Acceptance with 40 images per class vs recovery rate G for all six dataset-encoder profiles.
(c) Exact variance of the retention statistic vs recovery rate G (Cor. 1 / Eq. 13: linear in G).
(d) Alignment ablation, ViT-B/16, averaged over CIFAR-100, miniImageNet and CUB-200.

Inputs: experiments/expansion_gate_calibration/coupling/*.json and
        experiments/autodl_fecam_replay2_20261108/summary.json
Output: ../fig_mechanism.pdf / .png
"""
from pathlib import Path
import json
from figstyle import plt, style, tag, save, RED, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, INK, INK2, SEQ_BLUE

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / 'experiments'
COUP = EXP / 'expansion_gate_calibration' / 'coupling'
SUMMARY = EXP / 'autodl_fecam_replay2_20261108' / 'summary.json'


def load(name):
    return json.loads((COUP / f'{name}.json').read_text())['records']


fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85), gridspec_kw=dict(wspace=0.5))

# (a) acceptance vs calibration size, CIFAR L/14, per recovery fraction
ax = axes[0]
recs = load('l14_cifar')
fracs = sorted({r['recovery_fraction'] for r in recs})
for i, f in enumerate(fracs):
    rows = sorted([r for r in recs if r['recovery_fraction'] == f], key=lambda r: r['cap'])
    zero = f == 0
    ax.plot([r['cap'] for r in rows], [100 * r['acceptance'] for r in rows],
            color=RED if zero else SEQ_BLUE[i], marker='*' if zero else 'o',
            markersize=6 if zero else 3.2, linewidth=1.8 if zero else 1.0,
            markeredgecolor='white', markeredgewidth=0.4, zorder=5 if zero else 3,
            label='zero recovery' if zero else f'{int(100 * f)}% of max')
ax.set_xscale('log', base=2)
ax.set_xticks([5, 10, 20, 40])
ax.set_xticklabels(['5', '10', '20', '40'])
ax.set_xlabel('Calibration images per class')
ax.set_ylabel('Test acceptance (%)')
ax.set_title('Zero recovery certifies', color=INK)
ax.set_ylim(-3, 103)
ax.legend(frameon=False, fontsize=5.6, loc='upper left', handlelength=1.4, labelspacing=0.25)
style(ax)

# (b) acceptance at cap 40 vs recovery rate, (c) variance vs recovery rate, six profiles
PROFILES = [('l14_cifar', 'CIFAR L/14', BLUE), ('b32_cifar', 'CIFAR B/32', ORANGE),
            ('l14_pets', 'Pets L/14', AQUA), ('b32_pets', 'Pets B/32', YELLOW),
            ('l14_imagenetr', 'IN-R L/14', MAGENTA), ('b32_imagenetr', 'IN-R B/32', GREEN)]
MARKS = ['o', 's', 'D', '^', 'v', 'P']
for (name, label, color), mk in zip(PROFILES, MARKS):
    rows = sorted([r for r in load(name) if r['cap'] == 40], key=lambda r: r['recovered'])
    g = [100 * r['recovered'] for r in rows]
    axes[1].plot(g, [100 * r['acceptance'] for r in rows], color=color, marker=mk, markersize=3.2,
                 linewidth=1.0, markeredgecolor='white', markeredgewidth=0.4, label=label)
    axes[2].plot(g, [1e4 * r['exact_mean_variance'] for r in rows], color=color, marker=mk,
                 markersize=3.2, linewidth=1.0, markeredgecolor='white', markeredgewidth=0.4)
ax = axes[1]
ax.set_xlabel('Recovery rate $G$ (%)')
ax.set_ylabel('Acceptance, 40 per class (%)')
ax.set_title('Acceptance vs. recovery', color=INK)
ax.set_ylim(-3, 103)
ax.legend(frameon=False, fontsize=5.4, loc='upper right', handlelength=1.3, labelspacing=0.2, ncol=1)
style(ax)
ax = axes[2]
ax.set_xlabel('Recovery rate $G$ (%)')
ax.set_ylabel(r'Variance ($\times10^{-4}$)')
ax.set_title('Variance vs. recovery', color=INK)
style(ax)

# (d) alignment ablation, B/16 average over three benchmarks
ax = axes[3]
avg = json.loads(SUMMARY.read_text())['average']
VAR = [('dense', 'Dense', MAGENTA), ('none', 'w/o cent.', '#b8b6ae'),
       ('fixed', 'Fixed', '#898781'), ('full', 'Ours', RED)]
w = 0.36
for i, (k, lab, c) in enumerate(VAR):
    a, ce = avg[f'b16/{k}']['Avg'], avg[f'b16/{k}']['certified_common']
    ax.bar(i - w / 2 - 0.02, a, width=w, color=c, edgecolor='white', linewidth=0.6, zorder=3)
    ax.bar(i + w / 2 + 0.02, ce, width=w, color=c, edgecolor='white', linewidth=0.6, alpha=0.55,
           hatch='////', zorder=3)
    if k == 'full':
        ax.text(i - w / 2 - 0.02, a + 1.5, f'{a:.1f}', ha='center', fontsize=5.8, fontweight='bold')
        ax.text(i + w / 2 + 0.02, ce + 1.5, f'{ce:.1f}', ha='center', fontsize=5.8, fontweight='bold')
ax.set_xticks(range(len(VAR)))
ax.set_xticklabels([v[1] for v in VAR], fontsize=6.2, rotation=25, ha='right', rotation_mode='anchor')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 118)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_title('Alignment ablation', color=INK)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#9a9a9a', label='Average acc.'),
                   Patch(facecolor='#d0d0d0', hatch='////', label='Certified acc.')],
          frameon=False, fontsize=5.6, loc='upper left', handlelength=1.2, ncol=2, columnspacing=0.8)
style(ax, xgrid=False)

tag(axes, y=-0.36)
save(fig, HERE.parent / 'fig_mechanism')
