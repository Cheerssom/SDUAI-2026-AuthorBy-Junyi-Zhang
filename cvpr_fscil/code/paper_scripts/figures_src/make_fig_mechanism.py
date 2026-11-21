"""Mechanism figure for main_34 (fig:mechanism): recovery governs certification.

(a) Alignment ablation of the four-source method, ViT-B/16, averaged over CIFAR-100, miniImageNet and CUB-200.
(b) Real benchmarks: certified new-class accuracy (r=.95, B/16) vs calibration images per class, averaged over
    CIFAR-100 and miniImageNet (CUB-200 reserves 10 per class), from the --sweep runs.
(c) Retention-test acceptance vs calibration images per class, one curve per recovery fraction
    (fixed class-wise accuracies of CIFAR-100 with L/14; Supplement L study).
(d) Exact variance of the retention statistic vs recovery rate G (Cor. 1 / Eq. 13: linear in G, slope 2r/(Bn)).

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
SUMMARY = EXP / 'autodl_round4_20261109' / 'extracted' / 'work' / 'summary.json'
SWEEP = EXP / 'autodl_round5_20261110' / 'extracted' / 'work' / 'summary_sweep_ref4.json'  # deployed predictor = four-source closed head


def load(name):
    return json.loads((COUP / f'{name}.json').read_text())['records']


fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85), gridspec_kw=dict(wspace=0.5))

# (c) acceptance vs calibration size, CIFAR L/14, per recovery fraction
ax = axes[2]
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

# (d) variance vs recovery rate, six profiles
PROFILES = [('l14_cifar', 'CIFAR L/14', BLUE), ('b32_cifar', 'CIFAR B/32', ORANGE),
            ('l14_pets', 'Pets L/14', AQUA), ('b32_pets', 'Pets B/32', YELLOW),
            ('l14_imagenetr', 'IN-R L/14', MAGENTA), ('b32_imagenetr', 'IN-R B/32', GREEN)]
MARKS = ['o', 's', 'D', '^', 'v', 'P']
for (name, label, color), mk in zip(PROFILES, MARKS):
    rows = sorted([r for r in load(name) if r['cap'] == 40], key=lambda r: r['recovered'])
    g = [100 * r['recovered'] for r in rows]
    axes[3].plot(g, [1e4 * r['exact_mean_variance'] for r in rows], color=color, marker=mk,
                 markersize=3.2, linewidth=1.0, markeredgecolor='white', markeredgewidth=0.4, label=label)
ax = axes[3]
ax.set_xlabel('Recovery rate $G$ (%)')
ax.set_ylabel(r'Variance ($\times10^{-4}$)')
ax.set_title('Variance vs. recovery', color=INK)
ax.legend(frameon=False, fontsize=5.4, loc='upper left', handlelength=1.3, labelspacing=0.2, ncol=1)
style(ax)

# (b) real benchmarks: certified accuracy vs calibration budget (r=.95), mean over CIFAR-100 and miniImageNet
ax = axes[1]
sw = json.loads(SWEEP.read_text())
CAPS = [5, 10, 20, 40]
for k, lab, c, mk in [('dense', 'Dense', MAGENTA, '^'), ('BiMC_transfer', 'BiMC', GREEN, 'o'),
                      ('FeCAM_transfer', 'FeCAM', AQUA, 'D'), ('GDA_transfer', 'GDA', ORANGE, 's'),
                      ('full_gda', 'Ours', RED, '*')]:
    v = [sum(sw[f'b16/{k}/0.95_{cap}'][ds]['novel'] for ds in ('cifar100', 'mini')) / 2 for cap in CAPS]
    ours = k == 'full_gda'
    ax.plot(CAPS, v, color=c, marker=mk, markersize=6 if ours else 3.2, linewidth=1.8 if ours else 1.0,
            markeredgecolor='white', markeredgewidth=0.4, zorder=5 if ours else 3, label=lab)
ax.set_xscale('log', base=2)
ax.set_xticks(CAPS)
ax.set_xticklabels([str(c) for c in CAPS])
ax.set_xlabel('Calibration images per class')
ax.set_ylabel('Certified acc. (%)')
ax.set_title('Real benchmarks', color=INK)
ax.set_ylim(-3, 112)
ax.set_yticks([0, 20, 40, 60, 80])
ax.legend(frameon=False, fontsize=5.4, loc='upper center', handlelength=1.3, labelspacing=0.2, ncol=3, columnspacing=0.7)
style(ax)

# (a) alignment ablation, B/16 average over three benchmarks
ax = axes[0]
avg = json.loads(SUMMARY.read_text())['average']
VAR = [('full', 'w/o cov.', '#d6d4cc'), ('full_gda_rawg', 'Unanch.', '#b8b6ae'),
       ('full_gda_none', 'w/o cent.', '#a09e96'), ('full_gda_fixed', 'Fixed', '#7d7b74'), ('full_gda', 'Ours', RED)]
w = 0.38
for i, (k, lab, c) in enumerate(VAR):
    a, ce = avg[f'b16/{k}']['Avg'], avg[f'b16/{k}']['certified_common']
    ax.bar(i - w / 2 - 0.02, a, width=w, color=c, edgecolor='white', linewidth=0.6, zorder=3)
    ax.bar(i + w / 2 + 0.02, ce, width=w, color=c, edgecolor='white', linewidth=0.6, alpha=0.55,
           hatch='////', zorder=3)
    if k == 'full_gda':
        ax.text(i - w / 2 - 0.02, a + 0.5, f'{a:.1f}', ha='center', fontsize=5.4, fontweight='bold')
        ax.text(i + w / 2 + 0.02, ce + 0.5, f'{ce:.1f}', ha='center', fontsize=5.4, fontweight='bold')
ax.set_xticks(range(len(VAR)))
ax.set_xticklabels([v[1] for v in VAR], fontsize=6.0, rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(60, 85)
ax.set_yticks([60, 65, 70, 75, 80, 85])
ax.set_title('Alignment ablation', color=INK, pad=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#9a9a9a', label='Average acc.'),
                   Patch(facecolor='#d0d0d0', hatch='////', label='Certified acc.')],
          frameon=False, fontsize=5.6, loc='lower center', bbox_to_anchor=(0.5, 0.99), handlelength=1.2, ncol=2, columnspacing=0.8,
          borderaxespad=0.1)
style(ax, xgrid=False)

for a_ in axes:  # align all panel titles with (a), whose legend sits between title and axes
    a_.set_title(a_.get_title(), color=INK, pad=11)
tag(axes, y=-0.36)
save(fig, HERE.parent / 'fig_mechanism')
