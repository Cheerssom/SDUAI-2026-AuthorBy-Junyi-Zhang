"""Visual recognition report: confusion matrix, per-class accuracy, samples."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

# ---- load predictions from the 90.42 TTE run ----
# Recompute from cached scores (same as final_tte.py but load from disk)
# Use the v1 model (which got 90.42)

from config import CFG

with open(CFG.cifar_root / 'meta', 'rb') as f:
    names = pickle.load(f, encoding='latin1')['fine_label_names']
names = [n.replace('_', ' ') for n in names]

CLS = np.array(json.load(open(
    CFG.baseline_dir / 'experiment_results' / 'all_results.json')
    )['_cls_num_list'])
BUCKET = np.where(CLS > 100, 'head', np.where(CLS >= 20, 'medium', 'tail'))
cmap = {'head': '#2196F3', 'medium': '#FF9800', 'tail': '#F44336'}

from lt_dataset import make_lt_datasets
from train import TF_CLIP, split_train_val

# quick: use frozen ViT-L features already cached
d = torch.load('vitl_feats.pt')
Wd = torch.load('vitl_std.pt')
T, Wf, bf = Wd['T'], Wd['W'], Wd['b']
jtst = d['jtst']
yt = d['ytst'].numpy()

# approximate the 90.42 ensemble: LoRA logits not cached, use frozen-only
# (82.28 model) for visualization — enough to show the pattern
from linear_probe import zscore_rows
lp = torch.tensor(np.log(CLS / CLS.sum()), dtype=torch.float32)
zs = zscore_rows(jtst @ T.T).numpy()
ln = zscore_rows(jtst @ Wf.T + bf - 0.3 * lp[None, :]).numpy()
sc = 0.5 * ln + 0.25 * zs
pred = sc.argmax(1)
hit = pred == yt

# ---- figure 1: per-class accuracy (sorted, colored by bucket) ----
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('CIFAR-100-LT ρ=100 — Frozen ViT-L/14 Unified Readout (82.3%)',
             fontsize=14, fontweight='bold')

# (0,0) per-class accuracy
ax = axes[0, 0]
order = np.argsort(-np.array([hit[yt == c].mean() for c in range(100)]))
per_cls = np.array([hit[yt == c].mean() for c in range(100)])[order]
colors = [cmap[BUCKET[c]] for c in order]
ax.bar(range(100), per_cls * 100, color=colors, edgecolor='none', width=0.8)
ax.set_xlabel('Classes (sorted by accuracy)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Per-Class Accuracy (sorted)')
ax.axhline(y=82.3, color='k', linestyle='--', alpha=0.5, label='Overall 82.3%')
ax.legend()
patches = [mpatches.Patch(color=cmap[b], label=f'{b} ({(BUCKET==b).sum()} cls)')
           for b in ('head', 'medium', 'tail')]
ax.legend(handles=patches + [plt.Line2D([0], [0], color='k', ls='--',
                                         label='Overall')], loc='lower right')
ax.set_ylim(0, 105)

# (0,1) confusion matrix (top-20 most confused pairs)
ax = axes[0, 1]
cm = np.zeros((100, 100), dtype=int)
for t, p in zip(yt, pred):
    cm[t, p] += 1
np.fill_diagonal(cm, 0)  # remove correct
# top confused pairs
pairs = []
for i in range(100):
    for j in range(100):
        if i != j and cm[i, j] > 0:
            pairs.append((cm[i, j], i, j))
pairs.sort(reverse=True)
top = pairs[:20]
labels = [f'{names[i][:8]}→{names[j][:8]}' for _, i, j in top]
vals = [v for v, _, _ in top]
ypos = range(len(top))[::-1]
bc = [cmap[BUCKET[i]] for _, i, _ in top]
ax.barh(ypos, vals, color=bc, edgecolor='none', height=0.7)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Misclassified count')
ax.set_title('Top-20 Most Confused Class Pairs')

# (1,0) accuracy by bucket (box plot)
ax = axes[1, 0]
data_by_bucket = []
for b in ('head', 'medium', 'tail'):
    cls_in = [c for c in range(100) if BUCKET[c] == b]
    data_by_bucket.append([hit[yt == c].mean() * 100 for c in cls_in])
bp = ax.boxplot(data_by_bucket, labels=['Head\n(>100 img)', 'Medium\n(20-100)',
                                        'Tail\n(<20)'], patch_artist=True,
                widths=0.5)
for patch, b in zip(bp['boxes'], ('head', 'medium', 'tail')):
    patch.set_facecolor(cmap[b]); patch.set_alpha(0.6)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Distribution by Bucket')
ax.grid(axis='y', alpha=0.3)

# (1,1) training samples vs accuracy scatter
ax = axes[1, 1]
for b in ('head', 'medium', 'tail'):
    cls_in = [c for c in range(100) if BUCKET[c] == b]
    accs = [hit[yt == c].mean() * 100 for c in cls_in]
    ax.scatter(CLS[cls_in], accs, c=cmap[b], s=30, alpha=0.7, label=b.capitalize())
ax.set_xlabel('Training samples per class')
ax.set_ylabel('Test accuracy (%)')
ax.set_title('Sample Count vs Accuracy (imbalance effect)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('../experiments/recognition_report.png', dpi=150,
            bbox_inches='tight')
print('saved ../experiments/recognition_report.png')

# ---- text summary ----
print('\n=== 详细识别报告 ===')
print(f'\n总体: {hit.mean()*100:.2f}%  ({hit.sum()}/{len(yt)})')
for b in ('head', 'medium', 'tail'):
    cls_in = [c for c in range(100) if BUCKET[c] == b]
    accs = {c: hit[yt == c].mean() * 100 for c in cls_in}
    print(f'\n{b.upper()} ({len(cls_in)} classes, '
          f'{sum(CLS[cls_in])} train samples):')
    print(f'  avg: {np.mean(list(accs.values())):.1f}%  '
          f'min: {min(accs.values()):.1f}%  max: {max(accs.values()):.1f}%')

per_cls_acc = {c: hit[yt == c].mean() * 100 for c in range(100)}
best5 = sorted(per_cls_acc, key=per_cls_acc.get, reverse=True)[:5]
worst5 = sorted(per_cls_acc, key=per_cls_acc.get)[:5]
print('\n最好 5 类:')
for c in best5:
    print(f'  {names[c]:<20} {per_cls_acc[c]:.0f}%  '
          f'({CLS[c]} train, {BUCKET[c]})')
print('\n最差 5 类:')
for c in worst5:
    print(f'  {names[c]:<20} {per_cls_acc[c]:.0f}%  '
          f'({CLS[c]} train, {BUCKET[c]})')

print('\n最严重混淆 (真实→预测):')
for cnt, i, j in top[:5]:
    print(f'  {names[i]:<18} → {names[j]:<18} {cnt} 次'
          f'  ({BUCKET[i]}→{BUCKET[j]})')
