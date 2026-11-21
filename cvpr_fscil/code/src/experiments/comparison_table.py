"""Multi-metric comparison table for CIFAR-100-LT rho=100.

Sources:
  - coursework per-class results (E:\机器学习\大作业) -> overall/macro/h/m/t
  - our cached ViT-L features (vitl_feats.pt) -> full metrics incl. ECE
  - literature (LPT/VPT/BALLAD): overall only, from their papers
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
import torch.nn.functional as F
from config import CFG

CLS = json.load(open(
    CFG.baseline_dir / 'experiment_results' / 'all_results.json'))['_cls_num_list']
CLS = np.array(CLS)
BUCKET = np.where(CLS > 100, 'head', np.where(CLS >= 20, 'medium', 'tail'))


def buckets_from_percls(per):
    per = np.array(per, dtype=float) / 100.0     # stored in percent
    return (per.mean(), per.mean(),
            per[BUCKET == 'head'].mean(),
            per[BUCKET == 'medium'].mean(),
            per[BUCKET == 'tail'].mean())


def ece15(probs, ys, bins=15):
    conf = probs.max(1)
    hit = (probs.argmax(1) == ys).astype(float)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.sum():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return e


rows = []

# ---- coursework from-scratch track (ResNet-32) ----
src = json.load(open(
    CFG.baseline_dir / 'experiment_results_final' / 'all_results.json'))
for name in ['ResNet32+CE', 'ResNet32+LDAM-DRW', 'ResNet32+BalancedSoftmax',
             'GLMC', 'GLMC + MaxNorm']:
    per = src[name]['Balanced Test']['per_cls_acc']
    o, ma, h, m, t = buckets_from_percls(per)
    rows.append((name, 'from-scratch ResNet-32', o, ma, h, m, t, None))

# ---- our frozen-CLIP track (computed) ----
d = torch.load('vitl_feats.pt')
jtst, ytst = d['jtst'], d['ytst'].numpy()
W = torch.load('vitl_std.pt')
counts = CLS
lp = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)


def to_np(x):
    return x.numpy() if isinstance(x, torch.Tensor) else x


def eval_preds(logits, probs_for_ece=None):
    pred = logits.argmax(1)
    hit = pred == ytst
    b_of_y = BUCKET[ytst]                       # per-sample bucket
    ma = float(np.mean([hit[ytst == c].mean() for c in range(100)]))
    e = ece15(probs_for_ece, ytst) if probs_for_ece is not None else None
    return (float(hit.mean()), ma,
            float(hit[b_of_y == 'head'].mean()),
            float(hit[b_of_y == 'medium'].mean()),
            float(hit[b_of_y == 'tail'].mean()), e)


# temperature fit on val (grid), ECE on test at fitted T
def fit_T(score_val, y_val):
    best = (1.0, 1e9)
    for T in np.concatenate([np.geomspace(0.01, 1, 12), np.geomspace(1, 100, 12)]):
        p = torch.softmax(torch.tensor(score_val / T), 1).numpy()
        e = ece15(p, y_val)
        if e < best[1]:
            best = (T, e)
    return best[0]


jval, yval = d['jval'], d['yval'].numpy()

# zero-shot ensemble
zs = (jtst @ W['T'].T).numpy()
zs_v = (jval @ W['T'].T).numpy()
T0 = fit_T(zs_v, yval)
r = eval_preds(zs, torch.softmax(torch.tensor(zs / T0), 1).numpy())
rows.append(('CLIP-L/14 zero-shot (40 tpl)', 'frozen ViT-L/14', *r))

# balanced linear probe
lt = (jtst.to('cuda') @ W['W'].T.to('cuda') + W['b'].to('cuda')).cpu().numpy()
lv = (jval.to('cuda') @ W['W'].T.to('cuda') + W['b'].to('cuda')).cpu().numpy()
T1 = fit_T(lv, yval)
r = eval_preds(lt, torch.softmax(torch.tensor(lt / T1), 1).numpy())
rows.append(('linear probe (balanced)', 'frozen ViT-L/14', *r))

# ensemble tau=0.3 w=(0.5,0.25) — reproduce from cached pieces
from linear_probe import zscore_rows
at = zscore_rows(torch.tensor(lt) - 0.3 * lp[None, :]).numpy()
zst = zscore_rows(torch.tensor(zs)).numpy()
sc = 0.5 * at + 0.25 * zst
sc_v = 0.5 * zscore_rows(torch.tensor(lv) - 0.3 * lp[None, :]).numpy() \
    + 0.25 * zscore_rows(torch.tensor(zs_v)).numpy()
T2 = fit_T(sc_v, yval)
r = eval_preds(sc, torch.softmax(torch.tensor(sc / T2), 1).numpy())
rows.append(('JLT unified (ours, final)', 'frozen ViT-L/14', *r))

# B/32 final for the backbone ablation row
r0 = (71.68 / 100, None, None, None, None, None)
rows.append(('JLT unified (B/32 version)', 'frozen ViT-B/32',
             0.7168, None, None, None, None, None))

# ---- literature (overall only) ----
rows.append(('BALLAD (Ma et al. 21)', 'frozen CLIP (per LPT table)',
             0.778, None, None, None, None, None))
rows.append(('VPT (per LPT table)', 'ViT-B/16 IN21k + prompt-tune',
             0.810, None, None, None, None, None))
rows.append(('LPT (ICLR23) SOTA', 'ViT-B/16 IN21k + prompt-tune',
             0.891, None, None, None, None, None))

# ---- print ----
print(f"{'method':<28} {'track':<26} {'overall':>7} {'macro':>7} "
      f"{'head':>7} {'med':>7} {'tail':>7} {'ECE':>7}")
print('-' * 100)
for name, tr, o, ma, h, m, t, e in rows:
    f = lambda x: f'{x*100:7.2f}' if x is not None else '      -'
    print(f"{name:<28} {tr:<26} {f(o)} {f(ma)} {f(h)} {f(m)} {f(t)} "
          f"{f(e) if e is not None else '      -'}")
