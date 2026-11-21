"""Full metric report for the final model: frozen CLIP ViT-L/14 unified readout."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from linear_probe import zscore_rows

d = torch.load('vitl_feats.pt')
jtst, ytst = d['jtst'], d['ytst'].numpy()
jval, yval = d['jval'], d['yval'].numpy()
Wd = torch.load('vitl_std.pt')
W, b, T = Wd['W'], Wd['b'], Wd['T']

counts = np.load if False else None
import json
from config import CFG
CLS = np.array(json.load(open(
    CFG.baseline_dir / 'experiment_results' / 'all_results.json'))
    ['_cls_num_list'])
BUCKET = np.where(CLS > 100, 'head', np.where(CLS >= 20, 'medium', 'tail'))
b_of_y = BUCKET[ytst]
lp = torch.tensor(np.log(CLS / CLS.sum()), dtype=torch.float32)

# final ensemble: tau=0.3, w=(0.5, 0.25) (val-selected)
lt = (jtst @ W.T + b)
lv = (jval @ W.T + b)
at = zscore_rows(lt - 0.3 * lp[None, :]).numpy()
av = zscore_rows(lv - 0.3 * lp[None, :]).numpy()
bt = zscore_rows(jtst @ T.T).numpy()
bv = zscore_rows(jval @ T.T).numpy()
sc = 0.5 * at + 0.25 * bt
scv = 0.5 * av + 0.25 * bv

# temperature fit on val
def ece_of(p, y, bins=15):
    conf = p.max(1); hit = (p.argmax(1) == y).astype(float)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.sum():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(e)

best = (1.0, 1e9)
for TT in np.concatenate([np.geomspace(0.05, 1, 10), np.geomspace(1, 50, 10)]):
    e = ece_of(torch.softmax(torch.tensor(scv / TT), 1).numpy(), yval)
    if e < best[1]:
        best = (float(TT), e)
Tcal = best[0]
probs = torch.softmax(torch.tensor(sc / Tcal), 1).numpy()

pred = sc.argmax(1)
hit = pred == ytst
macro = float(np.mean([hit[ytst == c].mean() for c in range(100)]))
top5 = torch.tensor(sc).topk(5, dim=1).indices.numpy()
hit5 = np.array([ytst[i] in top5[i] for i in range(len(ytst))])

# risk-coverage (threshold on calibrated max-prob)
conf = probs.max(1)
order = np.argsort(-conf)
cum = 1 - np.cumsum(hit[order]) / np.arange(1, len(hit) + 1)
aurc = float(cum.mean())
r90 = float(cum[int(0.9 * len(hit)) - 1])
r95 = float(cum[int(0.95 * len(hit)) - 1])

per_cls = np.array([hit[ytst == c].mean() for c in range(100)])
worst = np.argsort(per_cls)[:5]

print('=== 最终模型完整指标 (CIFAR-100-LT ρ=100, 平衡测试集) ===')
print(f'overall top-1 : {hit.mean()*100:.2f}')
print(f'overall top-5 : {hit5.mean()*100:.2f}')
print(f'macro top-1   : {macro*100:.2f}')
for bname in ('head', 'medium', 'tail'):
    m = b_of_y == bname
    pm = torch.softmax(torch.tensor(sc[m] / Tcal), 1).numpy()
    print(f'{bname:>6}({m.sum():>5}): top-1 {hit[m].mean()*100:6.2f}  '
          f'top-5 {hit5[m].mean()*100:6.2f}  ECE {ece_of(pm, ytst[m])*100:5.2f}')
print(f'ECE (global, T={Tcal:.3f}): {ece_of(probs, ytst)*100:.2f}')
print(f'risk-coverage: AURC {aurc*100:.2f}  risk@cov90 {r90*100:.2f}  '
      f'risk@cov95 {r95*100:.2f}')
print(f'per-class acc: min {per_cls.min()*100:.1f} (class {worst[0]})  '
      f'median {np.median(per_cls)*100:.1f}  max {per_cls.max()*100:.1f}')
print(f'classes >=90%: {(per_cls>=0.9).sum()}   <50%: {(per_cls<0.5).sum()}')
