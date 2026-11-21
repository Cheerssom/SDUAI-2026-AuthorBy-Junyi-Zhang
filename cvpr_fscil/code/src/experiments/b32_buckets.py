"""Backfill h/m/t + ECE for the B/32 intermediate ensemble (71.68 run)."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from lt_dataset import make_lt_datasets
from train import TF_CLIP, split_train_val, cache_features
from clip_encoder import CLIPVisionEncoder
from linear_probe import zscore_rows

device = 'cuda'
ck = torch.load('../runs/ckpt_clip_std100.pt', map_location=device,
                weights_only=False)
enc = CLIPVisionEncoder().to(device)
enc.load_state_dict(ck['encoder']); enc.eval()
train_eval, _, _, test, meta = make_lt_datasets(
    rho=100, train_tf=TF_CLIP, support_tf=TF_CLIP, test_tf=TF_CLIP)
fit, val = split_train_val(train_eval)
counts = np.array(meta['cls_num_list'])
BUCKET = np.where(counts > 100, 'head', np.where(counts >= 20, 'medium', 'tail'))
lp = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)

cfit = cache_features(enc, fit, device)
cval = cache_features(enc, val, device)
ctst = cache_features(enc, test, device)

# same recipe as the 71.68 run: 60ep AdamW lr1e-3 wd0.01 balanced
W = nn.Linear(768, 100).to(device)
opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.01)
w = torch.tensor(1.0 / counts[cfit.ys.numpy()], dtype=torch.double)
smp = torch.utils.data.WeightedRandomSampler(w, len(cfit.pooled),
                                             replacement=True)
ds = TensorDataset(cfit.pooled, cfit.ys)
g = torch.Generator().manual_seed(0)
for ep in range(60):
    for xb, yb in DataLoader(ds, batch_size=256, sampler=smp, generator=g):
        loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    lv = (cval.pooled.to(device) @ W.weight.T + W.bias).cpu()
    lt = (ctst.pooled.to(device) @ W.weight.T + W.bias).cpu()
yv = cval.ys.numpy(); yt = ctst.ys.numpy()

# B/32 text anchors (single prompt, saved earlier)
T = torch.load('text_anchors.pt')  # (100,512) — needs joint-space image embeds
from transformers import CLIPModel
hf = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').cuda().eval()


@torch.no_grad()
def jemb(ds):
    out = []
    for x, _, _, _ in DataLoader(ds, batch_size=256):
        e = hf.get_image_features(pixel_values=x.cuda())
        out.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(out)


jv, jt = jemb(val), jemb(test)
zs_v = (jv @ T.T).numpy(); zs_t = (jt @ T.T).numpy()

# reproduce the val-selected ensemble (tau grid + weights)
best = (-1.0, None)
for tau in np.arange(0, 1.01, 0.2):
    a = zscore_rows(lv - tau * lp[None, :]).numpy()
    b = zscore_rows(torch.tensor(zs_v)).numpy()
    for wa in (0, 0.5, 1, 2):
        for wb in (0, 0.5, 1, 2):
            acc = float(((wa * a + wb * b).argmax(1) == yv).mean())
            if acc > best[0]:
                best = (acc, (tau, wa, wb))
tau, wa, wb = best[1]
a = zscore_rows(lv - tau * lp[None, :]).numpy()
b = zscore_rows(torch.tensor(zs_v)).numpy()
sc = wa * a + wb * b
pred = sc.argmax(1)
hit = pred == yt
b_of_y = BUCKET[yt]
macro = float(np.mean([hit[yt == c].mean() for c in range(100)]))
# temperature fit on val then ECE
scv = wa * zscore_rows(lv - tau * lp[None, :]).numpy() + wb * \
    zscore_rows(torch.tensor(zs_v)).numpy()
bestT = (1.0, 1e9)
for TT in np.concatenate([np.geomspace(0.05, 1, 10), np.geomspace(1, 50, 10)]):
    p = torch.softmax(torch.tensor(scv / TT), 1).numpy()
    conf = p.max(1); h_ = (p.argmax(1) == yv).astype(float)
    e = sum(((conf > lo) & (conf <= lo + 1/15)).mean() *
            abs(h_[(conf > lo) & (conf <= lo + 1/15)].mean() -
                conf[(conf > lo) & (conf <= lo + 1/15)].mean())
            for lo in np.linspace(0, 1, 16)[:-1] if ((conf > lo) &
            (conf <= lo + 1/15)).sum() > 0)
    if e < bestT[1]:
        bestT = (TT, e)
p = torch.softmax(torch.tensor(sc / bestT[0]), 1).numpy()
conf = p.max(1); h_ = hit.astype(float)
ece = sum(((conf > lo) & (conf <= lo + 1/15)).mean() *
          abs(h_[(conf > lo) & (conf <= lo + 1/15)].mean() -
              conf[(conf > lo) & (conf <= lo + 1/15)].mean())
          for lo in np.linspace(0, 1, 16)[:-1] if ((conf > lo) &
          (conf <= lo + 1/15)).sum() > 0)
print(f'B/32 ensemble (tau={tau:.1f}, w=({wa},{wb})): '
      f'overall {hit.mean():.4f}  macro {macro:.4f}  '
      f'h {hit[b_of_y=="head"].mean():.4f}  '
      f'm {hit[b_of_y=="medium"].mean():.4f}  '
      f't {hit[b_of_y=="tail"].mean():.4f}  ECE {ece:.4f}')
