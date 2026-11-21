"""ViT-L/14 zero-retrain expansion protocol (frozen, joint space).

TRAIN(40) candidates carry: learned linear weights + text anchor + prior.
NEW(40) candidates carry:   text anchor + k-shot visual prototype.
Gates (a_lin x b_txt x c_proto x delta) selected on a pre-declared 20% test
slice (seed 123); reported on the remaining 80%. NEG(20) classes untouched.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPTokenizer
from lt_dataset import (make_lt_datasets, IndexedCifar, subset_by_class,
                        k_shots_per_class, partition_classes)
from train import TF_CLIP, split_train_val
from linear_probe import zscore_rows

from config import CFG

device = 'cuda'
with open(CFG.cifar_root / 'meta', 'rb') as f:
    names_all = pickle.load(f, encoding='latin1')['fine_label_names']
names_all = [n.replace('_', ' ') for n in names_all]
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda().eval()
tkz = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

TEMPLATES = ['a bad photo of a {}.', 'a sculpture of a {}.',
 'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
 'a painting of the {}.', 'a cropped photo of the {}.', 'a tattoo of the {}.',
 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.',
 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.',
 'a photo of the cool {}.', 'a close-up photo of a {}.',
 'a black and white photo of the {}.', 'a pixelated photo of the {}.',
 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.',
 'a photo of a clean {}.', 'a origami {}.', 'a photo of a large {}.',
 'a rendition of the {}.', 'a photo of a small {}.', 'itap of the {}.',
 'a photo of a nice {}.', 'a photo of a weird {}.', 'a photo of the large {}.',
 'a photo of the small {}.', 'a photo of a {}.']


@torch.no_grad()
def jemb(ds):
    out = []
    for x, _, _, _ in DataLoader(ds, batch_size=128):
        out.append(F.normalize(
            model.get_image_features(pixel_values=x.cuda()).float(), dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def text_anchor(ids):
    Ts = []
    for t in TEMPLATES:
        b = tkz([t.format(names_all[c]) for c in ids],
                return_tensors='pt', padding=True)
        Ts.append(F.normalize(
            model.get_text_features(**{k: v.cuda() for k, v in b.items()}).float(),
            dim=1).cpu())
    return F.normalize(torch.stack(Ts).mean(0), dim=1)


train_eval, _, _, test, meta = make_lt_datasets(
    rho=100, train_tf=TF_CLIP, support_tf=TF_CLIP, test_tf=TF_CLIP)
tids, nids, gids = partition_classes(meta, 40, 20, seed=0)
fit, _ = split_train_val(train_eval)

# TRAIN linear probe (40-way) on fit split of TRAIN classes
keep = np.isin(fit.fine, tids)
fit_train = IndexedCifar(fit.x[keep], fit.fine[keep], fit.coarse[keep], TF_CLIP)
pos_of = {int(c): i for i, c in enumerate(tids)}
jfit = jemb(fit_train)
yfit = torch.tensor([pos_of[int(y)] for y in fit_train.fine])
counts40 = np.bincount(yfit.numpy(), minlength=40).astype(float) + 1e-9
W40 = nn.Linear(768, 40).to(device)
opt = torch.optim.AdamW(W40.parameters(), lr=1e-3, weight_decay=0.05)
w = torch.tensor(1.0 / counts40[yfit.numpy()], dtype=torch.double)
smp = torch.utils.data.WeightedRandomSampler(w, len(jfit), replacement=True)
ds = TensorDataset(jfit, yfit)
g = torch.Generator().manual_seed(0)
for ep in range(100):
    for xb, yb in DataLoader(ds, batch_size=256, sampler=smp, generator=g):
        loss = F.cross_entropy(W40(xb.to(device)), yb.to(device))
        opt.zero_grad(); loss.backward(); opt.step()

sub_ids = np.concatenate([tids, nids])
K0, Kn = 40, len(nids)
T80 = text_anchor(sub_ids)                      # (80,768)
lp_old = torch.tensor(np.log(counts40 / counts40.sum()),
                      dtype=torch.float32)

# slice / rest split of test (TRAIN+NEW classes)
rng = np.random.RandomState(123)
sel_idx = []
for c in sub_ids:
    idx = np.where(test.fine == c)[0]
    rng.shuffle(idx)
    sel_idx.extend(idx[:20])
sel_mask = np.zeros(len(test.fine), dtype=bool)
sel_mask[np.array(sel_idx)] = True
sel_set = IndexedCifar(test.x[sel_mask], test.fine[sel_mask],
                       test.coarse[sel_mask], TF_CLIP)
rest_set = IndexedCifar(test.x[~sel_mask], test.fine[~sel_mask],
                        test.coarse[~sel_mask], TF_CLIP)
js, jr = jemb(sel_set), jemb(rest_set)
id2pos = {int(c): i for i, c in enumerate(sub_ids)}
tgt_s = np.array([id2pos[int(y)] for y in sel_set.fine])
m_new_r = np.isin(rest_set.fine, nids)
m_old_r = np.isin(rest_set.fine, tids)

zT_s = zscore_rows(js @ T80.T).numpy()
zT_r = zscore_rows(jr @ T80.T).numpy()
with torch.no_grad():
    lin_s = (js.to(device) @ W40.weight.T + W40.bias).cpu()
    lin_r = (jr.to(device) @ W40.weight.T + W40.bias).cpu()
zlin_s = zscore_rows(torch.cat(
    [lin_s, torch.full((len(lin_s), Kn), float('-inf'))], 1)).numpy()
zlin_r = zscore_rows(torch.cat(
    [lin_r, torch.full((len(lin_r), Kn), float('-inf'))], 1)).numpy()
new_pool, _ = subset_by_class(train_eval, nids)

for k in (1, 2, 5, 10):
    shots = k_shots_per_class(new_pool, nids, k, seed=7)
    jsh = jemb(shots)
    proto = {}
    for p, yy in zip(jsh, shots.fine):
        proto.setdefault(int(yy), []).append(p)
    P80 = torch.stack([torch.stack(proto[int(c)]).mean(0) for c in nids])
    P80 = F.normalize(P80, dim=1)
    Pfull = torch.cat([torch.zeros(K0, 768), P80])   # prototypes only for NEW
    cos_s = torch.cat([torch.full((len(js), K0), float('-inf')),
                       js @ P80.T], 1)
    cos_r = torch.cat([torch.full((len(jr), K0), float('-inf')),
                       jr @ P80.T], 1)
    zP_s = zscore_rows(cos_s).numpy()
    zP_r = zscore_rows(cos_r).numpy()
    m_blk = np.array([0.0] * K0 + [1.0] * Kn)
    best = (-1.0, None)
    for a in (0.0, 0.5, 1.0):
        for bt in (0.25, 0.5, 1.0, 2.0):
            for cp in (0.0, 0.25, 0.5, 1.0):
                for d in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
                    sc = a * zlin_s + bt * zT_s + cp * zP_s + d * m_blk[None, :]
                    m_ = float(np.mean([np.mean(sc[tgt_s == c].argmax(1) == c)
                                        for c in np.unique(tgt_s)]))
                    if m_ > best[0]:
                        best = (m_, (a, bt, cp, d))
    a, bt, cp, d = best[1]
    sc_r = a * zlin_r + bt * zT_r + cp * zP_r + d * m_blk[None, :]
    pred = sub_ids[sc_r.argmax(1)]
    old = float((pred == rest_set.fine)[m_old_r].mean())
    new = float((pred == rest_set.fine)[m_new_r].mean())
    print(f'shots={k:>2} gates(a={a},bt={bt},cp={cp},d={d}): '
          f'old {old:.4f}  new {new:.4f}  (zero grad steps, 80% test)')

# reference points on the same slice protocol
zs_new = float((sub_ids[(zT_r[:, K0:].argmax(1) + K0)] == rest_set.fine)[m_new_r].mean())
print(f'[ref] text-only new-class acc: {zs_new:.4f}')
