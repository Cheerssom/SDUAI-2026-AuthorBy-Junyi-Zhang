"""Fusion-rule ablation under fixed features and partition (audit #2).

Holds features, partition (seed 0), support shots (seed 7, k=5) fixed and
varies ONLY the combination operator:
  raw      weighted sum of raw branch scores (learned logits / cosines)
  zscore   per-row z-scored branches (ours)
  probavg  per-branch softmax at temperature T, then weighted sum of probs
Each rule selects (a,b,c,d) on the same gate-val (old train-val + one held-out
support shot per new class); the grid is identical across rules.
Then a dedicated d-offset sweep at the pre-registered operating point.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import io
sys = _sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda'
t0 = time.time()
CACHE = _Path(__file__).resolve().parents[1] / 'expand_cache.pt'
assert CACHE.exists(), 'run expansion_baselines.py first (builds the cache)'

from lt_dataset import make_lt_datasets, partition_classes, subset_by_class, \
    k_shots_per_class
from train import TF_CLIP, split_train_val

train_eval, _, _, test, meta = make_lt_datasets(
    rho=100, train_tf=TF_CLIP, support_tf=TF_CLIP, test_tf=TF_CLIP)
fit, val = split_train_val(train_eval)
y_test = test.fine.astype(np.int64)
fit_fine = fit.fine.astype(np.int64)
val_fine = val.fine.astype(np.int64)

d_cache = torch.load(CACHE)
E_fit, F_fit, E_val, F_val, E_test, F_test, T100 = (
    d_cache['E_fit'], d_cache['F_fit'], d_cache['E_val'], d_cache['F_val'],
    d_cache['E_test'], d_cache['F_test'], d_cache['T100'])
print(f'cache loaded ({time.time()-t0:.0f}s)', flush=True)

from transformers import CLIPModel
mL = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device).eval()
CL_MEAN = torch.tensor([.48145466, .4578275, .40821073]).view(1, 3, 1, 1).to(device)
CL_STD = torch.tensor([.26862954, .26130258, .27577711]).view(1, 3, 1, 1).to(device)
IM_MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1).to(device)
IM_STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1).to(device)

import timm
from lora_v2 import LoRALinear
ck = torch.load('lora_in21k_b16.pt', map_location=device, weights_only=False)
m16 = timm.create_model(ck.get('model', 'vit_base_patch16_224.augreg_in21k'),
                        pretrained=True, num_classes=0).to(device)
for blk_ in m16.blocks:
    blk_.attn.qkv = LoRALinear(blk_.attn.qkv, ck.get('rank', 8),
                               ck.get('alpha', 16)).to(device)
m16.load_state_dict(ck['lora'], strict=False)
m16.eval()


@torch.no_grad()
def clip_embed(ds, bs=128):
    out = []
    for i in range(0, len(ds), bs):
        x = torch.from_numpy(ds.x[i:i + bs]).float().div_(255.)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x = (x.to(device) - CL_MEAN) / CL_STD
        with torch.autocast('cuda', torch.float16):
            v = mL.vision_model(pixel_values=x)
            e = mL.visual_projection(v.pooler_output)
        out.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def lora_embed(ds, bs=192):
    out = []
    for i in range(0, len(ds), bs):
        x = torch.from_numpy(ds.x[i:i + bs]).float().div_(255.)
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x = (x.to(device) - IM_MEAN) / IM_STD
        with torch.autocast('cuda', torch.float16):
            f = m16.forward_features(x)
        out.append(f[:, 0].float().cpu())
    return torch.cat(out)


PART_SEED, SUP_SEED, K = 0, 7, 5
tids, nids, _ = partition_classes(meta, 40, 20, seed=PART_SEED)
K0, Kn = len(tids), len(nids)
sub_ids = np.concatenate([tids, nids])
pos_of = {int(c): i for i, c in enumerate(sub_ids)}
T80 = T100[sub_ids]
m_old = np.isin(y_test, tids)
m_new = np.isin(y_test, nids)
pos_y = np.full(len(y_test), -1, dtype=np.int64)
for i, y in enumerate(y_test):
    p = pos_of.get(int(y))
    if p is not None:
        pos_y[i] = p
blk = np.array([0] * K0 + [1] * Kn)

# learned column (probe on fit, partition 0)
keep = np.isin(fit_fine, tids)
Ff, yf = F_fit[keep], torch.tensor([pos_of[int(y)] for y in fit_fine[keep]])
counts40 = np.bincount(yf.numpy(), minlength=K0).astype(float) + 1
W = nn.Linear(Ff.shape[1], K0).to(device)
opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
w_samp = torch.tensor(1.0 / counts40[yf.numpy()], dtype=torch.double)
smp = torch.utils.data.WeightedRandomSampler(w_samp, len(Ff), replacement=True)
ds_probe = TensorDataset(Ff, yf)
g = torch.Generator().manual_seed(0)
for ep in range(100):
    for xb, yb in DataLoader(ds_probe, batch_size=256, sampler=smp, generator=g):
        loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    Ww, bb = W.weight.cpu(), W.bias.cpu()
print(f'probe done ({time.time()-t0:.0f}s)', flush=True)

# support shots: k for prototypes + 1 held out per class for gate-val.
# Tail classes may not afford k+1 images; group by class instead of slicing.
new_pool, _ = subset_by_class(train_eval, nids)
shots = k_shots_per_class(new_pool, nids, K + 1, seed=SUP_SEED)
proto_rows, val_rows, val_cls = [], [], []
for c in nids:
    idx = np.where(shots.fine == c)[0]
    proto_rows.append(idx[:K])
    if len(idx) > K:
        val_rows.append(idx[K])
        val_cls.append(int(c))
proto_idx = np.concatenate(proto_rows)
E_proto = clip_embed(type(new_pool)(shots.x[proto_idx], shots.fine[proto_idx],
                                    shots.coarse[proto_idx], None))
P_new = torch.stack([
    F.normalize(E_proto[shots.fine[proto_idx] == c].mean(0), dim=0) for c in nids])
val_idx = np.array(val_rows, dtype=np.int64)
E_vshot = clip_embed(type(new_pool)(shots.x[val_idx], shots.fine[val_idx],
                                    shots.coarse[val_idx], None))
F_vshot = lora_embed(type(new_pool)(shots.x[val_idx], shots.fine[val_idx],
                                    shots.coarse[val_idx], None))

# gate-val rows: old = train-val restricted to tids; new = held-out shots
v_old = np.isin(val_fine, tids)
E_gv = torch.cat([E_val[v_old], E_vshot])
F_gv = torch.cat([F_val[v_old], F_vshot])
y_gv = np.array([pos_of[int(y)] for y in val_fine[v_old]]
                + [pos_of[c] for c in val_cls])
print(f'gate-val rows: {len(y_gv)} '
      f'(old {int(v_old.sum())} + new {len(val_cls)})', flush=True)

NEG = -1e9


def branch_raw(E, Fm):
    l = torch.cat([Fm @ Ww.T + bb, torch.zeros(len(E), Kn)], 1).numpy()
    t = (E @ T80.T).numpy()
    p = torch.cat([torch.zeros(len(E), K0), E @ P_new.T], 1).numpy()
    return l, t, p


def zsr_np(a):
    mu = a.mean(axis=1, keepdims=True)
    sd = np.maximum(a.std(axis=1, keepdims=True), 1e-6)
    return (a - mu) / sd


def branch_z(E, Fm):
    l, t, p = branch_raw(E, Fm)
    lz = zsr_np(l[:, :K0])
    l80 = np.concatenate([lz, np.zeros((len(E), Kn))], 1)
    p80 = np.concatenate([np.zeros((len(E), K0)), zsr_np(p[:, K0:])], 1)
    return l80, zsr_np(t), p80


def softmax_np(x, T):
    z = x / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def branch_prob(E, Fm, T):
    l, t, p = branch_raw(E, Fm)
    lm = np.concatenate([l[:, :K0], np.full((len(E), Kn), NEG)], 1)
    pm = np.concatenate([np.full((len(E), K0), NEG), p[:, K0:]], 1)
    return softmax_np(lm, T), softmax_np(t, T), softmax_np(pm, T)


GRID = (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
DGRID = (0.0, 0.5, 1.0)


def select_and_eval(name, bv, bt_test, get_branches):
    best = (-1, None)
    for a in GRID:
        for b in GRID:
            for c in GRID:
                for dd in DGRID:
                    sc = a * bv[0] + b * bv[1] + c * bv[2] + dd * blk[None, :]
                    acc = float((sc.argmax(1) == y_gv).mean())
                    if acc > best[0]:
                        best = (acc, (a, b, c, dd))
    a, b, c, dd = best[1]
    tb = get_branches()
    sc = a * tb[0] + b * tb[1] + c * tb[2] + dd * blk[None, :]
    hit = sc.argmax(1) == pos_y
    out = {'gates': best[1], 'val': best[0],
           'old': float(hit[m_old].mean()),
           'new': float(hit[m_new].mean()),
           'overall': float(hit[m_old | m_new].mean())}
    print(f'{name}: gates={best[1]} val={best[0]:.4f} '
          f'old {out["old"]:.4f} new {out["new"]:.4f}', flush=True)
    return out


rules = {}
# --- raw ---
raw_v = branch_raw(E_gv, F_gv)
rules['raw'] = select_and_eval(
    'raw', raw_v, None, lambda: branch_raw(E_test, F_test))
# --- zscore (ours) ---
z_v = branch_z(E_gv, F_gv)
rules['zscore'] = select_and_eval(
    'zscore', z_v, None, lambda: branch_z(E_test, F_test))
# --- probavg: temperature selected on val ---
best_t = (-1, None)
for T in (0.5, 1.0, 2.0):
    pv = branch_prob(E_gv, F_gv, T)
    best = (-1, None)
    for a in GRID:
        for b in GRID:
            for c in GRID:
                for dd in DGRID:
                    sc = a * pv[0] + b * pv[1] + c * pv[2] + dd * blk[None, :]
                    acc = float((sc.argmax(1) == y_gv).mean())
                    if acc > best[0]:
                        best = (acc, (a, b, c, dd))
    if best[0] > best_t[0]:
        best_t = (best[0], (T,) + best[1])
T_sel, a, b, c, dd = best_t[1]
pb = branch_prob(E_test, F_test, T_sel)
sc = a * pb[0] + b * pb[1] + c * pb[2] + dd * blk[None, :]
hit = sc.argmax(1) == pos_y
rules['probavg'] = {'T': T_sel, 'gates': (a, b, c, dd), 'val': best_t[0],
                    'old': float(hit[m_old].mean()),
                    'new': float(hit[m_new].mean()),
                    'overall': float(hit[m_old | m_new].mean())}
print(f'probavg: T={T_sel} gates={(a,b,c,dd)} val={best_t[0]:.4f} '
      f'old {rules["probavg"]["old"]:.4f} new {rules["probavg"]["new"]:.4f}',
      flush=True)

# --- d-offset sweep at the pre-registered operating point ---
lz80, tz80, pz80 = branch_z(E_test, F_test)
d_sweep = []
for dd in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    sc = 1.0 * lz80 + 0.5 * tz80 + 0.5 * pz80 + dd * blk[None, :]
    hit = sc.argmax(1) == pos_y
    d_sweep.append({'d': dd, 'old': float(hit[m_old].mean()),
                    'new': float(hit[m_new].mean()),
                    'overall': float(hit[m_old | m_new].mean())})
    print(f'd={dd:.2f}: old {d_sweep[-1]["old"]:.4f} '
          f'new {d_sweep[-1]["new"]:.4f} overall {d_sweep[-1]["overall"]:.4f}',
          flush=True)

with open('../experiments/fusion_ablation.json', 'w') as f:
    json.dump({'rules': rules, 'd_sweep': d_sweep,
               'setting': {'part': PART_SEED, 'sup': SUP_SEED, 'k': K}},
              f, indent=1)
print(f'saved ({time.time()-t0:.0f}s total)')
