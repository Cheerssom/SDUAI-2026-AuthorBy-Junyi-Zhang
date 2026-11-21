"""Val-gated expansion across all 9 combos (follow-up to expansion_baselines).

Same 3 partitions x 3 support seeds x 4 shot budgets, but gates selected per
combo on the gate-val (old train-val + one held-out shot per new class),
identical grid to fusion_ablation.py. Produces the 'Ours (val-selected)' row
with mean +/- std, directly comparable to the fixed-gate row and SimpleFusion.
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


def zsr_np(a):
    mu = a.mean(axis=1, keepdims=True)
    sd = np.maximum(a.std(axis=1, keepdims=True), 1e-6)
    return (a - mu) / sd


PART_SEEDS = (0, 1, 2)
SUP_SEEDS = (7, 8, 9)
KS = (1, 2, 5, 10)
GRID = (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
DGRID = (0.0, 0.5, 1.0)

results = []
for ps in PART_SEEDS:
    tids, nids, _ = partition_classes(meta, 40, 20, seed=ps)
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
        for xb, yb in DataLoader(ds_probe, batch_size=256, sampler=smp,
                                 generator=g):
            loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Ww, bb = W.weight.cpu(), W.bias.cpu()

    v_old = np.isin(val_fine, tids)
    E_gv_old, F_gv_old = E_val[v_old], F_val[v_old]
    y_gv_old = np.array([pos_of[int(y)] for y in val_fine[v_old]])

    new_pool, _ = subset_by_class(train_eval, nids)
    for ss in SUP_SEEDS:
        for k in KS:
            shots = k_shots_per_class(new_pool, nids, k + 1, seed=ss)
            proto_idx, val_idx, val_cls = [], [], []
            for c in nids:
                idx = np.where(shots.fine == c)[0]
                proto_idx.append(idx[:k])
                if len(idx) > k:
                    val_idx.append(idx[k])
                    val_cls.append(int(c))
            proto_idx = np.concatenate(proto_idx)
            proto_ds = type(new_pool)(shots.x[proto_idx], shots.fine[proto_idx],
                                      shots.coarse[proto_idx], None)
            val_ds = type(new_pool)(shots.x[val_idx], shots.fine[val_idx],
                                    shots.coarse[val_idx], None) if val_idx else None
            E_proto = clip_embed(proto_ds)
            P_new = torch.stack([
                F.normalize(E_proto[proto_ds.fine == c].mean(0), dim=0)
                for c in nids])

            t80_z_test = zsr_np((E_test @ T80.T).numpy())
            l40_z_test = zsr_np((F_test @ Ww.T + bb).numpy())
            pn_z_test = zsr_np((E_test @ P_new.T).numpy())
            L_test = np.concatenate([l40_z_test, np.zeros((len(E_test), Kn))], 1)
            P_test = np.concatenate([np.zeros((len(E_test), K0)), pn_z_test], 1)

            if val_ds is not None and len(val_ds):
                E_gv = torch.cat([E_gv_old, clip_embed(val_ds)])
                F_gv = torch.cat([F_gv_old, lora_embed(val_ds)])
                y_gv = np.concatenate([y_gv_old,
                                       [pos_of[c] for c in val_cls]])
            else:
                E_gv, F_gv, y_gv = E_gv_old, F_gv_old, y_gv_old

            t80_z_gv = zsr_np((E_gv @ T80.T).numpy())
            l40_z_gv = zsr_np((F_gv @ Ww.T + bb).numpy())
            pn_z_gv = zsr_np((E_gv @ P_new.T).numpy())
            L_gv = np.concatenate([l40_z_gv, np.zeros((len(E_gv), Kn))], 1)
            P_gv = np.concatenate([np.zeros((len(E_gv), K0)), pn_z_gv], 1)

            best = (-1, None)
            for a in GRID:
                for b in GRID:
                    for c in GRID:
                        sc_gv = a * L_gv + b * t80_z_gv + c * P_gv
                        for dd in DGRID:
                            acc = float(((sc_gv + dd * blk[None, :]).argmax(1)
                                         == y_gv).mean())
                            if acc > best[0]:
                                best = (acc, (a, b, c, dd))
            a, b, c, dd = best[1]
            sc = a * L_test + b * t80_z_test + c * P_test + dd * blk[None, :]
            hit = sc.argmax(1) == pos_y
            row = {'part': ps, 'sup': ss, 'k': k, 'gates': best[1],
                   'val': best[0],
                   'old': float(hit[m_old].mean()),
                   'new': float(hit[m_new].mean()),
                   'overall': float(hit[m_old | m_new].mean())}
            results.append(row)
            print(f'[{ps}/{ss}/k{k}] gates={best[1]} val={best[0]:.3f} '
                  f'old {row["old"]:.3f} new {row["new"]:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)

with open('../experiments/expansion_frontier.json', 'w') as f:
    json.dump(results, f, indent=1)

print('\n' + '=' * 68)
print('OURS (val-selected gates): mean +/- std over 9 combos')
print('=' * 68)
for k in KS:
    rows = [r for r in results if r['k'] == k]
    arr = np.array([[r['old'], r['new'], r['overall']] for r in rows])
    mu, sd = arr.mean(0), arr.std(0)
    print(f'k={k}: old {mu[0]*100:5.1f}±{sd[0]*100:4.1f}  '
          f'new {mu[1]*100:5.1f}±{sd[1]*100:4.1f}  '
          f'overall {mu[2]*100:5.1f}±{sd[2]*100:4.1f}')
print(f'total {time.time()-t0:.0f}s')
