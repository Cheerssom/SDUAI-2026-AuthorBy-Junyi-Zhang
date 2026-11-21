"""Expansion baselines under identical partitions and support shots (audit #1+#3).

Compares, at the same class partition / support budget:
  ZS          CLIP zero-shot text anchors (80-way)
  NCM         prototypes only (old = fit class means, new = k-shot means)
  TipAdapter  text logits + training-free feature cache (beta=5.95, alpha=1)
  SimpleFusion 0.5*cos_text + 0.5*cos_proto, raw, no z-score, no offset
  Ours        per-branch z-scored voting, fixed pre-registered gates + d=1.0
3 class partitions x 3 support seeds x k in {1,2,5,10}; reports old/new/overall.
Also records old-class flip rates (closed-set correct -> expanded wrong) at k=5.
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

import timm
from lt_dataset import make_lt_datasets, partition_classes, subset_by_class, \
    k_shots_per_class, IndexedCifar
from train import TF_CLIP, split_train_val
from lora_v2 import LoRALinear

print('loading data + models...', flush=True)
train_eval, _, _, test, meta = make_lt_datasets(
    rho=100, train_tf=TF_CLIP, support_tf=TF_CLIP, test_tf=TF_CLIP)
fit, val = split_train_val(train_eval)
counts = np.array(meta['cls_num_list'])

import pickle
from config import CFG
with open(CFG.cifar_root / 'meta', 'rb') as f:
    names = [n.replace('_', ' ') for n in
             pickle.load(f, encoding='latin1')['fine_label_names']]

from transformers import CLIPModel, CLIPTokenizer
mL = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device).eval()
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
ck = torch.load('lora_in21k_b16.pt', map_location=device, weights_only=False)
m16 = timm.create_model(ck.get('model', 'vit_base_patch16_224.augreg_in21k'),
                        pretrained=True, num_classes=0).to(device)
for blk in m16.blocks:
    blk.attn.qkv = LoRALinear(blk.attn.qkv, ck.get('rank', 8),
                              ck.get('alpha', 16)).to(device)
m16.load_state_dict(ck['lora'], strict=False)
m16.eval()

CL_MEAN = torch.tensor([.48145466, .4578275, .40821073]).view(1, 3, 1, 1).to(device)
CL_STD = torch.tensor([.26862954, .26130258, .27577711]).view(1, 3, 1, 1).to(device)
IM_MEAN = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1).to(device)
IM_STD = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1).to(device)


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


TEMPLATES = ['a photo of a {}.', 'a bad photo of a {}.',
             'a cropped photo of a {}.', 'a close-up photo of a {}.',
             'a bright photo of a {}.', 'a good photo of a {}.',
             'a photo of the {}.', 'a photo of a {}.']


@torch.no_grad()
def text_anchor_all():
    Ts = []
    for t in TEMPLATES:
        prompts = [t.format(n) for n in names]
        tok = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        out = mL.text_model(input_ids=tok['input_ids'].to(device),
                            attention_mask=tok['attention_mask'].to(device))
        e = mL.text_projection(out.pooler_output)
        Ts.append(F.normalize(e.float(), dim=1).cpu())
    return F.normalize(torch.stack(Ts).mean(0), dim=1)


def zsr(a):
    mu = a.mean(dim=1, keepdim=True)
    sd = a.std(dim=1, keepdim=True).clamp(min=1e-6)
    return (a - mu) / sd


def zsr_np(a):
    mu = a.mean(axis=1, keepdims=True)
    sd = np.maximum(a.std(axis=1, keepdims=True), 1e-6)
    return (a - mu) / sd


if CACHE.exists():
    d = torch.load(CACHE)
    E_fit, F_fit, E_val, F_val, E_test, F_test, T100 = (
        d['E_fit'], d['F_fit'], d['E_val'], d['F_val'],
        d['E_test'], d['F_test'], d['T100'])
    print(f'cache loaded ({time.time()-t0:.0f}s)', flush=True)
else:
    print('embedding fit/val/test (one-time)...', flush=True)
    E_fit, F_fit = clip_embed(fit), lora_embed(fit)
    E_val, F_val = clip_embed(val), lora_embed(val)
    E_test, F_test = clip_embed(test), lora_embed(test)
    T100 = text_anchor_all()
    torch.save({'E_fit': E_fit, 'F_fit': F_fit, 'E_val': E_val, 'F_val': F_val,
                'E_test': E_test, 'F_test': F_test, 'T100': T100}, CACHE)
    print(f'cache built ({time.time()-t0:.0f}s)', flush=True)

y_test = test.fine.astype(np.int64)
fit_fine = fit.fine.astype(np.int64)
val_fine = val.fine.astype(np.int64)

PART_SEEDS = (0, 1, 2)
SUP_SEEDS = (7, 8, 9)
KS = (1, 2, 5, 10)
BETA, ALPHA = 5.95, 1.0     # Tip-Adapter training-free defaults
CAP_OLD = 20                # Tip-Adapter cache cap per old class

results = []
preds_archive = None

for ps in PART_SEEDS:
    tids, nids, _ = partition_classes(meta, 40, 20, seed=ps)
    K0, Kn = len(tids), len(nids)
    sub_ids = np.concatenate([tids, nids])
    pos_of = {int(c): i for i, c in enumerate(sub_ids)}
    T80 = T100[sub_ids]
    m_old = np.isin(y_test, tids)
    m_new = np.isin(y_test, nids)
    # NEG classes map to -1: they can never match an 80-way prediction
    pos_y = np.full(len(y_test), -1, dtype=np.int64)
    for i, y in enumerate(y_test):
        p = pos_of.get(int(y))
        if p is not None:
            pos_y[i] = p

    keep = np.isin(fit_fine, tids)
    Ff, yf = F_fit[keep], torch.tensor(
        [pos_of[int(y)] for y in fit_fine[keep]])
    counts40 = np.bincount(yf.numpy(), minlength=K0).astype(float) + 1
    W = nn.Linear(Ff.shape[1], K0).to(device)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
    w_samp = torch.tensor(1.0 / counts40[yf.numpy()], dtype=torch.double)
    smp = torch.utils.data.WeightedRandomSampler(w_samp, len(Ff), replacement=True)
    ds = TensorDataset(Ff, yf)
    g = torch.Generator().manual_seed(0)
    for ep in range(100):
        for xb, yb in DataLoader(ds, batch_size=256, sampler=smp, generator=g):
            loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Ww, bb = W.weight.cpu(), W.bias.cpu()
    print(f'[part {ps}] probe done ({time.time()-t0:.0f}s)', flush=True)

    l_test = (F_test @ Ww.T + bb)                     # (n,40)
    t_test = (E_test @ T80.T)                          # (n,80)
    l40_z = zsr_np(l_test.numpy())
    t80_z = zsr_np(t_test.numpy())

    # old-class closed-set prediction (our method, 40-way) for flip stats
    sc40 = 1.0 * l40_z + 0.5 * t80_z[:, :K0]
    pred40 = sub_ids[sc40.argmax(1)]

    # old prototypes in CLIP space (fit class means)
    keep_fit_ids = np.isin(fit_fine, sub_ids)
    proto_old = torch.stack([
        F.normalize(E_fit[fit_fine == c].mean(0), dim=0) for c in tids])

    new_pool, _ = subset_by_class(train_eval, nids)
    for ss in SUP_SEEDS:
        rng_cache = np.random.RandomState(ss)
        for k in KS:
            shots = k_shots_per_class(new_pool, nids, k, seed=ss)
            E_shot = clip_embed(shots)
            proto_new = torch.stack([
                F.normalize(E_shot[shots.fine == c].mean(0), dim=0) for c in nids])
            P_new = proto_new

            # old-class NCM prototypes from fit means (same CLIP space)
            P80 = torch.cat([proto_old, P_new])
            cos_p = (E_test @ P80.T).numpy()

            row = {'part': ps, 'sup': ss, 'k': k}
            # --- ZS ---
            p = t_test.argmax(1).numpy()
            hit = p == pos_y
            row['ZS'] = [float(hit[m_old].mean()), float(hit[m_new].mean()),
                         float(hit[m_old | m_new].mean())]
            # --- NCM ---
            p = cos_p.argmax(1)
            hit = p == pos_y
            row['NCM'] = [float(hit[m_old].mean()), float(hit[m_new].mean()),
                          float(hit[m_old | m_new].mean())]
            # --- Tip-Adapter ---
            keys, vals = [], []
            for i, c in enumerate(tids):
                idx = np.where(fit_fine == c)[0]
                sel = rng_cache.choice(idx, size=min(len(idx), CAP_OLD),
                                       replace=False)
                keys.append(E_fit[sel])
                vals += [i] * len(sel)
            for i, c in enumerate(nids):
                idx = np.where(shots.fine == c)[0]
                keys.append(E_shot[idx])
                vals += [K0 + i] * len(idx)
            A = torch.cat(keys)
            V = torch.zeros(len(A), K0 + Kn)
            for r, v in enumerate(vals):
                V[r, v] = 1.0
            aff = E_test @ A.T
            cache_logits = (-BETA * (1.0 - aff)) @ V * ALPHA
            tip = 100.0 * t_test + cache_logits
            p = tip.argmax(1).numpy()
            hit = p == pos_y
            row['TipAdapter'] = [float(hit[m_old].mean()),
                                 float(hit[m_new].mean()),
                                 float(hit[m_old | m_new].mean())]
            # --- SimpleFusion: 0.5 raw cos text + 0.5 raw cos proto ---
            sc = 0.5 * t_test.numpy() + 0.5 * cos_p
            p = sc.argmax(1)
            hit = p == pos_y
            row['SimpleFusion'] = [float(hit[m_old].mean()),
                                   float(hit[m_new].mean()),
                                   float(hit[m_old | m_new].mean())]
            # --- Ours: per-branch z-score voting, fixed gates, d=1.0 ---
            proto_new_z = zsr_np((E_test @ P_new.T).numpy())
            sc = np.zeros((len(E_test), K0 + Kn))
            sc[:, :K0] = 1.0 * l40_z + 0.5 * t80_z[:, :K0]
            sc[:, K0:] = 0.5 * t80_z[:, K0:] + 0.5 * proto_new_z + 1.0
            p = sc.argmax(1)
            hit = p == pos_y
            row['Ours'] = [float(hit[m_old].mean()), float(hit[m_new].mean()),
                           float(hit[m_old | m_new].mean())]
            results.append(row)

            if k == 5:
                pred80 = sub_ids[sc.argmax(1)]
                was_right = pred40 == y_test
                now_wrong = pred80 != y_test
                row['flip_wrong'] = float((was_right & now_wrong)[m_old].mean())
                row['flip_right'] = float(((~was_right) & (~now_wrong))[m_old].mean())
                if ps == 0 and ss == 7:
                    preds_archive = {'pred40': pred40, 'pred80': pred80,
                                     'y': y_test, 'tids': tids, 'nids': nids}
            print(f'  [part {ps} sup {ss} k {k}] '
                  f'ZS {row["ZS"][1]:.3f} NCM {row["NCM"][1]:.3f} '
                  f'Tip {row["TipAdapter"][1]:.3f} SF {row["SimpleFusion"][1]:.3f} '
                  f'Ours {row["Ours"][1]:.3f} (new) '
                  f'({time.time()-t0:.0f}s)', flush=True)

with open('../experiments/expansion_baselines.json', 'w') as f:
    json.dump(results, f, indent=1)
if preds_archive is not None:
    np.savez('../experiments/expansion_preds_k5_p0s7.npz', **preds_archive)

# ---- summary: mean +/- std over the 9 combos, per method per k ----
print('\n' + '=' * 72)
print('EXPANSION BASELINES (mean +/- std over 3 partitions x 3 support seeds)')
print('=' * 72)
methods = ['ZS', 'NCM', 'TipAdapter', 'SimpleFusion', 'Ours']
for k in KS:
    print(f'k={k}')
    rows = [r for r in results if r['k'] == k]
    for mth in methods:
        arr = np.array([r[mth] for r in rows])        # (9,3) old,new,overall
        mu, sd = arr.mean(0), arr.std(0)
        print(f'  {mth:>13}: old {mu[0]*100:5.1f}±{sd[0]*100:4.1f}  '
              f'new {mu[1]*100:5.1f}±{sd[1]*100:4.1f}  '
              f'overall {mu[2]*100:5.1f}±{sd[2]*100:4.1f}')
fl = [r for r in results if r['k'] == 5]
fw = np.array([r['flip_wrong'] for r in fl])
fr = np.array([r['flip_right'] for r in fl])
print(f'flip (k=5, old classes): correct->wrong {fw.mean()*100:.2f}±{fw.std()*100:.2f}%  '
      f'wrong->correct {fr.mean()*100:.2f}±{fr.std()*100:.2f}%')
print(f'total {time.time()-t0:.0f}s')
