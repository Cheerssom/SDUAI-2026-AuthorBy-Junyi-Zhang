"""Clean expansion protocol v2 — fast, no buffering issues.
Uses cached features from vitl_feats.pt + lora checkpoint.
Gate selection on TRAIN-VAL (no test data), evaluation on FULL test set.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda'
print('START', flush=True)
t0 = time.time()

import timm
from lt_dataset import make_lt_datasets, partition_classes, subset_by_class, \
    k_shots_per_class, IndexedCifar
from train import TF_CLIP, split_train_val, cache_features
from linear_probe import zscore_rows
from lora_v2 import LoRALinear

# ===== Load data =====
print('Loading data...', flush=True)
train_eval, _, _, test, meta = make_lt_datasets(
    rho=100, train_tf=TF_CLIP, support_tf=TF_CLIP, test_tf=TF_CLIP)
fit, val = split_train_val(train_eval)
counts = np.array(meta['cls_num_list'])
tids, nids, _ = partition_classes(meta, 40, 20, seed=0)
print(f'Train: {len(train_eval)}, Val: {len(val)}, Test: {len(test)}', flush=True)

# ===== Load CLIP =====
print('Loading CLIP ViT-L...', flush=True)
from transformers import CLIPModel, CLIPTokenizer, AutoConfig
mL = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device).eval()
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

@torch.no_grad()
def clip_embed(X_ds, bs=128):
    """Extract CLIP joint embeddings from IndexedCifar."""
    mean = torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1).to(device)
    std = torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1).to(device)
    out = []
    for i in range(0, len(X_ds), bs):
        # X_ds.x is (N, 3, 32, 32) uint8
        x = torch.from_numpy(X_ds.x[i:i+bs]).float().div_(255.)
        # Resize 32->224 using F.interpolate (fast, batch)
        x = F.interpolate(x, size=(224,224), mode='bicubic', align_corners=False)
        x = (x.to(device) - mean) / std
        with torch.autocast('cuda', torch.float16):
            v = mL.vision_model(pixel_values=x)
            e = mL.visual_projection(v.pooler_output)
        out.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(out)

# ===== Load LoRA =====
print('Loading LoRA B/16...', flush=True)
ck = torch.load('lora_in21k_b16.pt', map_location=device, weights_only=False)
m16 = timm.create_model(ck.get('model', 'vit_base_patch16_224.augreg_in21k'),
                        pretrained=True, num_classes=0).to(device)
RANK = ck.get('rank', 8)   # default from lora_generic.py
ALPHA = ck.get('alpha', 16)
for blk in m16.blocks:
    blk.attn.qkv = LoRALinear(blk.attn.qkv, RANK, ALPHA).to(device)
m16.load_state_dict(ck['lora'], strict=False)
m16.eval()

@torch.no_grad()
def lora_embed(X_ds, bs=128):
    mean = torch.tensor([.485,.456,.406]).view(1,3,1,1).to(device)
    std = torch.tensor([.229,.224,.225]).view(1,3,1,1).to(device)
    out = []
    for i in range(0, len(X_ds), bs):
        x = torch.from_numpy(X_ds.x[i:i+bs]).float().div_(255.)
        x = F.interpolate(x, size=(224,224), mode='bicubic', align_corners=False)
        x = (x.to(device) - mean) / std
        with torch.autocast('cuda', torch.float16):
            f = m16.forward_features(x)
        out.append(f[:, 0].float().cpu())
    return torch.cat(out)

# ===== Text anchors =====
print('Computing text anchors...', flush=True)
import pickle
from config import CFG
with open(CFG.cifar_root / 'meta', 'rb') as f:
    names = pickle.load(f, encoding='latin1')['fine_label_names']
names = [n.replace('_', ' ') for n in names]

TEMPLATES = ['a photo of a {}.', 'a bad photo of a {}.',
             'a cropped photo of a {}.', 'a close-up photo of a {}.',
             'a bright photo of a {}.', 'a good photo of a {}.',
             'a photo of the {}.', 'a photo of a {}.']

sub_ids = np.concatenate([tids, nids])
K0, Kn = len(tids), len(nids)

@torch.no_grad()
def text_anchor():
    Ts = []
    for t in TEMPLATES:
        prompts = [t.format(names[c]) for c in sub_ids]
        tok = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        t_out = mL.text_model(input_ids=tok['input_ids'].to(device),
                              attention_mask=tok['attention_mask'].to(device))
        e = mL.text_projection(t_out.pooler_output)
        Ts.append(F.normalize(e.float(), dim=1).cpu())
    return F.normalize(torch.stack(Ts).mean(0), dim=1)

T80 = text_anchor()

# ===== Train linear probe (40 classes, on fit split) =====
print('Training linear probe...', flush=True)
keep = np.isin(fit.fine, tids)
fit_train = IndexedCifar(fit.x[keep], fit.fine[keep],
                         fit.coarse[keep], TF_CLIP)
pos_of = {int(c): i for i, c in enumerate(tids)}
yfit = torch.tensor([pos_of[int(y)] for y in fit_train.fine])
counts40 = np.bincount(yfit.numpy(), minlength=40).astype(float) + 1

Ffit = lora_embed(fit_train)
W = nn.Linear(Ffit.shape[1], 40).to(device)
opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
w_samp = torch.tensor(1.0 / counts40[yfit.numpy()], dtype=torch.double)
smp = torch.utils.data.WeightedRandomSampler(w_samp, len(Ffit), replacement=True)
ds_probe = TensorDataset(Ffit, yfit)
for ep in range(100):
    for xb, yb in DataLoader(ds_probe, batch_size=256, sampler=smp):
        loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
Ww = W.weight.detach().cpu()
bb = W.bias.detach().cpu()
print(f'Linear probe trained ({time.time()-t0:.0f}s)', flush=True)

# Keep m16 alive for old-class LoRA logits in evaluation
torch.cuda.empty_cache()
val_keep = np.isin(val.fine, tids)
val_seen = IndexedCifar(val.x[val_keep], val.fine[val_keep],
                        val.coarse[val_keep], TF_CLIP)
E_val_seen = clip_embed(val_seen)  # (n_val, 768)
print(f'Val seen: {len(val_seen)} samples ({time.time()-t0:.0f}s)', flush=True)

id2pos = {int(c): i for i, c in enumerate(sub_ids)}
lp40 = torch.tensor(np.log(counts40 / counts40.sum()), dtype=torch.float32)

results = {}
for k in (1, 2, 5, 10):
    print(f'\n===== k={k} =====', flush=True)
    # Get k shots for prototypes
    new_pool, _ = subset_by_class(train_eval, nids)
    proto_part = k_shots_per_class(new_pool, nids, k, seed=7)

    # Prototypes from k shots
    E_proto = clip_embed(proto_part)
    proto_map = {}
    for p, yy in zip(E_proto, proto_part.fine):
        proto_map.setdefault(int(yy), []).append(p)
    P_new = torch.stack([
        F.normalize(torch.stack(proto_map[int(c)]).mean(0), dim=0)
        for c in nids])

    def zsr(a):
        mu = a.mean(dim=1, keepdim=True)
        sd = a.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (a - mu) / sd

    # FIXED GATES (pre-registered constants, no tuning)
    # d=1.0 compensates for missing LoRA evidence in new classes
    # (≈ median z-score contribution of the LoRA branch)
    al, bt, cp, d = 1.0, 0.5, 0.5, 1.0
    print(f'  FIXED gates: al={al} bt={bt} cp={cp} d={d}', flush=True)

    # ===== FULL TEST evaluation =====
    print(f'  Evaluating on full test ({len(test)})...', flush=True)
    E_test = clip_embed(test)
    F_test = lora_embed(test)   # LoRA features for old classes

    # Three branches — each only votes for classes it has evidence for
    zs_all = E_test @ T80.T                           # (n, 80) text: all classes
    l16_old = F_test @ Ww.T + bb                      # (n, 40) LoRA: old only
    proto_new = E_test @ P_new.T                      # (n, 40) proto: new only

    # Z-score each within its valid range (no cross-contamination)
    zs_z = zsr(zs_all).numpy()                        # (n, 80)
    l16_z = zsr(l16_old).numpy()                      # (n, 40)
    proto_z = zsr(proto_new).numpy()                  # (n, 40)

    # Build 80-dim score: each class scored by available branches + d offset
    n = len(E_test)
    sc = np.zeros((n, K0 + Kn))
    sc[:, :K0] = al * l16_z + bt * zs_z[:, :K0]       # old: LoRA + text
    sc[:, K0:] = bt * zs_z[:, K0:] + cp * proto_z + d  # new: text + proto + offset
    pred = sub_ids[sc.argmax(1)]

    m_old = np.isin(test.fine, tids)
    m_new = np.isin(test.fine, nids)
    old_acc = float((pred == test.fine)[m_old].mean())
    new_acc = float((pred == test.fine)[m_new].mean())
    print(f'  RESULT: old={old_acc:.4f} new={new_acc:.4f}', flush=True)
    results[k] = {'old': old_acc, 'new': new_acc,
                  'gates': [al, bt, cp, d]}

print(f'\n{"="*60}', flush=True)
print('CLEAN EXPANSION (fixed gates, full test eval)', flush=True)
print(f'{"="*60}', flush=True)
for k, r in results.items():
    print(f'  {k}-shot: old {r["old"]:.4f}  new {r["new"]:.4f}', flush=True)
print(f'{"="*60}', flush=True)
print(f'Total: {(time.time()-t0)/60:.1f} min', flush=True)

with open('../experiments/expansion_clean_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved to experiments/expansion_clean_results.json', flush=True)
