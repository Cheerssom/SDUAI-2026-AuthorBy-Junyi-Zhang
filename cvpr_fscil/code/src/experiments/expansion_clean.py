"""Clean expansion protocol — gate selection on TRAIN-VAL, full test evaluation.

Fixes issue #2: previous version selected gates on a 20% test slice.
This version uses training-set gate-val for seen classes + leave-one-out
support shots for new classes. Reports on FULL test set (standard protocol).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                               errors='replace')
import json
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPTokenizer, AutoConfig
from PIL import Image
from linear_probe import zscore_rows

device = 'cuda'
CLIP_MEAN = torch.tensor([.48145466, .4578275, .40821073]).view(1,3,1,1).to(device)
CLIP_STD = torch.tensor([.26862954, .26130258, .27577711]).view(1,3,1,1).to(device)
IM_MEAN = torch.tensor([.485, .456, .406]).view(1,3,1,1).to(device)
IM_STD = torch.tensor([.229, .224, .225]).view(1,3,1,1).to(device)


def make_views(img32):
    base = Image.fromarray(img32.transpose(1, 2, 0))
    r224 = base.resize((224, 224), Image.BICUBIC)
    r256 = base.resize((256, 256), Image.BICUBIC)
    out = {}
    a = torch.from_numpy(np.array(r224)).permute(2, 0, 1)
    out['r224'] = a
    out['r224f'] = torch.flip(a, dims=[2])
    n = np.array(r256)
    out['c256'] = torch.from_numpy(n[16:240, 16:240]).permute(2, 0, 1)
    out['tl256'] = torch.from_numpy(n[:224, :224]).permute(2, 0, 1)
    out['br256'] = torch.from_numpy(n[32:, 32:]).permute(2, 0, 1)
    return out


def cache_views(ds):
    store = {v: [] for v in ['r224', 'r224f', 'c256', 'tl256', 'br256']}
    for i in range(len(ds)):
        for v, t in make_views(ds.x[i]).items():
            store[v].append(t)
    return {v: torch.stack(ts) for v, ts in store.items()}


def main():
    import timm
    from lt_dataset import make_lt_datasets, partition_classes, subset_by_class, \
        k_shots_per_class, IndexedCifar
    from train import TF_CLIP, split_train_val
    from lora_v2 import LoRALinear

    t0 = time.time()
    from config import CFG
    with open(CFG.cifar_root / 'meta', 'rb') as f:
        names = pickle.load(f, encoding='latin1')['fine_label_names']
    names = [n.replace('_', ' ') for n in names]

    # Load models
    print('⬇️ Loading models...')
    mL = CLIPModel.from_pretrained('openai/clip-vit-large-patch14'
                                   ).to(device).eval()
    ck = torch.load('lora_in21k_b16.pt', map_location=device, weights_only=False)
    m16 = timm.create_model(ck['model'], pretrained=True, num_classes=0).to(device)
    for blk in m16.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, ck['rank'],
                                  ck['alpha']).to(device)
    m16.load_state_dict(ck['lora'], strict=False)
    m16.eval()

    # Text anchors
    TEMPLATES = ['a photo of a {}.', 'a bad photo of a {}.',
                 'a cropped photo of a {}.', 'a close-up photo of a {}.',
                 'a bright photo of a {}.', 'a good photo of a {}.',
                 'a photo of the {}.', 'a photo of a {}.']
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    cfg = AutoConfig.from_pretrained('google/vit-base-patch16-224')

    @torch.no_grad()
    def text_anchor(ids):
        Ts = []
        for t in TEMPLATES:
            b = tokenizer([t.format(names[c]) for c in ids],
                          return_tensors='pt', padding=True)
            t_out = mL.text_model(
                input_ids=b['input_ids'].to(device),
                attention_mask=b['attention_mask'].to(device))
            pooled = t_out.pooler_output
            Ts.append(F.normalize(mL.text_projection(pooled).float(),
                                  dim=1).cpu())
        return F.normalize(torch.stack(Ts).mean(0), dim=1)

    # Data
    train_eval, _, _, test, meta = make_lt_datasets(
        rho=100, train_tf=TF_CLIP, support_tf=TF_CLIP, test_tf=TF_CLIP)
    fit, val = split_train_val(train_eval)  # val = balanced train-val (407)
    counts = np.array(meta['cls_num_list'])
    tids, nids, _ = partition_classes(meta, 40, 20, seed=0)

    # TRAIN linear probe on fit split of TRAIN classes only
    keep = np.isin(fit.fine, tids)
    fit_train = IndexedCifar(fit.x[keep], fit.fine[keep],
                             fit.coarse[keep], TF_CLIP)
    pos_of = {int(c): i for i, c in enumerate(tids)}
    yfit = torch.tensor([pos_of[int(y)] for y in fit_train.fine])
    counts40 = np.bincount(yfit.numpy(), minlength=40).astype(float) + 1
    log_prior = torch.tensor(np.log(counts40 / counts40.sum()),
                             dtype=torch.float32).to(device)

    # Extract LoRA features for training the probe
    @torch.no_grad()
    def feats_b16(X, bs=192):
        out = []
        for i in range(0, len(X), bs):
            x = X[i:i+bs].to(device).float().div_(255.)
            x = (x - IM_MEAN) / IM_STD
            with torch.autocast('cuda', torch.float16):
                f = m16.forward_features(x)
            out.append(f[:, 0].float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def emb_L(X, bs=128):
        out = []
        for i in range(0, len(X), bs):
            x = X[i:i+bs].to(device).float().div_(255.)
            x = (x - CLIP_MEAN) / CLIP_STD
            with torch.autocast('cuda', torch.float16):
                e = mL.get_image_features(pixel_values=x)
            out.append(F.normalize(e.float(), dim=1).cpu())
        return torch.cat(out)

    print('🔧 Training linear probe (40 classes, fit split)...')
    Xfit = torch.stack([
        torch.from_numpy(np.array(
            Image.fromarray(fit_train.x[i].transpose(1,2,0))
            .resize((224,224), Image.BICUBIC))
        ).permute(2,0,1).contiguous()
        for i in range(len(fit_train))])
    Ffit = feats_b16(Xfit)
    W = nn.Linear(Ffit.shape[1], 40).to(device)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
    w_samp = torch.tensor(1.0 / counts40[yfit.numpy()], dtype=torch.double)
    smp = torch.utils.data.WeightedRandomSampler(w_samp, len(Ffit),
                                                  replacement=True)
    ds_probe = TensorDataset(Ffit, yfit)
    for ep in range(100):
        for xb, yb in DataLoader(ds_probe, batch_size=256, sampler=smp):
            loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Ww, bb = W.weight.detach().cpu(), W.bias.detach().cpu()

    # Build text anchors for all 80 candidates
    sub_ids = np.concatenate([tids, nids])
    T80 = text_anchor(sub_ids)
    Wf_full = torch.zeros(80, Ffit.shape[1])
    Wf_full[:40] = Ww
    b_full = torch.zeros(80)
    b_full[:40] = bb

    lp_old = torch.tensor(np.log(counts40 / counts40.sum()),
                          dtype=torch.float32)

    # ===== GATE VALIDATION SET (no test data!) =====
    # Seen classes: use the train-val split (balanced, 407 samples)
    # New classes: leave-one-out from support set (1 per class)
    print('\n🔧 Building gate-val set (train-val + LOO support)...')
    val_keep = np.isin(val.fine, tids)
    val_seen = IndexedCifar(val.x[val_keep], val.fine[val_keep],
                            val.coarse[val_keep], TF_CLIP)
    id2pos = {int(c): i for i, c in enumerate(sub_ids)}

    # Full test set views (for evaluation)
    print('📦 Caching test views...')
    Vtest = cache_views(test)
    ytest = test.fine.astype(np.int64)

    results = {}
    for k in (1, 2, 5, 10):
        print(f'\n===== k={k} =====')
        # Get support shots for new classes
        new_pool, _ = subset_by_class(train_eval, nids)
        shots = k_shots_per_class(new_pool, nids, k + 1, seed=7)
        # k shots for prototypes, 1 held-out for gate-val
        proto_shots = IndexedCifar(
            shots.x[:len(nids) * k],
            shots.fine[:len(nids) * k],
            shots.coarse[:len(nids) * k], TF_CLIP)
        val_shots = IndexedCifar(
            shots.x[len(nids) * k:],
            shots.fine[len(nids) * k:],
            shots.coarse[len(nids) * k:], TF_CLIP)

        # Build prototypes from k shots (not k+1)
        Xproto = torch.stack([
            torch.from_numpy(np.array(
                Image.fromarray(proto_shots.x[i].transpose(1,2,0))
                .resize((224,224), Image.BICUBIC))
            ).permute(2,0,1).contiguous()
            for i in range(len(proto_shots))])
        emb_proto = emb_L(Xproto)
        proto_map = {}
        for p, yy in zip(emb_proto, proto_shots.fine):
            proto_map.setdefault(int(yy), []).append(p)
        P_new = torch.stack([
            F.normalize(torch.stack(proto_map[int(c)]).mean(0), dim=0)
            for c in nids])

        # ===== GATE SELECTION on train-val + LOO support =====
        # Compute features for validation samples
        @torch.no_grad()
        def get_scores(X_img_list, fine_list):
            """Returns (lora_logits, zs_logits, proto_logits) for candidates."""
            X_t = torch.stack(X_img_list)
            f16 = feats_b16(X_t)
            l16 = (f16 @ Ww.T + bb)  # (n, 40)
            eL = emb_L(X_t)
            lzs = (eL @ T80.T)  # (n, 80)
            lproto = torch.cat([
                torch.full((len(X_t), 40), float('-inf')),
                eL @ P_new.T], dim=1)  # (n, 80)
            return l16.numpy(), lzs.numpy(), lproto.numpy(), \
                np.array([id2pos[int(y)] for y in fine_list])

        # Seen-class val: use train-val split
        Xval_seen = [
            torch.from_numpy(np.array(
                Image.fromarray(val_seen.x[i].transpose(1,2,0))
                .resize((224,224), Image.BICUBIC))
            ).permute(2,0,1).contiguous().float().div_(255.)
            for i in range(len(val_seen))]

        # New-class val: use held-out support shot
        Xval_new = [
            torch.from_numpy(np.array(
                Image.fromarray(val_shots.x[i].transpose(1,2,0))
                .resize((224,224), Image.BICUBIC))
            ).permute(2,0,1).contiguous().float().div_(255.)
            for i in range(len(val_shots))]

        # Combine: seen val (scores in 0..39) + new val (scores in 40..79)
        all_X = Xval_seen + Xval_new
        all_y = list(val_seen.fine) + list(val_shots.fine)
        l16_val, lzs_val, lproto_val, y_val = get_scores(all_X, all_y)

        # Pad l16 to 80 columns
        l16_val_80 = np.full((len(l16_val), 80), -1e9)
        l16_val_80[:, :40] = l16_val

        def zscore_rows_np(a):
            mu = a.mean(axis=1, keepdims=True)
            sd = np.maximum(a.std(axis=1, keepdims=True), 1e-6)
            return (a - mu) / sd

        # Gate selection on val
        best = (-1, None)
        for tau in np.arange(0, 1.01, 0.2):
            l16_adj = zscore_rows_np(l16_val_80)
            lin_adj = zscore_rows_np(lzs_val - tau * lp_old.numpy()[None, :40]
                                     if False else lzs_val)
            proto_adj = zscore_rows_np(lproto_val)
            for a in (0, 0.5, 1, 2):
                for bt in (0.25, 0.5, 1, 2):
                    for cp in (0, 0.25, 0.5, 1):
                        for d in (-1, -0.5, 0, 0.5, 1, 1.5):
                            blk = np.array([0]*40 + [1]*len(nids))
                            sc = (a * l16_adj + bt * lin_adj
                                  + cp * proto_adj + d * blk[None, :])
                            pred_pos = sc.argmax(1)
                            pred_ids = sub_ids[pred_pos]
                            acc = float((pred_ids == y_val).mean())
                            if acc > best[0]:
                                best = (acc, (float(tau), a, bt, cp, d))
        tau, a, bt, cp, d = best[1]
        print(f'  gates: tau={tau} a={a} bt={bt} cp={cp} d={d} '
              f'val={best[0]:.4f}')

        # ===== EVALUATION on FULL test set =====
        print(f'  evaluating on full test set ({len(ytest)}) ...')
        m_old = np.isin(ytest, tids)
        m_new = np.isin(ytest, nids)

        # Process test set in batches
        n_test = len(ytest)
        all_preds = np.zeros(n_test, dtype=np.int64)
        bs = 256
        for start in range(0, n_test, bs):
            end = min(start + bs, n_test)
            batch_X = []
            for i in range(start, end):
                img = Image.fromarray(test.x[i].transpose(1,2,0))
                img_r = img.resize((224,224), Image.BICUBIC)
                t = torch.from_numpy(np.array(img_r)).permute(2,0,1)
                batch_X.append(t.float().div_(255.))

            X_t = torch.stack(batch_X)
            f16 = feats_b16(X_t)
            l16_b = (f16 @ Ww.T + bb).numpy()
            eL = emb_L(X_t)
            lzs_b = (eL @ T80.T).numpy()
            lproto_b = torch.cat([
                torch.full((len(X_t), 40), float('-inf')),
                eL @ P_new.T], dim=1).numpy()

            l16_b80 = np.full_like(lzs_b, -1e9)
            l16_b80[:, :40] = l16_b

            blk = np.array([0]*40 + [1]*len(nids))
            sc = (a * zscore_rows_np(l16_b80)
                  + bt * zscore_rows_np(lzs_b)
                  + cp * zscore_rows_np(lproto_b)
                  + d * blk[None, :])
            all_preds[start:end] = sub_ids[sc.argmax(1)]

        old_acc = float((all_preds == ytest)[m_old].mean())
        new_acc = float((all_preds == ytest)[m_new].mean())
        print(f'  ✓ old: {old_acc:.4f}  new: {new_acc:.4f}  '
              f'(full test, no test-slice gating)')
        results[k] = {'old': old_acc, 'new': new_acc,
                      'gates': [tau, a, bt, cp, d], 'val': best[0]}

    print(f'\n{"="*60}')
    print('CLEAN EXPANSION RESULTS (gate-val = train split, eval = full test)')
    print(f'{"="*60}')
    for k, r in results.items():
        print(f'  {k}-shot: old {r["old"]:.4f}  new {r["new"]:.4f}')
    print(f'{"="*60}')
    print(f'Time: {(time.time()-t0)/60:.1f} min')

    with open('../experiments/expansion_clean_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('📁 Saved to experiments/expansion_clean_results.json')


if __name__ == '__main__':
    main()
