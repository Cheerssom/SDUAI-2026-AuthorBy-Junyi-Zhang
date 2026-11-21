"""Issue #3: 3-seed pre-registered re-verification of the 90.42% TTE result.

Pre-registered BEFORE this run (docs/issue_tracker.md, 2026-09-22):
  gate grids:  tau in {0.0, 0.1, ..., 1.0};  wa, wb, wc each in
               {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0}
  seeds:       {0, 1, 2} varying LoRA training + both linear probes

Fixed across seeds (documented deviations):
  data split   (LT generation seed=0, train/val split seed=1) — benchmark is fixed
  text anchors T (deterministic, loaded from vitl_std.pt)
  frozen CLIP ViT-L view features (deterministic forward, extracted once)

Varying per seed:
  LoRA A-matrix init + resampling order + flip draws (torch.manual_seed)
  probe on LoRA features (sampler/init, generator seed s+1000)
  CLIP linear probe on cached jfit (sampler/init, generator seed s+2000)
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
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from linear_probe import zscore_rows
from lora_generic import LoRALinear, RANK, ALPHA, EPOCHS, BS, LR

device = 'cuda'
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
CL_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
CL_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
VIEWS = ['r224', 'r224f', 'c256', 'tl256', 'br256']

# ==== pre-registered search grids (do not change) ====
TAUS = [round(t, 2) for t in np.arange(0.0, 1.01, 0.1)]
WGRID = (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SEEDS = (0, 1, 2)


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
    store = {v: [] for v in VIEWS}
    for i in range(len(ds)):
        for v, t in make_views(ds.x[i]).items():
            store[v].append(t)
    return {v: torch.stack(ts) for v, ts in store.items()}


@torch.no_grad()
def clip_embed(X, mL, bs=128):
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device).float().div_(255.)
        x = (x - CL_MEAN) / CL_STD
        with torch.autocast('cuda', torch.float16):
            v = mL.vision_model(pixel_values=x)
            e = mL.visual_projection(v.pooler_output)
        out.append(F.normalize(e.float(), dim=1).cpu())
    return torch.cat(out)


def train_probe(Ftr, ytr, counts, n_cls, gen_seed, epochs=100):
    W = nn.Linear(Ftr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
    w = torch.tensor(1.0 / counts[ytr.numpy()], dtype=torch.double)
    smp = torch.utils.data.WeightedRandomSampler(w, len(Ftr), replacement=True)
    ds = TensorDataset(Ftr, ytr)
    g = torch.Generator().manual_seed(gen_seed)
    for ep in range(epochs):
        for xb, yb in DataLoader(ds, batch_size=256, sampler=smp, generator=g):
            loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return W.weight.cpu(), W.bias.cpu()


def main():
    import timm
    from lt_dataset import make_lt_datasets
    from train import split_train_val
    from transformers import CLIPModel
    t0 = time.time()

    train_eval, _, _, test, meta = make_lt_datasets(rho=100)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])
    BUCKET = np.where(counts > 100, 'head',
                      np.where(counts >= 20, 'medium', 'tail'))
    lp = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)
    y_fit = torch.tensor(fit.fine.astype(np.int64))
    y_val = val.fine.astype(np.int64)
    y_tst = test.fine.astype(np.int64)
    print(f'data ready: fit {len(fit)} val {len(val)} test {len(test)} '
          f'({time.time()-t0:.0f}s)', flush=True)

    Wd = torch.load('vitl_std.pt')
    T = Wd['T'].cpu()
    cache = torch.load('vitl_feats.pt')
    jfit, y_jfit = cache['jfit'], cache['yfit']

    print('caching 5-view tensors...', flush=True)
    Vf, Vv, Vt = cache_views(fit), cache_views(val), cache_views(test)
    print(f'views cached ({time.time()-t0:.0f}s)', flush=True)

    mL = CLIPModel.from_pretrained('openai/clip-vit-large-patch14'
                                   ).to(device).eval()
    E = {}
    for tag, Vd in (('fit', Vf), ('val', Vv), ('test', Vt)):
        E[tag] = {}
        for v in VIEWS:
            E[tag][v] = clip_embed(Vd[v], mL)
            print(f'  [clip-{tag}-{v}] done ({time.time()-t0:.0f}s)', flush=True)
    del mL
    torch.cuda.empty_cache()

    zs_val = {v: zscore_rows(E['val'][v] @ T.T).numpy() for v in VIEWS}
    zs_tst = {v: zscore_rows(E['test'][v] @ T.T).numpy() for v in VIEWS}
    sv2 = np.mean(np.stack([zs_val[v] for v in VIEWS]), axis=0)
    st2 = np.mean(np.stack([zs_tst[v] for v in VIEWS]), axis=0)

    model_name = 'vit_base_patch16_224.augreg_in21k'
    results = []
    for s in SEEDS:
        print(f'\n{"="*60}\nSEED {s}\n{"="*60}', flush=True)
        torch.manual_seed(s)
        m16 = timm.create_model(model_name, pretrained=True,
                                num_classes=0).to(device)
        for blk in m16.blocks:
            blk.attn.qkv = LoRALinear(blk.attn.qkv, RANK, ALPHA).to(device)
        head = nn.Linear(m16.embed_dim, 100).to(device)
        lora_params = [blk.attn.qkv.A for blk in m16.blocks] + \
                      [blk.attn.qkv.B for blk in m16.blocks]
        opt = torch.optim.AdamW(list(lora_params) + list(head.parameters()),
                                lr=LR, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        w = torch.tensor(1.0 / counts[fit.fine], dtype=torch.double)
        Xf, Xv = Vf['r224'], Vv['r224']
        Yf = y_fit

        @torch.no_grad()
        def lora_feats(X, bs=192):
            out = []
            for i in range(0, len(X), bs):
                x = X[i:i + bs].to(device).float().div_(255.)
                x = (x - IM_MEAN) / IM_STD
                with torch.autocast('cuda', torch.float16):
                    f = m16.forward_features(x)
                out.append(f[:, 0].float().cpu())
            return torch.cat(out)

        best_val, best_state = -1.0, None
        for ep in range(EPOCHS):
            m16.train(); head.train()
            perm = torch.multinomial(w, len(Xf), replacement=True)
            for i in range(0, len(perm) - BS + 1, BS):
                idx = perm[i:i + BS]
                x = Xf[idx].to(device).float().div_(255.)
                flip = torch.rand(x.shape[0], device=device) < 0.5
                x[flip] = torch.flip(x[flip], dims=[3])
                x = (x - IM_MEAN) / IM_STD
                y = Yf[idx].to(device)
                with torch.autocast('cuda', torch.float16):
                    loss = F.cross_entropy(head(m16.forward_features(x)
                                                [:, 0].float()), y)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            m16.eval(); head.eval()
            vhit = {}
            with torch.no_grad():
                fv = lora_feats(Xv)
                pred = (head(fv.to(device)).argmax(1)).cpu()
                for yi, pi in zip(val.fine, pred.tolist()):
                    vhit.setdefault(int(yi), []).append(int(yi == pi))
            vacc = float(np.mean([np.mean(v) for v in vhit.values()]))
            if vacc > best_val:
                best_val = vacc
                best_state = {k: v.clone() for k, v in m16.state_dict().items()
                              if k.endswith('.A') or k.endswith('.B')}
            print(f'  ep{ep+1}/{EPOCHS}: loss {loss.item():.3f} '
                  f'val {vacc:.4f} ({time.time()-t0:.0f}s)', flush=True)

        m16.load_state_dict(best_state, strict=False)
        m16.eval()
        del head
        torch.cuda.empty_cache()

        Ff = lora_feats(Xf)
        Ww, bb = train_probe(Ff, y_fit, counts, 100, gen_seed=s + 1000)
        Wf, bf = train_probe(jfit, y_jfit, counts, 100, gen_seed=s + 2000)

        def branch(Vd, Etag):
            b1, b3 = [], []
            for v in VIEWS:
                f16 = lora_feats(Vd[v])
                with torch.no_grad():
                    b1.append(zscore_rows(torch.tensor(
                        (f16 @ Ww.T + bb).numpy())).numpy())
                    b3.append(zscore_rows(torch.tensor(
                        (E[Etag][v] @ Wf.T + bf).numpy())).numpy())
            return (np.mean(np.stack(b1), axis=0),
                    np.mean(np.stack(b3), axis=0))

        sv1, sv3 = branch(Vv, 'val')
        st1, st3 = branch(Vt, 'test')
        del m16
        torch.cuda.empty_cache()

        def zt(a, tau):
            return zscore_rows(torch.tensor(a - tau * lp[None, :].numpy())
                               ).numpy()

        best = (-1.0, None)
        for tau in TAUS:
            a, c = zt(sv1, tau), zt(sv3, tau)
            for wa in WGRID:
                for wb in WGRID:
                    for wc in WGRID:
                        acc = float(((wa * a + wb * sv2 + wc * c).argmax(1)
                                     == y_val).mean())
                        if acc > best[0]:
                            best = (acc, (tau, wa, wb, wc))
        tau, wa, wb, wc = best[1]
        sc = (wa * zt(st1, tau) + wb * st2 + wc * zt(st3, tau))
        pred = sc.argmax(1)
        hit = pred == y_tst
        macro = float(np.mean([hit[y_tst == c].mean() for c in range(100)]))
        r = {'seed': s, 'val': best[0], 'gates': [tau, wa, wb, wc],
             'overall': float(hit.mean()), 'macro': macro}
        for bn in ('head', 'medium', 'tail'):
            m = BUCKET[y_tst] == bn
            r[bn] = float(hit[m].mean())
        results.append(r)
        print(f'SEED {s}: gates tau={tau} w=({wa},{wb},{wc}) '
              f'val {best[0]:.4f}', flush=True)
        print(f'  overall {r["overall"]:.4f}  macro {macro:.4f}  '
              f'head {r["head"]:.4f}  med {r["medium"]:.4f}  '
              f'tail {r["tail"]:.4f}', flush=True)

    print(f'\n{"="*60}')
    print(f'3-SEED PRE-REGISTERED VERIFICATION ({(time.time()-t0)/60:.0f} min)')
    print(f'{"="*60}')
    agg = {}
    for key in ('overall', 'macro', 'head', 'medium', 'tail'):
        vals = np.array([r[key] for r in results])
        agg[key] = {'mean': float(vals.mean()), 'std': float(vals.std())}
        print(f'  {key:>6}: {vals.mean()*100:.2f} +/- {vals.std()*100:.2f}')
    ref = np.mean([r['overall'] for r in results])
    print(f'  vs LIFT+ 89.70: {"ABOVE" if ref > 0.897 else "below"}; '
          f'vs 90.00: {"90+" if ref >= 0.90 else f"short {(0.90-ref)*100:.2f}"}')
    print(f'{"="*60}')

    with open('../experiments/seed_verify_results.json', 'w') as f:
        json.dump({'per_seed': results, 'aggregate': agg,
                   'grids': {'taus': TAUS, 'w': list(WGRID),
                             'seeds': list(SEEDS)}}, f, indent=2)
    print('saved ../experiments/seed_verify_results.json', flush=True)


if __name__ == '__main__':
    main()
