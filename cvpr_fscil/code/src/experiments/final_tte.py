"""TTE (5-view test-time ensemble) on the final 3-branch model.

Views: [224, 224-flip, center-224-of-256, TL-224-of-256, BR-224-of-256].
Per branch: extract features per view -> z-scored logits -> average over
views. Gates re-selected on val WITH TTE (protocol-clean), report on test.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from linear_probe import zscore_rows
from lora_generic import LoRALinear, RANK, ALPHA

device = 'cuda'
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
CL_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
CL_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
VIEWS = ['r224', 'r224f', 'c256', 'tl256', 'br256']


def make_views(img32):
    """uint8 CHW 32x32 -> dict of normalized view tensors (1,3,224,224)."""
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
    """Returns dict view -> (N,3,224,224) uint8 tensor."""
    store = {v: [] for v in VIEWS}
    for i in range(len(ds)):
        for v, t in make_views(ds.x[i]).items():
            store[v].append(t)
    return {v: torch.stack(ts) for v, ts in store.items()}


def main():
    import timm
    from lt_dataset import make_lt_datasets
    from train import split_train_val
    from transformers import CLIPModel
    t0 = time.time()
    # ---- branch 1: LoRA IN21k B/16 ----
    ck = torch.load('lora_in21k_b16.pt', map_location=device, weights_only=False)
    m16 = timm.create_model(ck['model'], pretrained=True, num_classes=0).cuda()
    for blk in m16.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, RANK, ALPHA).cuda()
    m16.load_state_dict(ck['lora'], strict=False)
    m16.eval()
    # ---- branch 2+3: frozen CLIP ViT-L ----
    mL = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda().eval()

    train_eval, _, _, test, meta = make_lt_datasets(rho=100)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])
    BUCKET = np.where(counts > 100, 'head', np.where(counts >= 20, 'medium', 'tail'))
    lp = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)
    Wd = torch.load('vitl_std.pt')
    T = Wd['T'].cpu()

    print('caching 5-view tensors...', flush=True)
    Vf = cache_views(fit)      # probe uses center view
    Vv = cache_views(val)
    Vt = cache_views(test)
    print(f'views cached ({time.time()-t0:.0f}s)', flush=True)

    @torch.no_grad()
    def feats_b16(X, bs=192):
        out = []
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(device).float().div_(255.)
            x = (x - IM_MEAN) / IM_STD
            with torch.autocast('cuda', torch.float16):
                t = m16.forward_features(x)
            out.append(t[:, 0].float().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def emb_L(X, bs=128):
        out = []
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(device).float().div_(255.)
            x = (x - CL_MEAN) / CL_STD
            with torch.autocast('cuda', torch.float16):
                e = mL.get_image_features(pixel_values=x)
            out.append(F.normalize(e.float(), dim=1).cpu())
        return torch.cat(out)

    # probe on fit (center view), as in final_report
    Ff = feats_b16(Vf['r224'])
    Yf = torch.tensor(fit.fine.astype(np.int64))
    W = nn.Linear(Ff.shape[1], 100).to(device)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
    w = torch.tensor(1.0 / counts[fit.fine], dtype=torch.double)
    smp = torch.utils.data.WeightedRandomSampler(w, len(Ff), replacement=True)
    dstr = TensorDataset(Ff, Yf)
    g = torch.Generator().manual_seed(0)
    for ep in range(100):
        for xb, yb in DataLoader(dstr, batch_size=256, sampler=smp, generator=g):
            loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Ww, bb = W.weight.cpu(), W.bias.cpu()

    # frozen ViT-L linear branch (from cached single-view vitl run, reuse Wf)
    Wf, bf = Wd['W'], Wd['b']

    def branch_scores(Vd, tag):
        """per-branch logits averaged over views (z-score per view)."""
        zs = {1: [], 2: [], 3: []}
        for v in VIEWS:
            X = Vd[v]
            f16 = feats_b16(X)
            eL = emb_L(X)
            with torch.no_grad():
                l16 = (f16 @ Ww.T + bb).numpy()
            lzs = (eL @ T.T).numpy()
            llr = (eL @ Wf.T + bf).numpy()
            zs[1].append(zscore_rows(torch.tensor(l16)).numpy())
            zs[2].append(zscore_rows(torch.tensor(lzs)).numpy())
            zs[3].append(zscore_rows(torch.tensor(llr)).numpy())
            print(f'  [{tag}] view {v} done ({time.time()-t0:.0f}s)', flush=True)
        return {k: np.mean(np.stack(z), axis=0) for k, z in zs.items()}

    sv = branch_scores(Vv, 'val')
    st = branch_scores(Vt, 'test')
    yv = val.fine.astype(np.int64)
    yt = test.fine.astype(np.int64)

    def adj(a, tau):
        return a - tau * lp[None, :].numpy() if False else a

    # gates on val WITH TTE (fine grid); tau applies pre-zscore -> we z-scored
    # already per view, so emulate tau by subtracting tau*lp then re-zscoring
    def zt(a, tau):
        return zscore_rows(torch.tensor(a - tau * lp[None, :].numpy())).numpy()

    best = (-1.0, None)
    for tau in np.arange(0.0, 1.01, 0.2):
        a, c = zt(sv[1], float(tau)), zt(sv[3], float(tau))
        b = sv[2]
        for wa in (0, 0.5, 1, 1.5, 2, 3):
            for wb in (0, 0.25, 0.5, 0.75, 1, 1.5):
                for wc in (0, 0.25, 0.5, 0.75, 1, 1.5):
                    acc = float(((wa * a + wb * b + wc * c).argmax(1)
                                 == yv).mean())
                    if acc > best[0]:
                        best = (acc, (float(tau), wa, wb, wc))
    tau, wa, wb, wc = best[1]
    sc = (wa * zt(st[1], tau) + wb * st[2] + wc * zt(st[3], tau))
    pred = sc.argmax(1)
    hit = pred == yt
    b_of_y = BUCKET[yt]
    macro = float(np.mean([hit[yt == c].mean() for c in range(100)]))
    print('=== FINAL + TTE (5 views) ===')
    print(f'gates tau={tau:.1f} w=({wa},{wb},{wc})  val {best[0]:.4f}')
    print(f'overall {hit.mean():.4f}  macro {macro:.4f}')
    for bn in ('head', 'medium', 'tail'):
        m = b_of_y == bn
        print(f'{bn:>6}: {hit[m].mean():.4f}')
    print(f'LIFT+ TTE 89.70 -> '
          f'{"ABOVE" if hit.mean() > 0.897 else "below"}; '
          f'90.00 -> {"*** 90+ ***" if hit.mean() >= 0.90 else f"short {(0.90-hit.mean())*100:.2f}"}')


if __name__ == '__main__':
    main()
