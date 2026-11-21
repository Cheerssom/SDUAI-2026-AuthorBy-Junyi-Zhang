"""TTE ensemble with LoRA v2 (joint head + balanced softmax) + frozen branches."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from linear_probe import zscore_rows
from lora_v2 import LoRALinear, RANK, ALPHA

device = 'cuda'
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
CL_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).cuda()
CL_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).cuda()
VIEWS = ['r224', 'r224f', 'c256', 'tl256', 'br256']


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


def main():
    import timm
    from lt_dataset import make_lt_datasets
    from train import split_train_val
    from transformers import CLIPModel
    t0 = time.time()
    # LoRA v2 model
    ck = torch.load('lora_b16_v2.pt', map_location=device, weights_only=False)
    m16 = timm.create_model('vit_base_patch16_224.augreg_in21k',
                            pretrained=True, num_classes=0).cuda()
    for blk in m16.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, ck['rank'],
                                  ck['alpha']).cuda()
    m16.load_state_dict(ck['lora'], strict=False)
    m16.eval()
    # v2 joint head
    dim = m16.embed_dim
    W2 = nn.Linear(dim, 100).cuda()
    W2.load_state_dict(ck['head'])
    W2.eval()
    # frozen ViT-L
    mL = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda().eval()

    train_eval, _, _, test, meta = make_lt_datasets(rho=100)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])
    BUCKET = np.where(counts > 100, 'head',
                      np.where(counts >= 20, 'medium', 'tail'))
    lp = torch.tensor(np.log(counts / counts.sum()),
                      dtype=torch.float32).cpu()
    Wd = torch.load('vitl_std.pt')
    T = Wd['T'].cpu()
    Wf, bf = Wd['W'], Wd['b']

    print('caching views...', flush=True)
    Vv = cache_views(val)
    Vt = cache_views(test)
    print(f'cached ({time.time()-t0:.0f}s)', flush=True)

    @torch.no_grad()
    def all_branch_scores(X):
        """Returns (v2_lora_logits, zs_logits, lin_logits) for one view."""
        x16 = X.to(device).float().div_(255.)
        x16n = (x16 - IM_MEAN) / IM_STD
        with torch.autocast('cuda', torch.float16):
            f16 = m16.forward_features(x16n)[:, 0]
        l_v2 = W2(f16.float()).cpu()          # joint head, no prior adj
        xL = (x16 - CL_MEAN) / CL_STD
        with torch.autocast('cuda', torch.float16):
            eL = mL.get_image_features(pixel_values=xL)
        eL = F.normalize(eL.float(), dim=1).cpu()
        l_zs = (eL @ T.T)
        l_lin = (eL @ Wf.T + bf)
        return l_v2, l_zs, l_lin

    # average z-scored logits across views per branch
    def tte_scores(Vd, tag):
        acc = {1: [], 2: [], 3: []}
        for v in VIEWS:
            X = Vd[v]
            a, b, c = [], [], []
            for i in range(0, len(X), 192):
                l_v2, l_zs, l_lin = all_branch_scores(X[i:i + 192])
                a.append(l_v2); b.append(l_zs); c.append(l_lin)
            acc[1].append(zscore_rows(torch.cat(a)).numpy())
            acc[2].append(zscore_rows(torch.cat(b)).numpy())
            acc[3].append(zscore_rows(torch.cat(c)).numpy())
            print(f'  [{tag}] {v} done ({time.time()-t0:.0f}s)', flush=True)
        return {k: np.mean(np.stack(z), axis=0) for k, z in acc.items()}

    sv = tte_scores(Vv, 'val')
    st = tte_scores(Vt, 'test')
    yv = val.fine.astype(np.int64)
    yt = test.fine.astype(np.int64)

    def zt(a, tau):
        return zscore_rows(torch.tensor(a - tau * lp[None, :].numpy())).numpy()

    best = (-1.0, None)
    for tau in np.arange(0.0, 1.01, 0.2):
        for wa in (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3):
            for wb in (0, 0.25, 0.5, 0.75, 1, 1.5):
                for wc in (0, 0.25, 0.5, 0.75, 1, 1.5):
                    a = zt(sv[1], float(tau))
                    c = zt(sv[3], float(tau))
                    b = sv[2]
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
    top5 = torch.tensor(sc).topk(5, 1).indices.numpy()
    hit5 = np.array([yt[i] in top5[i] for i in range(len(yt))])

    print('=== LoRA v2 + TTE ensemble ===')
    print(f'gates tau={tau:.1f} w=({wa},{wb},{wc})  val {best[0]:.4f}')
    print(f'overall {hit.mean():.4f}  top5 {hit5.mean():.4f}  '
          f'macro {macro:.4f}')
    for bn in ('head', 'medium', 'tail'):
        m = b_of_y == bn
        print(f'{bn:>6}: {hit[m].mean():.4f}')
    print(f'LIFT+ 89.70 -> {"ABOVE" if hit.mean() > 0.897 else "below"}')
    print(f'prev best 90.42 -> {"NEW BEST" if hit.mean() > 0.9042 else "no"}')
    print(f'90+ -> {"yes" if hit.mean() >= 0.90 else "no"}')


if __name__ == '__main__':
    main()
