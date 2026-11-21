"""Final ensemble: LoRA IN21k-B/16 features + frozen CLIP ViT-L branches."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
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
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


def main():
    import timm
    from lt_dataset import make_lt_datasets
    from train import split_train_val
    t0 = time.time()
    ck = torch.load('lora_in21k_b16.pt', map_location=device, weights_only=False)
    model = timm.create_model(ck['model'], pretrained=True, num_classes=0).cuda()
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, RANK, ALPHA).cuda()
    model.load_state_dict(ck['lora'], strict=False)
    model.eval()
    print(f'LoRA B/16 loaded (train val was {ck["val"]:.4f})', flush=True)

    train_eval, _, _, test, meta = make_lt_datasets(rho=100)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])
    BUCKET = np.where(counts > 100, 'head',
                      np.where(counts >= 20, 'medium', 'tail'))
    lp = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)

    def cache(ds):
        xs = []
        for i in range(len(ds)):
            img = Image.fromarray(ds.x[i].transpose(1, 2, 0))
            img = img.resize((224, 224), Image.BICUBIC)
            xs.append(torch.from_numpy(np.array(img)).permute(2, 0, 1)
                      .contiguous())
        return torch.stack(xs), torch.tensor(ds.fine.astype(np.int64))

    @torch.no_grad()
    def feats(X):
        out = []
        for i in range(0, len(X), 256):
            x = X[i:i + 256].to(device).float().div_(255.)
            x = (x - IMNET_MEAN) / IMNET_STD
            with torch.autocast('cuda', torch.float16):
                toks = model.forward_features(x)
            out.append(toks[:, 0].float().cpu())
        return torch.cat(out)

    print('extracting LoRA features...', flush=True)
    Xf, Yf = cache(fit); Xv, Yv = cache(val); Xt, Yt = cache(test)
    Ff, Fv, Ft = feats(Xf), feats(Xv), feats(Xt)
    yv, yt = Yv.numpy(), Yt.numpy()
    print(f'features done ({time.time()-t0:.0f}s)', flush=True)

    # decoupled balanced linear probe on LoRA features
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
        lv = (Fv.to(device) @ W.weight.T + W.bias).cpu().numpy()
        lt = (Ft.to(device) @ W.weight.T + W.bias).cpu().numpy()
    lora_lin = float((lt.argmax(1) == yt).mean())
    print(f'LoRA-B/16 linear probe: test {lora_lin:.4f}', flush=True)

    # frozen ViT-L branches
    d = torch.load('vitl_feats.pt')
    Wd = torch.load('vitl_std.pt')
    T, Wf, bf = Wd['T'], Wd['W'], Wd['b']
    fz_v = (d['jval'] @ T.T).numpy(); fz_t = (d['jtst'] @ T.T).numpy()
    fl_v = (d['jval'] @ Wf.T + bf).numpy()
    fl_t = (d['jtst'] @ Wf.T + bf).numpy()

    def zl(x, tau):
        return zscore_rows(torch.tensor(x) - tau * lp[None, :]).numpy()

    best = (-1.0, None)
    for tau in (0.0, 0.3, 0.6):
        a = zl(lv, tau)                       # LoRA-B/16
        c = zl(fl_v, tau)                     # frozen ViT-L linear
        b = zscore_rows(torch.tensor(fz_v)).numpy()   # zero-shot
        for wa in (0, 0.5, 1, 2):
            for wb in (0, 0.25, 0.5, 1):
                for wc in (0, 0.25, 0.5, 1):
                    acc = float(((wa * a + wb * b + wc * c).argmax(1)
                                 == yv).mean())
                    if acc > best[0]:
                        best = (acc, (tau, wa, wb, wc))
    tau, wa, wb, wc = best[1]
    sc = (wa * zl(lt, tau) + wb * zscore_rows(torch.tensor(fz_t)).numpy()
          + wc * zl(fl_t, tau))
    scv = (wa * zl(lv, tau)
           + wb * zscore_rows(torch.tensor(fz_v)).numpy()
           + wc * zl(fl_v, tau))
    pred = sc.argmax(1)
    hit = pred == yt
    b_of_y = BUCKET[yt]
    macro = float(np.mean([hit[yt == c].mean() for c in range(100)]))

    def ece_of(p, y, bins=15):
        conf = p.max(1)
        h = (p.argmax(1) == y).astype(float)
        e = 0.0
        for lo in np.linspace(0, 1, bins + 1)[:-1]:
            m = (conf > lo) & (conf <= lo + 1 / bins)
            if m.sum():
                e += m.mean() * abs(h[m].mean() - conf[m].mean())
        return float(e)

    bestT = (1.0, 1e9)
    for TT in np.concatenate([np.geomspace(0.05, 1, 10),
                              np.geomspace(1, 50, 10)]):
        e = ece_of(torch.softmax(torch.tensor(scv / TT), 1).numpy(), yv)
        if e < bestT[1]:
            bestT = (float(TT), e)
    probs = torch.softmax(torch.tensor(sc / bestT[0]), 1).numpy()
    top5 = torch.tensor(sc).topk(5, 1).indices.numpy()
    hit5 = np.array([yt[i] in top5[i] for i in range(len(yt))])

    print('=== FINAL ENSEMBLE (LoRA-B/16 + frozen ViT-L) ===')
    print(f'gates: tau={tau} w=({wa},{wb},{wc})  val {best[0]:.4f}')
    print(f'overall {hit.mean():.4f}  top5 {hit5.mean():.4f}  macro {macro:.4f}')
    for bn in ('head', 'medium', 'tail'):
        m = b_of_y == bn
        print(f'{bn:>6}: {hit[m].mean():.4f}')
    print(f'ECE (T={bestT[0]:.3f}): {ece_of(probs, yt):.4f}')
    print(f'TARGET LPT 89.10 -> {"*** EXCEEDED ***" if hit.mean() > 0.891 else "not yet"}')


if __name__ == '__main__':
    main()
