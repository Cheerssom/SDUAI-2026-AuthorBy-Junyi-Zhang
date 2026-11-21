"""Final metrics for the SOTA-surpassing model (LoRA-B/16 + frozen ViT-L)."""

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
TAU, WA, WB, WC = 1.0, 2, 0.75, 0.75   # val-selected


def main():
    import timm
    from lt_dataset import make_lt_datasets
    from train import split_train_val
    t0 = time.time()
    ck = torch.load('lora_in21k_b16.pt', map_location=device,
                    weights_only=False)
    model = timm.create_model(ck['model'], pretrained=True, num_classes=0)
    model = model.cuda()
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, RANK, ALPHA).cuda()
    model.load_state_dict(ck['lora'], strict=False)
    model.eval()
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
            x = (x - IM_MEAN) / IM_STD
            with torch.autocast('cuda', torch.float16):
                t = model.forward_features(x)
            out.append(t[:, 0].float().cpu())
        return torch.cat(out)

    Xf, Yf = cache(fit); Xv, Yv = cache(val); Xt, Yt = cache(test)
    Ff, Fv, Ft = feats(Xf), feats(Xv), feats(Xt)
    yv, yt = Yv.numpy(), Yt.numpy()
    print(f'features {time.time()-t0:.0f}s', flush=True)

    W = nn.Linear(Ff.shape[1], 100).to(device)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
    w = torch.tensor(1.0 / counts[fit.fine], dtype=torch.double)
    smp = torch.utils.data.WeightedRandomSampler(w, len(Ff),
                                                 replacement=True)
    ds0 = TensorDataset(Ff, Yf)
    g = torch.Generator().manual_seed(0)
    for ep in range(100):
        for xb, yb in DataLoader(ds0, batch_size=256, sampler=smp,
                                 generator=g):
            loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lv = (Fv.to(device) @ W.weight.T + W.bias).cpu().numpy()
        lt = (Ft.to(device) @ W.weight.T + W.bias).cpu().numpy()
    np.savez('lora_b16_logits.npz', lv=lv, lt=lt, yv=yv, yt=yt)

    d = torch.load('vitl_feats.pt')
    Wd = torch.load('vitl_std.pt')
    T, Wf, bf = Wd['T'], Wd['W'], Wd['b']
    fz_v = (d['jval'] @ T.T).numpy(); fz_t = (d['jtst'] @ T.T).numpy()
    fl_v = (d['jval'] @ Wf.T + bf).numpy()
    fl_t = (d['jtst'] @ Wf.T + bf).numpy()

    def zl(x, tau):
        return zscore_rows(torch.tensor(x) - tau * lp[None, :]).numpy()

    sc = (WA * zl(lt, TAU) + WB * zscore_rows(torch.tensor(fz_t)).numpy()
          + WC * zl(fl_t, TAU))
    scv = (WA * zl(lv, TAU)
           + WB * zscore_rows(torch.tensor(fz_v)).numpy()
           + VC * zl(fl_v, TAU)) if False else (
        WA * zl(lv, TAU) + WB * zscore_rows(torch.tensor(fz_v)).numpy()
        + WC * zl(fl_v, TAU))
    pred = sc.argmax(1)
    hit = pred == yt
    b_of_y = BUCKET[yt]
    top5 = torch.tensor(sc).topk(5, 1).indices.numpy()
    hit5 = np.array([yt[i] in top5[i] for i in range(len(yt))])
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

    print('=== FINAL MODEL (LoRA IN21k-B/16 + frozen CLIP-L/14 branches) ===')
    print(f'gates tau={TAU} w=({WA},{WB},{WC}) (val-selected)')
    print(f'overall {hit.mean():.4f}  top5 {hit5.mean():.4f}  '
          f'macro {macro:.4f}')
    for bn in ('head', 'medium', 'tail'):
        m = b_of_y == bn
        pm = torch.softmax(torch.tensor(sc[m] / bestT[0]), 1).numpy()
        print(f'{bn:>6}: top1 {hit[m].mean():.4f}  '
              f'top5 {hit5[m].mean():.4f}  '
              f'ECE {ece_of(pm, yt[m]) * 100:.2f}%')
    print(f'ECE global (T={bestT[0]:.2f}): {ece_of(probs, yt) * 100:.2f}%')
    print(f'vs LPT 89.10: {"+" if hit.mean() > 0.891 else ""}'
          f'{(hit.mean() - 0.891) * 100:.2f} '
          f'{"EXCEEDED" if hit.mean() > 0.891 else "short"}')


if __name__ == '__main__':
    main()
