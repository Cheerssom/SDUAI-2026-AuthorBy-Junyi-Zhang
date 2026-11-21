"""Evaluate LoRA-adapted CLIP ViT-L/14 on CIFAR-100-LT rho=100.

Pipeline (all gates selected on the held-out val, never on test):
  1. extract LoRA joint embeddings for fit/val/test
  2. decoupled balanced linear probe on LoRA features
  3. ensemble with the FROZEN branches (zero-shot anchors + frozen linear)
  4. full metrics: overall/top5/macro/h/m/t + ECE(T fit on val)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel

from lora_train import LoRALinear


def build_lora_model(device):
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda()
    for blk in model.vision_model.encoder.layers:
        for name in ('q_proj', 'v_proj'):
            lin = getattr(blk.self_attn, name)
            setattr(blk.self_attn, name, LoRALinear(lin, 8, 16).cuda())
    sd = torch.load('lora_vitl.pt', map_location=device, weights_only=False)
    model.load_state_dict(sd['lora'], strict=False)
    model.eval()
    return model


def main():
    device = 'cuda'
    from lt_dataset import make_lt_datasets
    from train import TF_CLIP, split_train_val
    from linear_probe import zscore_rows

    model = build_lora_model(device)
    train_eval, _, _, test, meta = make_lt_datasets(
        rho=100, train_tf=TF_CLIP, support_tf=TF_CLIP, test_tf=TF_CLIP)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])
    BUCKET = np.where(counts > 100, 'head',
                      np.where(counts >= 20, 'medium', 'tail'))
    lp = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)

    @torch.no_grad()
    def jemb_lora(ds):
        out = []
        for x, _, _, _ in DataLoader(ds, batch_size=128):
            e = model.get_image_features(pixel_values=x.cuda())
            out.append(F.normalize(e.float(), dim=1).cpu())
        return torch.cat(out)

    print('extracting LoRA features...', flush=True)
    jfit, jval, jtst = jemb_lora(fit), jemb_lora(val), jemb_lora(test)
    yfit, yval, ytst = (torch.tensor(fit.fine), torch.tensor(val.fine),
                        torch.tensor(test.fine))

    # decoupled balanced linear probe on LoRA features
    W = nn.Linear(768, 100).to(device)
    opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
    w = torch.tensor(1.0 / counts[fit.fine], dtype=torch.double)
    smp = torch.utils.data.WeightedRandomSampler(w, len(jfit),
                                                 replacement=True)
    ds = TensorDataset(jfit, yfit)
    g = torch.Generator().manual_seed(0)
    for ep in range(100):
        for xb, yb in DataLoader(ds, batch_size=256, sampler=smp, generator=g):
            loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lv = (jval.to(device) @ W.weight.T + W.bias).cpu()
        lt = (jtst.to(device) @ W.weight.T + W.bias).cpu()
    lin_t = float(((lt.argmax(1) == ytst).float().mean()))
    print(f'LoRA linear probe: test {lin_t:.4f}', flush=True)

    # frozen branches from cache
    d = torch.load('vitl_feats.pt')
    Wd = torch.load('vitl_std.pt')
    T = Wd['T']
    Wf, bf = Wd['W'], Wd['b']
    fv, ft_ = d['jval'], d['jtst']
    fz_v = (fv @ T.T).numpy()
    fz_t = (ft_ @ T.T).numpy()
    fl_v = (fv @ Wf.T + bf).numpy()
    fl_t = (ft_ @ Wf.T + bf).numpy()

    yv_np, yt_np = yval.numpy(), ytst.numpy()

    def zl(x, tau):
        return zscore_rows(torch.tensor(x) - tau * lp[None, :]).numpy()

    best = (-1.0, None)
    for tau in (0.0, 0.3, 0.6):
        a = zl(lv.numpy(), tau)
        c = zl(fl_v, tau)
        b = zscore_rows(torch.tensor(fz_v)).numpy()
        for wa in (0, 0.5, 1, 2):
            for wb in (0, 0.25, 0.5, 1):
                for wc in (0, 0.25, 0.5, 1):
                    acc = float(((wa * a + wb * b + wc * c).argmax(1)
                                 == yv_np).mean())
                    if acc > best[0]:
                        best = (acc, (tau, wa, wb, wc))
    tau, wa, wb, wc = best[1]
    at, ct = zl(lt.numpy(), tau), zl(fl_t, tau)
    bt = zscore_rows(torch.tensor(fz_t)).numpy()
    sc = wa * at + wb * bt + wc * ct
    scv = wa * zl(lv.numpy(), tau) + wb * zscore_rows(
        torch.tensor(fz_v)).numpy() + wc * zl(fl_v, tau)

    pred = sc.argmax(1)
    hit = pred == yt_np
    b_of_y = BUCKET[yt_np]
    top5 = torch.tensor(sc).topk(5, 1).indices.numpy()
    hit5 = np.array([yt_np[i] in top5[i] for i in range(len(yt_np))])
    macro = float(np.mean([hit[yt_np == c].mean() for c in range(100)]))

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
        e = ece_of(torch.softmax(torch.tensor(scv / TT), 1).numpy(), yv_np)
        if e < bestT[1]:
            bestT = (float(TT), e)
    probs = torch.softmax(torch.tensor(sc / bestT[0]), 1).numpy()

    print('=== LoRA final ensemble ===')
    print(f'gates: tau={tau} w=({wa},{wb},{wc})  val {best[0]:.4f}')
    print(f'overall {hit.mean():.4f}  top5 {hit5.mean():.4f}  '
          f'macro {macro:.4f}')
    for bn in ('head', 'medium', 'tail'):
        m = b_of_y == bn
        print(f'{bn:>6}: {hit[m].mean():.4f}')
    print(f'ECE (T={bestT[0]:.3f}): {ece_of(probs, yt_np):.4f}')
    print(f'TARGET: LPT 89.10  ->  '
          f'{"EXCEEDED" if hit.mean() > 0.891 else "not yet"}')


if __name__ == '__main__':
    main()
