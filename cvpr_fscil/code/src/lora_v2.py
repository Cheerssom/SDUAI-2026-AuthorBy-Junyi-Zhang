"""LoRA v2: LIFT+-style training (balanced softmax + semantic init + r=16).

Key changes from lora_generic:
  - balanced softmax loss (logit adjustment DURING training, no post-hoc)
  - head initialized from class-mean CLIP features (semantic init)
  - rank 16, 15 epochs
  - balanced sampler + joint head training (not decoupled)
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

RANK, ALPHA = 16, 32
EPOCHS, BS, LR = 15, 32, 3e-4
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def main(seed=0, tag='b16_v2'):
    import timm
    from lt_dataset import make_lt_datasets
    from train import split_train_val
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda'
    t0 = time.time()

    train_eval, _, _, test, meta = make_lt_datasets(rho=100)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])
    log_prior = torch.tensor(np.log(counts / counts.sum()),
                             dtype=torch.float32).to(device)

    def cache(ds):
        xs = []
        for i in range(len(ds)):
            img = Image.fromarray(ds.x[i].transpose(1, 2, 0))
            img = img.resize((224, 224), Image.BICUBIC)
            xs.append(torch.from_numpy(np.array(img)).permute(2, 0, 1)
                      .contiguous())
        return torch.stack(xs), torch.tensor(ds.fine.astype(np.int64))

    print(f'[{tag}] caching...', flush=True)
    Xf, Yf = cache(fit)
    Xv, Yv = cache(val)

    model = timm.create_model('vit_base_patch16_224.augreg_in21k',
                              pretrained=True, num_classes=0).cuda()
    dim = model.embed_dim
    lora_params = []
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, RANK, ALPHA).cuda()
        lora_params += [blk.attn.qkv.A, blk.attn.qkv.B]

    # semantic head init: class-mean of CLIP ViT-L joint embeddings (768d)
    # mapped to timm dim (768) via a learned projection is overkill;
    # simpler: class-mean of timm features at init (one forward pass)
    @torch.no_grad()
    def extract_cls(X, bs=256):
        out = []
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(device).float().div_(255.)
            x = (x - IM_MEAN) / IM_STD
            with torch.autocast('cuda', torch.float16):
                t = model.forward_features(x)
            out.append(t[:, 0].float().cpu())
        return torch.cat(out)

    print(f'[{tag}] extracting init features...', flush=True)
    F_init = extract_cls(Xf)
    cls_means = torch.stack([F_init[Yf == c].mean(0)
                             if (Yf == c).sum() > 0
                             else torch.zeros(dim)
                             for c in range(100)]).to(device)
    cls_means = F.normalize(cls_means, dim=1)
    head = nn.Linear(dim, 100).to(device)
    with torch.no_grad():
        head.weight.copy_(cls_means)     # semantic init
        head.bias.zero_()
    print(f'[{tag}] semantic head initialized ({time.time()-t0:.0f}s)',
          flush=True)

    params = lora_params + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    w = torch.tensor(1.0 / counts[fit.fine], dtype=torch.double)

    best_val, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train(); head.train()
        perm = torch.multinomial(w, len(Xf), replacement=True)
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            x = Xf[idx].to(device).float().div_(255.)
            flip = torch.rand(x.shape[0], device=device) < 0.5
            x[flip] = torch.flip(x[flip], dims=[3])
            x = (x - IM_MEAN) / IM_STD
            y = Yf[idx].to(device)
            with torch.autocast('cuda', torch.float16):
                feat = model.forward_features(x)[:, 0]
                logits = head(feat.float())
                # balanced softmax: add log-prior to logits during training
                loss = F.cross_entropy(logits + log_prior[None, :], y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); head.eval()
        vhit = {}
        with torch.no_grad():
            for i in range(0, len(Xv), 256):
                x = Xv[i:i + 256].to(device).float().div_(255.)
                x = (x - IM_MEAN) / IM_STD
                with torch.autocast('cuda', torch.float16):
                    feat = model.forward_features(x)[:, 0]
                    logits = head(feat.float())
                pred = logits.argmax(1).cpu()  # no prior at test (balanced)
                for yi, pi in zip(Yv[i:i + 256].tolist(), pred.tolist()):
                    vhit.setdefault(yi, []).append(int(yi == pi))
        vacc = float(np.mean([np.mean(v) for v in vhit.values()]))
        if vacc > best_val:
            best_val = vacc
            best_state = {'lora': copy.deepcopy(
                {k: v for k, v in model.state_dict().items()
                 if k.endswith('.A') or k.endswith('.B')}),
                'head': copy.deepcopy(head.state_dict())}
        print(f'[{tag}] ep{ep+1}/{EPOCHS}: loss {loss.item():.3f} '
              f'val {vacc:.4f} ({time.time()-t0:.0f}s)'
              f'{" *" if vacc == best_val else ""}', flush=True)

    torch.save({'lora': best_state['lora'], 'head': best_state['head'],
                'val': best_val, 'rank': RANK, 'alpha': ALPHA},
               f'lora_{tag}.pt')
    print(f'[{tag}] DONE val {best_val:.4f} ({time.time()-t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    tag = sys.argv[2] if len(sys.argv) > 2 else 'b16_v2'
    main(seed, tag)
