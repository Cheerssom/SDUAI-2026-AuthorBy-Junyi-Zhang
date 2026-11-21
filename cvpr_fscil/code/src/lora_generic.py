"""Generic fast LoRA trainer for timm ViT backbones (B/16 IN21k / CLIP B/32).

Backbone: timm ViT with num_classes=0 (forward_features -> tokens).
LoRA on q_proj/v_proj of all blocks; head on pooled CLS (pre-projection).
Pre-cached 224px uint8 images; GPU-side normalize+flip; autocast fp16.
Standard ImageNet normalization for IN21k backbones, CLIP stats for CLIP.
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

RANK, ALPHA = 8, 16
EPOCHS, BS, LR = 12, 32, 3e-4
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


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


def wrap_lora(model):
    """timm ViT: blocks[i].attn.qkv is fused — wrap the whole qkv Linear."""
    lora_params = []
    for blk in model.blocks:
        qkv = blk.attn.qkv
        blk.attn.qkv = LoRALinear(qkv, RANK, ALPHA).cuda()
        lora_params += [blk.attn.qkv.A, blk.attn.qkv.B]
    return lora_params


def main(model_name='vit_base_patch16_224.augreg_in21k', tag='in21k_b16',
         clip_stats=False):
    device = 'cuda'
    from lt_dataset import make_lt_datasets
    from train import split_train_val
    import timm
    t0 = time.time()
    train_eval, _, _, test, meta = make_lt_datasets(rho=100)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])

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
    mean = (CLIP_MEAN if clip_stats else IMNET_MEAN).to(device)
    std = (CLIP_STD if clip_stats else IMNET_STD).to(device)

    model = timm.create_model(model_name, pretrained=True, num_classes=0).cuda()
    dim = model.embed_dim
    lora_params = wrap_lora(model)
    head = nn.Linear(dim, 100).to(device)
    n_tr = (sum(p.numel() for p in lora_params)
            + sum(p.numel() for p in head.parameters()))
    print(f'[{tag}] LoRA r={RANK} trainable {n_tr/1e6:.2f}M '
          f'(d={dim}, {time.time()-t0:.0f}s)', flush=True)

    opt = torch.optim.AdamW(list(lora_params) + list(head.parameters()),
                            lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    w = torch.tensor(1.0 / counts[fit.fine], dtype=torch.double)

    def forward_pool(xb):
        toks = model.forward_features(xb)     # B, 1+P, d
        return toks[:, 0]

    best_val, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train(); head.train()
        perm = torch.multinomial(w, len(Xf), replacement=True)
        for i in range(0, len(perm) - BS + 1, BS):
            idx = perm[i:i + BS]
            x = Xf[idx].to(device).float().div_(255.)
            flip = torch.rand(x.shape[0], device=device) < 0.5
            x[flip] = torch.flip(x[flip], dims=[3])
            x = (x - mean) / std
            y = Yf[idx].to(device)
            with torch.autocast('cuda', torch.float16):
                loss = F.cross_entropy(head(forward_pool(x).float()), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); head.eval()
        vhit = {}
        with torch.no_grad():
            for i in range(0, len(Xv), 256):
                x = Xv[i:i + 256].to(device).float().div_(255.)
                x = (x - mean) / std
                with torch.autocast('cuda', torch.float16):
                    logits = head(forward_pool(x).float())
                pred = logits.argmax(1).cpu()
                for yi, pi in zip(Yv[i:i + 256].tolist(), pred.tolist()):
                    vhit.setdefault(yi, []).append(int(yi == pi))
        vacc = float(np.mean([np.mean(v) for v in vhit.values()]))
        if vacc > best_val:
            best_val = vacc
            best_state = {'lora': copy.deepcopy(
                {k: v for k, v in model.state_dict().items()
                 if k.endswith('.A') or k.endswith('.B')})}
        print(f'[{tag}] ep{ep+1}/{EPOCHS}: loss {loss.item():.3f} '
              f'val {vacc:.4f} ({time.time()-t0:.0f}s)'
              f'{" *" if vacc == best_val else ""}', flush=True)

    torch.save({'lora': best_state['lora'], 'model': model_name,
                'clip_stats': clip_stats, 'val': best_val},
               f'lora_{tag}.pt')
    print(f'[{tag}] DONE best val {best_val:.4f} '
          f'({time.time()-t0:.0f}s total)', flush=True)


if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else \
        'vit_base_patch16_224.augreg_in21k'
    tag = sys.argv[2] if len(sys.argv) > 2 else 'in21k_b16'
    clip = len(sys.argv) > 3 and sys.argv[3] == 'clip'
    main(name, tag, clip)
