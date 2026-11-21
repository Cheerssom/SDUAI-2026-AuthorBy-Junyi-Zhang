"""FAST LoRA training: 224px images pre-cached as uint8 tensors in RAM,
normalization + random flip done on GPU. Everything else identical to
lora_train.py (r=8 Q/V LoRA, balanced sampling, 12 epochs, val macro select).
Unbuffered prints for live progress.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel
import torchvision.transforms.functional as TF

RANK, ALPHA = 8, 16
EPOCHS, BS, LR = 12, 16, 1e-4
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


class CachedResized(Dataset):
    """Images pre-resized to 224 (uint8 CHW tensors); flip on GPU later."""

    def __init__(self, ds):
        from PIL import Image
        xs = []
        for i in range(len(ds)):
            img = Image.fromarray(ds.x[i].transpose(1, 2, 0))  # CHW->HWC uint8
            img = img.resize((224, 224), Image.BICUBIC)
            xs.append(torch.from_numpy(np.array(img)).permute(2, 0, 1)
                      .contiguous())
        self.xs = torch.stack(xs)                      # N,3,224,224 uint8
        self.ys = torch.tensor(ds.fine.astype(np.int64))

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.xs[i], int(self.ys[i])


def gpu_batch(x_u8, y, train, device):
    x = x_u8.to(device, non_blocking=True).float().div_(255.)
    if train:
        flip = torch.rand(x.shape[0], device=device) < 0.5
        x[flip] = torch.flip(x[flip], dims=[3])
    return (x - CLIP_MEAN.to(device)) / CLIP_STD.to(device), y.to(device)


def main():
    device = 'cuda'
    from lt_dataset import make_lt_datasets
    from train import split_train_val
    t0 = time.time()
    train_eval, _, _, test, meta = make_lt_datasets(rho=100)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])
    print(f'pre-caching 224px images ({len(fit)}+{len(val)})...', flush=True)
    cfit, cval = CachedResized(fit), CachedResized(val)
    print(f'cache done in {time.time()-t0:.0f}s '
          f'({cfit.xs.numel()/1e9:.1f}GB uint8)', flush=True)

    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda()
    lora_params = []
    for blk in model.vision_model.encoder.layers:
        for name in ('q_proj', 'v_proj'):
            lin = getattr(blk.self_attn, name)
            setattr(blk.self_attn, name, LoRALinear(lin, RANK, ALPHA).cuda())
            lora_params += list(getattr(blk.self_attn, name).parameters())
    head = nn.Linear(768, 100).to(device)
    params = lora_params + list(head.parameters())
    print(f'LoRA r={RANK}: {sum(p.numel() for p in lora_params)/1e6:.2f}M '
          'trainable', flush=True)

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    w = torch.tensor(1.0 / counts[fit.fine], dtype=torch.double)
    ys_fit = torch.tensor(fit.fine.astype(np.int64))
    idx_all = torch.arange(len(cfit))

    best_val, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.multinomial(w.double(), len(cfit), replacement=True)
        loader = DataLoader(torch.utils.data.TensorDataset(
            cfit.xs, ys_fit), batch_size=BS, sampler=perm.tolist(),
            num_workers=0, pin_memory=True, drop_last=True)
        te, nb = time.time(), 0
        for x_u8, y in loader:
            x, y = gpu_batch(x_u8, y, True, device)
            with torch.autocast('cuda', torch.float16):
                emb = model.get_image_features(pixel_values=x)
                loss = F.cross_entropy(head(emb.float()), y)
            opt.zero_grad(); loss.backward(); opt.step()
            nb += 1
        sched.step()
        model.eval(); head.eval()
        vhit = {}
        with torch.no_grad():
            for i in range(0, len(cval), 256):
                x_u8, y = cval.xs[i:i+256], cval.ys[i:i+256]
                x, y = gpu_batch(x_u8, y, False, device)
                with torch.autocast('cuda', torch.float16):
                    emb = model.get_image_features(pixel_values=x)
                pred = head(emb.float()).argmax(1).cpu()
                for yi, pi in zip(y.tolist(), pred.tolist()):
                    vhit.setdefault(yi, []).append(int(yi == pi))
        vacc = float(np.mean([np.mean(v) for v in vhit.values()]))
        if vacc > best_val:
            best_val = vacc
            best_state = {'lora': copy.deepcopy(
                {k: v for k, v in model.state_dict().items()
                 if 'self_attn.' in k and ('.A' in k or '.B' in k)})}
        print(f'ep{ep+1}/{EPOCHS}: loss {loss.item():.3f} val {vacc:.4f}'
              f' ({time.time()-te:.0f}s){" *" if vacc == best_val else ""}',
              flush=True)

    torch.save({'lora': best_state['lora'], 'val': best_val}, 'lora_vitl.pt')
    print(f'DONE best val {best_val:.4f} -> lora_vitl.pt '
          f'(total {time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
