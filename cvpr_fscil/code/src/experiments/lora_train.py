"""LoRA fine-tuning of CLIP ViT-L/14 vision tower on CIFAR-100-LT (fit split).

Manual LoRA (no peft dependency): wrap q_proj/v_proj of every vision-encoder
layer; everything else frozen. Classification head sits on the PROJECTED
768-d joint embedding so adapted features stay comparable to text anchors.
Balanced sampling; val macro acc monitored per epoch; best state kept.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel

RANK, ALPHA = 8, 16
EPOCHS, BS, LR = 12, 16, 1e-4


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


def main():
    device = 'cuda'
    from lt_dataset import make_lt_datasets
    from train import TF_CLIP_TRAIN, TF_CLIP, split_train_val

    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda()
    model.eval()
    lora_params = []
    for blk in model.vision_model.encoder.layers:
        for name in ('q_proj', 'v_proj'):
            lin = getattr(blk.self_attn, name)
            setattr(blk.self_attn, name, LoRALinear(lin, RANK, ALPHA).cuda())
            lora_params += list(getattr(blk.self_attn, name).parameters())
    head = nn.Linear(768, 100).to(device)
    params = lora_params + list(head.parameters())
    n_lora = sum(p.numel() for p in lora_params)
    print(f'LoRA r={RANK}: {n_lora/1e6:.2f}M trainable (+ head '
          f'{sum(p.numel() for p in head.parameters())/1e6:.2f}M), '
          f'backbone frozen', flush=True)

    train_eval, _, _, test, meta = make_lt_datasets(
        rho=100, train_tf=TF_CLIP_TRAIN, support_tf=TF_CLIP, test_tf=TF_CLIP)
    fit, val = split_train_val(train_eval)
    counts = np.array(meta['cls_num_list'])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    w = torch.tensor(1.0 / counts[fit.fine], dtype=torch.double)
    smp = torch.utils.data.WeightedRandomSampler(w, len(fit), replacement=True)
    loader = DataLoader(fit, batch_size=BS, sampler=smp, num_workers=0,
                        drop_last=True)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)

    best_val, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast('cuda', torch.float16):
                emb = model.get_image_features(pixel_values=x)
                loss = F.cross_entropy(head(emb.float()), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); head.eval()
        vhit = {}
        with torch.no_grad(), torch.autocast('cuda', torch.float16):
            for x, y, _, _ in val_loader:
                emb = model.get_image_features(pixel_values=x.to(device))
                pred = head(emb.float()).argmax(1).cpu()
                for yi, pi in zip(y.tolist(), pred.tolist()):
                    vhit.setdefault(yi, []).append(int(yi == pi))
        vacc = float(np.mean([np.mean(v) for v in vhit.values()]))
        if vacc > best_val:
            best_val = vacc
            best_state = {
                'lora': copy.deepcopy(
                    {k: v for k, v in model.state_dict().items()
                     if 'self_attn.' in k and ('.A' in k or '.B' in k)}),
                'head': copy.deepcopy(head.state_dict())}
        print(f'ep{ep+1}: loss {loss.item():.3f}  val macro {vacc:.4f}'
              + ('  *' if vacc == best_val else ''), flush=True)

    model.load_state_dict({**model.state_dict(), **best_state['lora']},
                          strict=False)
    torch.save({'lora': best_state['lora']}, 'lora_vitl.pt')
    print(f'best val macro {best_val:.4f}; saved lora_vitl.pt', flush=True)


if __name__ == '__main__':
    main()
