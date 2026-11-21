"""Diagnosis round 2: bf16 vs fp16 backward, and ViT-B/32 backward sanity."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import time
import torch
from transformers import CLIPModel

device = 'cuda'


def bench(name, fn, n=6, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t) / n
    print(f'{name:<44} {dt*1000:8.1f} ms', flush=True)


def make_step(model, x, dtype):
    def step():
        with torch.autocast('cuda', dtype):
            emb = model.get_image_features(pixel_values=x)
            loss = emb.float().pow(2).mean()
        loss.backward()
    return step


for tag, mid in (('ViT-L/14', 'openai/clip-vit-large-patch14'),
                 ('ViT-B/32', 'openai/clip-vit-base-patch32')):
    m = CLIPModel.from_pretrained(mid).cuda()
    for p in m.vision_model.encoder.layers[-1].parameters():
        p.requires_grad_(True)
    x = torch.randn(16, 3, 224, 224, device=device)
    bench(f'{tag} fwd+bwd amp-fp16 b16', make_step(m, x, torch.float16), n=3)
    bench(f'{tag} fwd+bwd amp-bf16 b16', make_step(m, x, torch.bfloat16), n=3)
    # also try pure-fp16 model weights (half()) with no autocast
    m2 = m.half()
    for p in m2.vision_model.encoder.layers[-1].parameters():
        p.requires_grad_(True)
    xh = x.half()

    def step_half(m2=m2, xh=xh):
        emb = m2.get_image_features(pixel_values=xh)
        loss = emb.float().pow(2).mean()
        loss.backward()
    bench(f'{tag} fwd+bwd weights-fp16 b16', step_half, n=3)
    del m, m2
    torch.cuda.empty_cache()
