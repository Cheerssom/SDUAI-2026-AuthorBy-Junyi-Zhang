"""3-minute bottleneck diagnosis: time each pipeline stage separately."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

device = 'cuda'
print('torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)
print('bf16 supported:', torch.cuda.is_bf16_supported(), flush=True)

model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda()
model.eval()

def bench(name, fn, n=8, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t) / n
    print(f'{name:<38} {dt*1000:8.1f} ms', flush=True)
    return dt

x16 = torch.randn(16, 3, 224, 224, device=device)
x32 = torch.randn(32, 3, 224, 224, device=device)

with torch.no_grad():
    bench('forward-only fp32 b16', lambda: model.get_image_features(pixel_values=x16))
    with torch.autocast('cuda', torch.float16):
        bench('forward-only autocast-fp16 b16', lambda: model.get_image_features(pixel_values=x16))

# training-like: fwd+bwd through vision tower with grad enabled on last layer
def train_step(x):
    model.vision_model.encoder.layers[-1].train()
    for p in model.vision_model.encoder.layers[-1].parameters():
        p.requires_grad_(True)
    emb = model.get_image_features(pixel_values=x)
    loss = emb.float().pow(2).mean()
    loss.backward()

bench('fwd+bwd (last layer grad) fp32 b16', lambda: train_step(x16), n=4)
with torch.autocast('cuda', torch.float16):
    bench('fwd+bwd (last layer grad) amp b16', lambda: train_step(x16), n=4)
    bench('fwd+bwd (last layer grad) amp b32', lambda: train_step(x32), n=4)
print('est epoch time at b16 =', 655 * '<see above>', flush=True)
