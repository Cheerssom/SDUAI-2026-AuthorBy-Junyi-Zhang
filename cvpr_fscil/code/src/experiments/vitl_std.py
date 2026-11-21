"""ViT-L/14 standard-table run: ensemble zero-shot + balanced linear probe."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPTokenizer
from lt_dataset import make_lt_datasets
from train import TF_CLIP, split_train_val
from linear_probe import zscore_rows

from config import CFG

device = 'cuda'
with open(CFG.cifar_root / 'meta', 'rb') as f:
    names = pickle.load(f, encoding='latin1')['fine_label_names']
names = [n.replace('_', ' ') for n in names]
model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').cuda().eval()
tkz = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

TEMPLATES = ['a bad photo of a {}.', 'a sculpture of a {}.',
 'a photo of the hard to see {}.', 'a low resolution photo of the {}.',
 'a painting of the {}.', 'a cropped photo of the {}.', 'a tattoo of the {}.',
 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.',
 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.',
 'a photo of the cool {}.', 'a close-up photo of a {}.',
 'a black and white photo of the {}.', 'a pixelated photo of the {}.',
 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.',
 'a photo of a clean {}.', 'a origami {}.', 'a photo of a large {}.',
 'a rendition of the {}.', 'a photo of a small {}.', 'itap of the {}.',
 'a photo of a nice {}.', 'a photo of a weird {}.', 'a photo of the large {}.',
 'a photo of the small {}.', 'a photo of a {}.']


@torch.no_grad()
def jemb(ds):
    out = []
    for x, _, _, _ in DataLoader(ds, batch_size=128):
        out.append(F.normalize(
            model.get_image_features(pixel_values=x.cuda()).float(), dim=1).cpu())
    return torch.cat(out)


@torch.no_grad()
def tanc():
    Ts = []
    for t in TEMPLATES:
        b = tkz([t.format(n) for n in names], return_tensors='pt', padding=True)
        Ts.append(F.normalize(
            model.get_text_features(**{k: v.cuda() for k, v in b.items()}).float(),
            dim=1).cpu())
    return F.normalize(torch.stack(Ts).mean(0), dim=1)


train_eval, _, _, test, meta = make_lt_datasets(
    rho=100, train_tf=TF_CLIP, support_tf=TF_CLIP, test_tf=TF_CLIP)
fit, val = split_train_val(train_eval)
counts = np.array(meta['cls_num_list'])
lp = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)
jfit, jval, jtst = jemb(fit), jemb(val), jemb(test)
T = tanc()
y_fit = fit.fine
y_val = torch.tensor(val.fine)
y_tst = torch.tensor(test.fine)

zs_v = float(((jval @ T.T).argmax(1) == y_val).float().mean())
zs_t = float(((jtst @ T.T).argmax(1) == y_tst).float().mean())
print(f'zero-shot(ens): val {zs_v:.4f} test {zs_t:.4f}')

W = nn.Linear(768, 100).to(device)
opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=0.05)
w = torch.tensor(1.0 / counts[y_fit], dtype=torch.double)
smp = torch.utils.data.WeightedRandomSampler(w, len(jfit), replacement=True)
ds = TensorDataset(jfit, torch.tensor(y_fit))
g = torch.Generator().manual_seed(0)
for ep in range(100):
    for xb, yb in DataLoader(ds, batch_size=256, sampler=smp, generator=g):
        loss = F.cross_entropy(W(xb.to(device)), yb.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    lv = (jval.to(device) @ W.weight.T + W.bias).cpu()
    lt = (jtst.to(device) @ W.weight.T + W.bias).cpu()
lin_v = float(((lv.argmax(1) == y_val).float().mean()))
lin_t = float(((lt.argmax(1) == y_tst).float().mean()))
print(f'linear (balanced, joint): val {lin_v:.4f} test {lin_t:.4f}')

torch.save({'W': W.weight.detach().cpu(), 'b': W.bias.detach().cpu(), 'T': T},
           'vitl_std.pt')
torch.save({'jfit': jfit, 'yfit': torch.tensor(y_fit), 'jval': jval,
            'yval': y_val, 'jtst': jtst, 'ytst': y_tst}, 'vitl_feats.pt')
zsv = zscore_rows(jval @ T.T).numpy()
zst = zscore_rows(jtst @ T.T).numpy()
best = (-1.0, None)
for tau in (0.0, 0.3, 0.6):
    a = zscore_rows(lv - tau * lp[None, :]).numpy()
    for wa in (0, 0.25, 0.5, 1, 2):
        for wb in (0, 0.25, 0.5, 1, 2):
            acc = float(((wa * a + wb * zsv).argmax(1) == y_val.numpy()).mean())
            if acc > best[0]:
                best = (acc, (tau, wa, wb))
tau, wa, wb = best[1]
at = zscore_rows(lt - tau * lp[None, :]).numpy()
sc = wa * at + wb * zst
ens_t = float((sc.argmax(1) == y_tst.numpy()).mean())
print(f'ENSEMBLE (tau={tau}, w=({wa},{wb})): val {best[0]:.4f}  test {ens_t:.4f}')
