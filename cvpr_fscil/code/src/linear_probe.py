"""Linear probe over cached encoder features — the learned-weight readout
that seen-class candidates carry in the unified candidate table (v3.1).

Rationale (results_log 2026-09-21): on CLIP features a balanced-sampled
linear probe reaches 65.8 vs the typed head's 57.5, while cosine prototypes
are the only readout that transfers to unseen candidates (54.0 @10-shot).
The candidate table therefore carries BOTH: learned weight vectors for
classes with training evidence, prototypes for everyone, and evidence
gating picks per candidate.

Trains on the FIT split only (val untouched for tau selection).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_linear_probe(encoder, fit_set, num_classes, device, meta=None,
                       epochs=60, lr=1e-3, wd=0.01, seed=0,
                       labels=None, cached_feats=None):
    """Returns (W, b) as (num_classes, d), (num_classes,) tensors on device.
    For the CLIP path pass cached_feats=(pooled, ys) to skip the encoder."""
    torch.manual_seed(seed)
    if cached_feats is not None:
        X, Y = cached_feats
    else:
        from train import cache_features
        c = cache_features(encoder, fit_set, device)
        X, Y = c.pooled, c.ys
    counts = None
    if labels is not None:
        counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = torch.ones(len(Y))
    if counts is not None:
        w = torch.tensor(1.0 / counts[Y.numpy()], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(w, len(Y), replacement=True)
    lin = nn.Linear(X.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    ds = TensorDataset(X, Y)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        for xb, yb in DataLoader(ds, batch_size=256, sampler=sampler,
                                 generator=g):
            loss = F.cross_entropy(lin(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    return lin.weight.detach(), lin.bias.detach()


@torch.no_grad()
def linear_logits(W, b, pooled):
    """(num_classes,d),(d,),(B,d) -> (B,num_classes)."""
    return pooled @ W.T + b


def zscore_rows(x):
    """Per-row standardization over the row's finite entries (linear branch
    is absent for new candidates -> -inf padding handled)."""
    x = x.clone().float()
    valid = torch.isfinite(x)
    x = torch.where(valid, x, torch.zeros_like(x))
    n = valid.sum(1, keepdim=True).clamp(min=1).float()
    mu = (x.sum(1, keepdim=True)) / n
    d = torch.where(valid, x - mu, torch.zeros_like(x))
    var = (d ** 2).sum(1, keepdim=True) / (n - 1).clamp(min=1)
    sd = var.sqrt().clamp(min=1e-6)
    z = torch.where(valid, (x - mu) / sd, torch.zeros_like(x))
    return z
