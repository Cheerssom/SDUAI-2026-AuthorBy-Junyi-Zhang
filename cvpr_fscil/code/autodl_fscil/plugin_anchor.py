"""Plug-in base anchoring: make other readouts invariant and measure what it does to certification.

HyCal standardizes its cosine and Mahalanobis scores across the active classes; GDA re-estimates its
shared covariance as supports arrive. Both therefore move base scores when classes arrive (recovery).
The anchored variants change only that: HyCal_anchor takes its per-query moments over the base classes;
GDA_anchor freezes the precision on base data and standardizes text and discriminant scores against the
base classes. Everything else (features, supports, splits, selection budget, certification) is shared.
Methods (recomputed here so that all four use one code path):
  HyCal_transfer, HyCal_anchor, GDA_transfer, GDA_anchor
Per session the row also records recovery/destruction of base test images relative to the method's own
session-0 predictions. Certified track: common contract = four-source AnchorCE closed head (as --ref4),
self contract = own closed head, plus the r x calibration-cap sweep.
Output: work/readouts_plugin/<ds>/<enc>/<track>/seed<s>/   Usage: python plugin_anchor.py --all [--encoders b16]
"""
from pathlib import Path
import json, argparse, hashlib, time
import numpy as np
import torch
import fscil_readouts as f
import extra_baselines as xb

OUT = f.D / 'readouts_plugin'
METHODS = ['HyCal_transfer', 'HyCal_anchor', 'GDA_transfer', 'GDA_anchor']
ALPHA_ANCHOR = [0., .25, .5, 1., 2., 4., 8.]


def base_std(v, nb):
    ref = v[:, :nb]
    return (v - ref.mean(1, keepdim=True)) / ref.std(1, correction=1, keepdim=True).clamp_min(1e-6)


def hycal_scores(E, T, train, y, classes, nb, anchored, reg=.001):
    """f.hycal with the per-query standardization taken over all active classes (original) or base only."""
    cs, md, counts = [], [], []
    for j, k in enumerate(classes):
        x = f.norm(train[y == k] + T[j]); mu = f.norm(x.mean(0)); A = (x - mu).double()
        L = torch.linalg.cholesky(A @ A.T + reg * (len(x) - 1) * torch.eye(len(x), device='cuda'))
        B = torch.linalg.solve_triangular(L, A, upper=False).float(); z = f.norm(E + T[j]); diff = z - mu
        cs.append(z @ mu); md.append(-((diff * diff).sum(1) - (diff @ B.T).square().sum(1)).clamp_min(0) / reg); counts.append(len(x))
    cs, md = torch.stack(cs, 1), torch.stack(md, 1)
    if anchored:
        phi = lambda v: .5 * (1 + torch.erf(base_std(v, nb) / 2 ** .5))
    else:
        phi = f.cdf
    w = torch.sigmoid(torch.tensor(counts, device='cuda') - 5.)
    return (1 - w) * phi(cs) + w * phi(md)


@torch.inference_mode()
def run(ds, enc, track, seed):
    start = time.monotonic()
    f.NB, f.WAY, f.S, f.NTUNE, f.NCAL, f.NC = f.CFG[ds]; nb, way, sessions, ntune, ncal, _ = f.CFG[ds]
    src = f.D / 'readouts' / ds / enc / track / f'seed{seed}'; out = OUT / ds / enc / track / f'seed{seed}'
    if (out / 'complete.json').exists():
        print('PLUGIN_ALREADY_COMPLETE', ds, enc, track, seed, flush=True); return
    assert (src / 'complete.json').exists(), f'missing archived task {src}'
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((src / 'selection.json').read_text())
    feature = f.D / 'features' / ds / enc / 'frozen.pt'; fh = hashlib.sha256(feature.read_bytes()).hexdigest()
    assert fh == sel['features_sha256']
    z = torch.load(feature, map_location='cpu', weights_only=False)
    E = z['E'].cuda(); T = z['T'].cuda(); y = np.asarray(z['y']); iid = np.asarray(z['ids'])
    index = {int(v): i for i, v in enumerate(iid)}; base = np.arange(nb)
    rng = np.random.default_rng(28012026 + seed)
    official = np.array([index[v] for v in z['sessions']['1']]); tune, cal, fit = [], [], []
    for k in base:
        q = rng.permutation(official[y[official] == k]); tune.extend(q[:ntune]); cal.extend(q[ntune:ntune + ncal]); fit.extend(q[ntune + ncal:])
    tune, cal, fit = map(np.array, (tune, cal, fit))
    actualfit = fit if track == 'certified' else official
    selectionfit = fit if track == 'certified' else np.setdiff1d(official, tune)
    assert np.array_equal(iid[actualfit], np.asarray(sel['fit_ids'])) and np.array_equal(iid[tune], np.asarray(sel['tune_ids']))
    test = np.flatnonzero(~z['train']); bt = test[y[test] < nb]
    refc = closed = None
    if track == 'certified':
        refc, closed = f.ref4(ds, enc, track, seed, y, bt, base)
    # Base-only selection of the GDA fusion weights (original grid for GDA, anchored-scale grid for GDA_anchor).
    yt, Et = y[tune], E[tune]
    mus, P = xb.precision(E[selectionfit], y[selectionfit], base)
    vis = xb.lda(Et, mus, P); zs = 100 * Et @ T[base].T
    alpha = max(xb.ALPHAS, key=lambda a: (xb.macro_acc(zs + a * vis, base, yt), -a))
    za, va = base_std(zs, nb), base_std(vis, nb)
    alpha_a = max(ALPHA_ANCHOR, key=lambda a: (xb.macro_acc(za + a * va, base, yt), -a))
    mb, Pb = xb.precision(E[actualfit], y[actualfit], base)            # frozen precision for GDA_anchor
    f.atomic(dict(gda_alpha=alpha, gda_anchor_alpha=alpha_a, feature_sha256=fh), out / 'selection_plugin.json')
    trainidx = list(actualfit); refs = {}; allrows = []
    for si in range(sessions):
        if si:
            trainidx.extend(index[v] for v in z['sessions'][str(si + 1)])
        novel = np.arange(nb, nb + way * si); ids = np.r_[base, novel]
        active = test[y[test] < len(ids)]; queries = np.r_[tune, cal, active]; tr = np.asarray(trainidx); Q = E[queries]
        scores = {}
        scores['HyCal_transfer'] = hycal_scores(Q, T[ids], E[tr], y[tr], ids, nb, anchored=False)
        scores['HyCal_anchor'] = hycal_scores(Q, T[ids], E[tr], y[tr], ids, nb, anchored=True)
        mu_all, P_all = xb.precision(E[tr], y[tr], ids)
        scores['GDA_transfer'] = 100 * Q @ T[ids].T + alpha * xb.lda(Q, mu_all, P_all)
        mus_a = torch.cat([mb, torch.stack([E[tr][y[tr] == k].double().mean(0) for k in novel])]) if len(novel) else mb
        scores['GDA_anchor'] = base_std(100 * Q @ T[ids].T, nb) + alpha_a * base_std(xb.lda(Q, mus_a, Pb), nb)
        n = len(tune); c = len(cal); ya = y[active]; br = ya < nb; rows = []
        for method in METHODS:
            s = scores[method]; assert bool(torch.isfinite(s).all())
            pred = f.prediction(s[n + c:], ids)
            row = dict(method=method, session=si, unconstrained_macro=f.macro(pred == ya, ya, ids),
                       unconstrained_top1=float(np.mean(pred == ya)), base=f.macro(pred[br] == ya[br], ya[br], base))
            if si == 0:
                refs[method + '_cal'] = f.prediction(s[n:n + c], base) == y[cal]
                refs[method + '_base_test'] = pred == ya
            else:
                own = refs[method + '_base_test']; now = pred[br] == ya[br]
                row['recovery'] = dict(G=float(np.mean(~own & now)), D=float(np.mean(own & ~now)))
                if track == 'certified':
                    vc = f.comp(s[:n], ids, y[tune], base, novel); cc = f.comp(s[n:n + c], ids, y[cal], base, novel); tc = f.comp(s[n + c:], ids, ya, base, novel)
                    cand = np.unique(np.quantile(vc['margin'], np.linspace(0, 1, 101))); w = f.risk.weights(y[cal], base)
                    row['sweep'] = f.sweep(s[n:n + c], tc, ids, y[cal], base, novel, refc, cand, closed)
                    row['sweep_self'] = f.sweep(s[n:n + c], tc, ids, y[cal], base, novel, refs[method + '_cal'],
                                                cand, f.macro(refs[method + '_base_test'], y[bt], base))
                    for kind in ['common', 'self']:
                        ref = refc if kind == 'common' else refs[method + '_cal']
                        floor = closed if kind == 'common' else f.macro(refs[method + '_base_test'], y[bt], base)
                        detail = f.paired.certify(cc, ref, w, cand, sessions=sessions - 1); d = detail['offset']
                        metrics = dict(base=floor, novel=0.) if d is None else f.evaluate(tc, d)
                        row[kind] = dict(**detail, **metrics, closed_base=floor, met=metrics['base'] >= .95 * floor - 1e-10, oracle=f.frontier(tc, .95 * floor))
            rows.append(row)
        if si == 0:
            np.savez_compressed(out / 'closed_references.npz', **refs)
        f.atomic(dict(rows=rows, session=si), out / f'session{si:02d}.json'); allrows += rows
        rec = ' '.join(f"{r['method']}:G={100 * r['recovery']['G']:.2f}" for r in rows if 'recovery' in r)
        print('PLUGIN_SESSION', ds, enc, track, seed, si, '/', sessions - 1, rec, round(time.monotonic() - start, 1), flush=True)
        del scores; torch.cuda.empty_cache()
    f.atomic(dict(complete=True, rows=len(allrows), methods=METHODS, gda_alpha=alpha, gda_anchor_alpha=alpha_a,
                  seconds=time.monotonic() - start), out / 'complete.json')
    print('PLUGIN_COMPLETE', ds, enc, track, seed, round(time.monotonic() - start, 1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(f.CFG)); p.add_argument('--encoder'); p.add_argument('--track', choices=['standard', 'certified'])
    p.add_argument('--seed', type=int); p.add_argument('--all', action='store_true')
    p.add_argument('--encoders', nargs='+', default=['b16']); p.add_argument('--tracks', nargs='+', default=['standard', 'certified'])
    a = p.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    if a.all:
        for ds in f.CFG:
            for enc in a.encoders:
                for track in a.tracks:
                    for seed in range(3):
                        run(ds, enc, track, seed)
    else:
        run(a.dataset, a.encoder, a.track, a.seed)


if __name__ == '__main__':
    main()
