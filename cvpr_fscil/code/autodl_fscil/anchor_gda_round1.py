"""AnchorCE with a fourth, covariance-aware evidence source (base-frozen LDA), on the standard FSCIL splits.

q_G(x,c) = x^T P mu_c - mu_c^T P mu_c / 2, with the shared precision P (GDA empirical-Bayes estimator)
estimated on base fitting data only and then frozen; mu_c are base means and few-shot support means.
Like the other sources, q_G is standardized per query against its base bank (the base-class scores)
and enters both groups with one coefficient g:
    s(x,b) = T^ + a L^ + g G^ ,   s(x,n) = T^ + u P^ + g G^ + d.
Because P and the base means are fixed, every base score stays invariant as classes arrive (zero
recovery is kept). (a,u,g,d) are selected on the same base-only pseudo-expansion episode; g=0 is the
original AnchorCE, and the script checks that its g=0 selection reproduces the archived one exactly.
Rows (method 'full_gda') go to work/readouts_anchorg/<ds>/<enc>/<track>/seed<s>/.
Usage: python anchor_gda.py --all     (or --dataset/--encoder/--track/--seed for one task)
"""
from pathlib import Path
import json, argparse, hashlib, time
import numpy as np
import torch
import fscil_readouts as f
import extra_baselines as xb

OUT = f.D / 'readouts_anchorg'
GGRID = [0., .25, .5, 1., 2.]
METHOD = 'full_gda'


def stdz(v, nb):
    ref = v[:, :nb]
    return (v - ref.mean(1, keepdim=True)) / ref.std(1, correction=0, keepdim=True).clamp_min(1e-6)


def fuse(co, G, a, u, g, nb):
    s = f.fusion(co, a, u, nb)
    return s + g * G if g else s


def select(co, G, ids, y, b, n):
    best = (-1, None)
    for g in GGRID:
        for a in f.GRID:
            for u in f.GRID:
                cc = f.comp(fuse(co, G, a, u, g, len(b)), ids, y, b, n)
                d = np.quantile(cc['margin'], np.linspace(0, 1, 101))
                ch = (d[:, None] > cc['margin'][None]) | ((d[:, None] == cc['margin'][None]) & (cc['novel_prediction'][None] < cc['base_prediction'][None]))
                ab = (~ch) @ (cc['weights'] * cc['base_correct']); an = ch @ (cc['weights'] * cc['novel_correct'])
                h = np.divide(2 * ab * an, ab + an, out=np.zeros_like(ab), where=ab + an > 0); j = int(h.argmax())
                if h[j] > best[0] + 1e-12:
                    best = (h[j], dict(a=a, u=u, g=g, offset=float(d[j]), harmonic=float(h[j])))
    return best[1]


def raw_means(E, y, classes):
    return torch.stack([E[y == k].double().mean(0) for k in classes])


@torch.inference_mode()
def run(ds, enc, track, seed):
    start = time.monotonic()
    f.NB, f.WAY, f.S, f.NTUNE, f.NCAL, f.NC = f.CFG[ds]; NB, WAY, S, NTUNE, NCAL, NC = f.CFG[ds]
    src = f.D / 'readouts' / ds / enc / track / f'seed{seed}'; out = OUT / ds / enc / track / f'seed{seed}'
    if (out / 'complete.json').exists():
        print('ANCHORG_ALREADY_COMPLETE', ds, enc, track, seed, flush=True); return
    assert (src / 'complete.json').exists(), f'missing archived task {src}'
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((src / 'selection.json').read_text())
    feature = f.D / 'features' / ds / enc / 'frozen.pt'; fh = hashlib.sha256(feature.read_bytes()).hexdigest()
    assert fh == sel['features_sha256']
    torch.manual_seed(2026 + seed)
    z = torch.load(feature, map_location='cpu', weights_only=False); E = z['E'].cuda(); T = z['T'].cuda()
    # --- identical split / pseudo-episode construction to fscil_readouts.run (same RNG stream) ---
    y = np.asarray(z['y']); iid = np.asarray(z['ids']); sessions = z['sessions']; base = np.arange(NB)
    rng = np.random.default_rng(28012026 + seed); index = {int(v): i for i, v in enumerate(iid)}
    official = np.array([index[v] for v in sessions['1']]); fit = []; tune = []; cal = []
    for k in base:
        q = rng.permutation(official[y[official] == k]); tune.extend(q[:NTUNE]); cal.extend(q[NTUNE:NTUNE + NCAL]); fit.extend(q[NTUNE + NCAL:])
    fit, tune, cal = [np.array(v) for v in [fit, tune, cal]]
    selectionfit = fit if track == 'certified' else np.setdiff1d(official, tune); actualfit = fit if track == 'certified' else official
    assert np.array_equal(iid[actualfit], np.asarray(sel['fit_ids'])) and np.array_equal(iid[tune], np.asarray(sel['tune_ids']))

    def fitted(ix, classes, valid):
        best = (-1, None, None)
        for lam in [.01, .1, 1., 10.]:
            w = f.ridge(E[ix], y[ix], classes, lam); acc = f.macro(f.prediction(f.lin(E[valid], w), classes) == y[valid], y[valid], classes)
            if acc > best[0]: best = (acc, w, lam)
        return best[1:]
    Ws, lam = fitted(selectionfit, base, tune); pb = np.arange(NB // 2); pn = np.arange(NB // 2, NB)
    pfit = selectionfit[np.isin(y[selectionfit], pb)]; Wp, plam = fitted(pfit, pb, tune[np.isin(y[tune], pb)])
    ps = np.concatenate([rng.choice(selectionfit[y[selectionfit] == k], 5, replace=False) for k in pn])
    Ppb = f.means(E[pfit], y[pfit], pb); Ppn = f.means(E[ps], y[ps], pn); fm = []
    for v in [E[tune] @ T[pb].T, f.lin(E[tune], Wp), E[tune] @ Ppb.T]:
        fm.append((float(v.mean()), float(v.std(1, correction=0).mean().clamp_min(1e-6))))
    assert lam == sel['ridge_lambda'] and plam == sel['pseudo_lambda']
    pids = np.r_[pb, pn]; co = f.components(E[tune], T[pids], Ppb, Ppn, Wp, fm, 'full')
    # Integrity: the g=0 slice must reproduce the archived AnchorCE selection exactly.
    ref_sel = f.select(co, pids, y[tune], pb, pn)
    arc = sel['selections']['full']
    assert ref_sel['a'] == arc['a'] and ref_sel['u'] == arc['u'] and abs(ref_sel['offset'] - arc['offset']) < 1e-5, (ref_sel, arc)
    mp, Pp = xb.precision(E[pfit], y[pfit], pb)
    Gp = stdz(xb.lda(E[tune], torch.cat([mp, raw_means(E[ps], y[ps], pn)]), Pp), len(pb))
    par = select(co, Gp, pids, y[tune], pb, pn)

    # --- deployed readout: same ridge/prototypes/moments as AnchorCE, plus the base-frozen LDA source ---
    W = f.ridge(E[actualfit], y[actualfit], base, lam); Pb = f.means(E[actualfit], y[actualfit], base); fm = []
    for v in [E[tune] @ T[base].T, f.lin(E[tune], W), E[tune] @ Pb.T]:
        fm.append((float(v.mean()), float(v.std(1, correction=0).mean().clamp_min(1e-6))))
    mb, P = xb.precision(E[actualfit], y[actualfit], base)  # frozen after session 0
    f.atomic(dict(selection=par, archived_full=sel['selections']['full'], g_grid=GGRID, feature_sha256=fh), out / 'selection_anchorg.json')
    arch = dict(np.load(src / 'closed_references.npz')); rc = arch['full_cal']
    test = np.flatnonzero(~z['train']); bt = test[y[test] < NB]; closed = sel['shared_reference_closed_test']
    assert abs(f.macro(arch['full_base_test'], y[bt], base) - closed) < 1e-12
    trainidx = list(actualfit); refs = {}; allrows = []
    for si in range(S):
        if si:
            trainidx.extend(index[v] for v in sessions[str(si + 1)])
        n = np.arange(NB, NB + WAY * si); ids = np.r_[base, n]; active = test[y[test] < len(ids)]
        queries = np.r_[tune, cal, active]; Q = E[queries]; tr = np.array(trainidx); N = len(tune); C = len(cal)
        Pn = f.means(E[tr], y[tr], n) if len(n) else Pb[:0]
        co = f.components(Q, T[ids], Pb, Pn, W, fm, 'full')
        mus = torch.cat([mb, raw_means(E[tr], y[tr], n)]) if len(n) else mb
        G = stdz(xb.lda(Q, mus, P), NB)
        s = fuse(co, G, par['a'], par['u'], par['g'], NB)
        if si == 0:  # zero-recovery check on the fixed tuning+calibration queries: base scores never change
            base0 = s[:N + C, :NB].clone()
        else:
            assert torch.allclose(s[:N + C, :NB], base0, atol=1e-4), 'base scores changed'
        ss = s[N + C:].clone()
        if si: ss[:, NB:] += par['offset']
        ya = y[active]; br = ya < NB; pred = f.prediction(ss, ids)
        row = dict(method=METHOD, session=si, unconstrained_macro=f.macro(pred == ya, ya, ids), unconstrained_top1=float(np.mean(pred == ya)),
                   base=f.macro(pred[br] == ya[br], ya[br], base))
        if si == 0:
            refs[METHOD + '_cal'] = f.prediction(s[N:N + C], base) == y[cal]; refs[METHOD + '_base_test'] = pred == ya
            np.savez_compressed(out / 'closed_references.npz', **refs)
        elif track == 'certified':
            vc = f.comp(s[:N], ids, y[tune], base, n); cc = f.comp(s[N:N + C], ids, y[cal], base, n); tc = f.comp(s[N + C:], ids, ya, base, n)
            cand = np.unique(np.quantile(vc['margin'], np.linspace(0, 1, 101))); w = f.risk.weights(y[cal], base)
            for kind in ['common', 'self']:
                ref = rc if kind == 'common' else refs[METHOD + '_cal']
                floor = closed if kind == 'common' else f.macro(refs[METHOD + '_base_test'], y[bt], base)
                detail = f.paired.certify(cc, ref, w, cand, sessions=S - 1); d = detail['offset']
                metrics = dict(base=floor, novel=0.) if d is None else f.evaluate(tc, d)
                row[kind] = dict(**detail, **metrics, closed_base=floor, met=metrics['base'] >= .95 * floor - 1e-10, oracle=f.frontier(tc, .95 * floor))
        f.atomic(dict(rows=[row], session=si), out / f'session{si:02d}.json'); allrows.append(row)
        print('ANCHORG_SESSION', ds, enc, track, seed, si, '/', S - 1, 'top1=%.2f' % (100 * row['unconstrained_top1']), 'g=%s' % par['g'], flush=True)
    f.atomic(dict(complete=True, rows=len(allrows), selection=par, seconds=time.monotonic() - start), out / 'complete.json')
    print('ANCHORG_COMPLETE', ds, enc, track, seed, 'g=%s a=%s u=%s' % (par['g'], par['a'], par['u']), round(time.monotonic() - start, 1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(f.CFG)); p.add_argument('--encoder'); p.add_argument('--track', choices=['standard', 'certified'])
    p.add_argument('--seed', type=int); p.add_argument('--all', action='store_true')
    a = p.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    if a.all:
        for ds in f.CFG:
            for enc in ['b16', 'l14', 'b32']:
                for track in ['standard', 'certified']:
                    for seed in range(3):
                        run(ds, enc, track, seed)
    else:
        run(a.dataset, a.encoder, a.track, a.seed)


if __name__ == '__main__':
    main()
