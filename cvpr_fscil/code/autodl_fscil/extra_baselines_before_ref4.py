"""Two additional training-free baselines replayed on the archived standard-FSCIL splits.

GDA  (Wang et al., ICLR 2024, "A hard-to-beat baseline for training-free CLIP-based adaptation"):
     Gaussian discriminant analysis with a shared covariance and the empirical-Bayes precision
     D * ((N-1) S + tr(S) I)^-1, fused with CLIP zero-shot logits, 100 q^T t_c + alpha * vision.
KLDA (Momeni et al., 2025, kernel LDA over foundation-model features): random Fourier features of
     an RBF kernel, class means and one shared covariance accumulated over all data seen so far,
     LDA scoring.

Both are closed-form (no gradient training). Hyper-parameters (GDA alpha, KLDA gamma) are chosen on
base-only tuning queries, exactly like the other readouts. The archived run (selection.json,
closed_references.npz) is read only; new rows go to work/readouts_extra/<ds>/<enc>/<track>/seed<s>/.
Usage: python extra_baselines.py --dataset cifar100 --encoder b16 --track certified --seed 0
"""
from pathlib import Path
import json, argparse, hashlib, time
import numpy as np
import torch
import torch.nn.functional as F
import fscil_readouts as f

OUT = f.D / 'readouts_extra'
SWEEP = False  # --sweep: write to readouts_extra_sweep and certify over r and calibration caps
ALPHAS = [0.] + (10. ** np.arange(-2, 3.01, .5)).tolist()
GAMMAS = [.5, 1., 2., 5., 10.]
RFF_DIM = 5000


def precision(X, y, classes):
    """Shared within-class covariance and its empirical-Bayes precision (GDA / KLDA)."""
    X = X.double()
    mus = torch.stack([X[y == k].mean(0) for k in classes])
    C = X - mus[torch.as_tensor(np.searchsorted(classes, y), device=X.device)]
    S = C.T @ C / max(len(X) - 1, 1)
    d = S.shape[0]
    A = (len(X) - 1) * S + torch.trace(S) * torch.eye(d, dtype=S.dtype, device=S.device)
    return mus, d * torch.cholesky_inverse(torch.linalg.cholesky(A))


def lda(Q, mus, P):
    W = mus @ P
    return (Q.double() @ W.T - .5 * (W * mus).sum(1)).float()


def rff(gamma, dim, seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    W = torch.randn(dim, RFF_DIM, generator=g, device='cuda') * (2 * gamma) ** .5
    b = torch.rand(RFF_DIM, generator=g, device='cuda') * 2 * np.pi
    return lambda X: (2 / RFF_DIM) ** .5 * torch.cos(X @ W + b)


def macro_acc(scores, classes, yq):
    return f.macro(f.prediction(scores, classes) == yq, yq, classes)


@torch.inference_mode()
def run(ds, enc, track, seed):
    start = time.monotonic()
    nb, way, sessions, ntune, ncal, _ = f.CFG[ds]
    f.NB, f.WAY, f.S, f.NTUNE, f.NCAL, f.NC = f.CFG[ds]
    src = f.D / 'readouts' / ds / enc / track / f'seed{seed}'
    out = (f.D / 'readouts_extra_sweep' if SWEEP else OUT) / ds / enc / track / f'seed{seed}'
    if (out / 'complete.json').exists():
        print('EXTRA_ALREADY_COMPLETE', ds, enc, track, seed, flush=True); return
    assert (src / 'complete.json').exists(), f'missing archived task {src}'
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((src / 'selection.json').read_text())
    feature = f.D / 'features' / ds / enc / 'frozen.pt'
    fh = hashlib.sha256(feature.read_bytes()).hexdigest()
    assert fh == sel['features_sha256']
    z = torch.load(feature, map_location='cpu', weights_only=False)
    E = z['E'].cuda(); T = z['T'].cuda(); y = np.asarray(z['y']); iid = np.asarray(z['ids'])
    index = {int(v): i for i, v in enumerate(iid)}; base = np.arange(nb)
    # Rebuild the archived splits with the original RNG and check them against selection.json.
    rng = np.random.default_rng(28012026 + seed)
    official = np.array([index[v] for v in z['sessions']['1']]); tune = []; cal = []; fit = []
    for k in base:
        q = rng.permutation(official[y[official] == k]); tune.extend(q[:ntune]); cal.extend(q[ntune:ntune + ncal]); fit.extend(q[ntune + ncal:])
    tune, cal, fit = map(np.array, (tune, cal, fit))
    actualfit = fit if track == 'certified' else official
    selectionfit = fit if track == 'certified' else np.setdiff1d(official, tune)
    assert np.array_equal(iid[actualfit], np.asarray(sel['fit_ids'])) and np.array_equal(iid[tune], np.asarray(sel['tune_ids']))
    if track == 'certified':
        assert np.array_equal(iid[cal], np.asarray(sel['cal_ids']))
    refs_arch = dict(np.load(src / 'closed_references.npz')); refc = refs_arch['full_cal']
    test = np.flatnonzero(~z['train']); bt = test[y[test] < nb]; closed = sel['shared_reference_closed_test']
    assert abs(f.macro(refs_arch['full_base_test'], y[bt], base) - closed) < 1e-12

    # Base-only hyper-parameter selection on the tuning queries (classes = base, session 0).
    yt = y[tune]; Et = E[tune]
    mus, P = precision(E[selectionfit], y[selectionfit], base)
    vis = lda(Et, mus, P); zs = 100 * Et @ T[base].T
    alpha = max(ALPHAS, key=lambda a: (macro_acc(zs + a * vis, base, yt), -a))
    gacc = {}
    for g in GAMMAS:
        phi = rff(g, E.shape[1], 9000 + seed)
        m, Pk = precision(phi(E[selectionfit]), y[selectionfit], base)
        gacc[g] = macro_acc(lda(phi(Et), m, Pk), base, yt)
        del m, Pk; torch.cuda.empty_cache()
    gamma = max(GAMMAS, key=lambda g: (gacc[g], -g))
    phi = rff(gamma, E.shape[1], 9000 + seed)
    f.atomic(dict(gda_alpha=alpha, klda_gamma=gamma, klda_tune_acc=gacc, rff_dim=RFF_DIM, feature_sha256=fh,
                  archived_selection_sha256=hashlib.sha256((src / 'selection.json').read_bytes()).hexdigest()), out / 'selection_extra.json')

    trainidx = list(actualfit); refs = {}; allrows = []
    for si in range(sessions):
        if si:
            trainidx.extend(index[v] for v in z['sessions'][str(si + 1)])
        novel = np.arange(nb, nb + way * si); ids = np.r_[base, novel]
        active = test[y[test] < len(ids)]; queries = np.r_[tune, cal, active]; tr = np.asarray(trainidx)
        Q = E[queries]; scores = {}
        mus, P = precision(E[tr], y[tr], ids)
        scores['GDA_transfer'] = 100 * Q @ T[ids].T + alpha * lda(Q, mus, P)
        mk, Pk = precision(phi(E[tr]), y[tr], ids)
        scores['KLDA_transfer'] = torch.cat([lda(phi(Q[i:i + 4096]), mk, Pk) for i in range(0, len(Q), 4096)])
        del mus, P, mk, Pk; torch.cuda.empty_cache()
        n = len(tune); c = len(cal); ya = y[active]; br = ya < nb; rows = []
        for method, s in scores.items():
            assert bool(torch.isfinite(s).all())
            pred = f.prediction(s[n + c:], ids)
            row = dict(method=method, session=si, unconstrained_macro=f.macro(pred == ya, ya, ids),
                       unconstrained_top1=float(np.mean(pred == ya)), base=f.macro(pred[br] == ya[br], ya[br], base))
            if si == 0:
                refs[method + '_cal'] = f.prediction(s[n:n + c], base) == y[cal]
                refs[method + '_base_test'] = pred == ya
            elif track == 'certified':
                vc = f.comp(s[:n], ids, y[tune], base, novel); cc = f.comp(s[n:n + c], ids, y[cal], base, novel); tc = f.comp(s[n + c:], ids, ya, base, novel)
                cand = np.unique(np.quantile(vc['margin'], np.linspace(0, 1, 101))); w = f.risk.weights(y[cal], base)
                if SWEEP:
                    row['sweep'] = f.sweep(s[n:n + c], tc, ids, y[cal], base, novel, refc, cand, closed)
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
        print('EXTRA_SESSION', ds, enc, track, seed, si, '/', sessions - 1,
              ' '.join(f"{r['method']}={100 * r['unconstrained_top1']:.2f}" for r in rows), round(time.monotonic() - start, 1), flush=True)
        del scores; torch.cuda.empty_cache()
    f.atomic(dict(complete=True, rows=len(allrows), methods=['GDA_transfer', 'KLDA_transfer'], gda_alpha=alpha,
                  klda_gamma=gamma, seconds=time.monotonic() - start), out / 'complete.json')
    print('EXTRA_COMPLETE', ds, enc, track, seed, round(time.monotonic() - start, 1), flush=True)


@torch.inference_mode()
def selftest():
    """Synthetic Gaussian classes: both closed-form readouts must classify almost perfectly."""
    torch.manual_seed(0); K, d, n = 20, 64, 200
    centers = F.normalize(torch.randn(K, d, device='cuda'), dim=1)
    X = F.normalize(centers.repeat_interleave(n, 0) + .08 * torch.randn(K * n, d, device='cuda'), dim=1)
    y = np.repeat(np.arange(K), n); classes = np.arange(K); te = np.arange(0, K * n, 7); trm = np.setdiff1d(np.arange(K * n), te)
    mus, P = precision(X[trm], y[trm], classes); a1 = macro_acc(lda(X[te], mus, P), classes, y[te])
    phi = rff(1., d, 0); m, Pk = precision(phi(X[trm]), y[trm], classes); a2 = macro_acc(lda(phi(X[te]), m, Pk), classes, y[te])
    print('SELFTEST gda_acc=%.3f klda_acc=%.3f' % (a1, a2), flush=True)
    assert a1 > .95 and a2 > .9, 'self-test failed'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(f.CFG)); p.add_argument('--encoder'); p.add_argument('--track', choices=['standard', 'certified'])
    p.add_argument('--seed', type=int); p.add_argument('--selftest', action='store_true'); p.add_argument('--sweep', action='store_true')
    p.add_argument('--encoders', nargs='+', default=['b16', 'l14', 'b32']); p.add_argument('--tracks', nargs='+', default=['standard', 'certified']); p.add_argument('--all', action='store_true', help='run every archived task (skips completed ones)')
    a = p.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    global SWEEP; SWEEP = a.sweep
    if SWEEP and a.tracks != ['certified']:
        p.error('--sweep requires --tracks certified')
    if a.selftest:
        selftest(); return
    if a.all:
        for ds in f.CFG:
            for enc in a.encoders:
                for track in a.tracks:
                    for seed in range(3):
                        if (f.D / 'readouts' / ds / enc / track / f'seed{seed}' / 'complete.json').exists():
                            run(ds, enc, track, seed)
                        else:
                            print('EXTRA_SKIP_NO_ARCHIVE', ds, enc, track, seed, flush=True)
    else:
        run(a.dataset, a.encoder, a.track, a.seed)


if __name__ == '__main__':
    main()
