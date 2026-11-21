"""Destruction/recovery of the four-source AnchorCE (full_gda) relative to the shared deployed reference.

Reviewer request (2026-11-10): Corollary 1 gives zero recovery relative to a readout's OWN closed head, but the
common contract certifies the four-source method against the closed head of the three-source AnchorCE core.
This script is a copy of anchor_gda.py that reruns the certified track unchanged and additionally records, at the
certified common-contract offset of every session, per-query base correctness of the extended readout (C_d) next
to the shared reference (R = archived core closed head), and the class-macro rates
    D = Pr(R=1, C_d=0),   G = Pr(R=0, C_d=1),   A_C = A_R - D + G
on the calibration queries and the base test queries. A fallback (no certified offset) keeps the reference, so
C_d = R. Nothing archived is modified; rows go to work/readouts_anchorg_recovery/<ds>/<enc>/certified/seed<s>/.
Each session row is compared with the archived anchor_gda.py row (work/readouts_anchorg) when it is present.
Usage: python anchor_gda_recovery.py --all --tracks certified
"""
from pathlib import Path
import json, argparse, hashlib, time
import numpy as np
import torch
import fscil_readouts as f
import extra_baselines as xb

GGRID = [0., .25, .5, 1., 2.]
VARIANT = 'full'; TEXT = 'names'; SWEEP = False
NAMES = {'full': 'full_gda', 'none': 'full_gda_none', 'fixed': 'full_gda_fixed', 'rawg': 'full_gda_rawg'}
MODE = {'full': 'full', 'none': 'none', 'fixed': 'fixed', 'rawg': 'full'}


def outdir():
    return f.D / ('readouts_anchorg_recovery' + ('' if VARIANT == 'full' else '_' + VARIANT) + ('_desc' if TEXT == 'desc' else ''))


def archived(ds, enc, track, seed, si):
    p = f.D / ('readouts_anchorg' + ('' if VARIANT == 'full' else '_' + VARIANT) + ('_desc' if TEXT == 'desc' else '')) / ds / enc / track / f'seed{seed}' / f'session{si:02d}.json'
    return json.loads(p.read_text())['rows'][0] if p.exists() else None


def gscale(v, nb, fixed=None):
    if VARIANT == 'rawg':
        return v
    if VARIANT == 'fixed':
        return (v - fixed[0]) / fixed[1]
    ref = v[:, :nb]; sd = ref.std(1, correction=0, keepdim=True).clamp_min(1e-6)
    return v / sd if VARIANT == 'none' else (v - ref.mean(1, keepdim=True)) / sd


def gmoments(v, nb):
    ref = v[:, :nb]
    return float(ref.mean()), float(ref.std(1, correction=0).mean().clamp_min(1e-6))


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


def rates(R, C, yy, base):
    return dict(D=f.macro(R & ~C, yy, base), G=f.macro(~R & C, yy, base), A_ref=f.macro(R, yy, base), A_C=f.macro(C, yy, base))


@torch.inference_mode()
def run(ds, enc, track, seed):
    start = time.monotonic()
    f.NB, f.WAY, f.S, f.NTUNE, f.NCAL, f.NC = f.CFG[ds]; NB, WAY, S, NTUNE, NCAL, NC = f.CFG[ds]
    METHOD = NAMES[VARIANT]; mode = MODE[VARIANT]
    src = f.D / ('readouts_desc' if TEXT == 'desc' else 'readouts') / ds / enc / track / f'seed{seed}'; out = outdir() / ds / enc / track / f'seed{seed}'
    if (out / 'complete.json').exists():
        print('RECOVERY_ALREADY_COMPLETE', VARIANT, TEXT, ds, enc, track, seed, flush=True); return
    assert (src / 'complete.json').exists(), f'missing archived task {src}'
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((src / 'selection.json').read_text())
    feature = f.D / 'features' / ds / enc / 'frozen.pt'; fh = hashlib.sha256(feature.read_bytes()).hexdigest()
    assert fh == sel['features_sha256']
    torch.manual_seed(2026 + seed)
    z = torch.load(feature, map_location='cpu', weights_only=False); E = z['E'].cuda(); T = z['T'].cuda()
    if TEXT == 'desc':
        T = f.norm(.5 * T + .5 * torch.load(f.D / 'description_features' / ds / f'{enc}.pt', map_location='cuda', weights_only=False).float())
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
    pids = np.r_[pb, pn]; co = f.components(E[tune], T[pids], Ppb, Ppn, Wp, fm, mode)
    ref_sel = f.select(co, pids, y[tune], pb, pn)
    arc = sel['selections'][mode]
    assert ref_sel['a'] == arc['a'] and ref_sel['u'] == arc['u'] and abs(ref_sel['harmonic'] - arc['harmonic']) < 1e-9 \
        and abs(ref_sel['offset'] - arc['offset']) <= 1e-4 * max(1., abs(arc['offset'])), (ref_sel, arc)
    mp, Pp = xb.precision(E[pfit], y[pfit], pb)
    Gp = xb.lda(E[tune], torch.cat([mp, raw_means(E[ps], y[ps], pn)]), Pp); Gp = gscale(Gp, len(pb), gmoments(Gp, len(pb)))
    par = select(co, Gp, pids, y[tune], pb, pn)

    W = f.ridge(E[actualfit], y[actualfit], base, lam); Pb = f.means(E[actualfit], y[actualfit], base); fm = []
    for v in [E[tune] @ T[base].T, f.lin(E[tune], W), E[tune] @ Pb.T]:
        fm.append((float(v.mean()), float(v.std(1, correction=0).mean().clamp_min(1e-6))))
    mb, P = xb.precision(E[actualfit], y[actualfit], base)
    gfix = gmoments(xb.lda(E[tune], mb, P), NB)
    f.atomic(dict(selection=par, archived_three_source=sel['selections'][mode], variant=VARIANT, text=TEXT, g_grid=GGRID, feature_sha256=fh), out / 'selection_anchorg.json')
    arch = dict(np.load(src / 'closed_references.npz')); rc = arch['full_cal']; rt = arch['full_base_test']
    test = np.flatnonzero(~z['train']); bt = test[y[test] < NB]; closed = sel['shared_reference_closed_test']
    assert abs(f.macro(rt, y[bt], base) - closed) < 1e-12
    trainidx = list(actualfit); refs = {}; allrows = []
    for si in range(S):
        if si:
            trainidx.extend(index[v] for v in sessions[str(si + 1)])
        n = np.arange(NB, NB + WAY * si); ids = np.r_[base, n]; active = test[y[test] < len(ids)]
        queries = np.r_[tune, cal, active]; Q = E[queries]; tr = np.array(trainidx); N = len(tune); C = len(cal)
        Pn = f.means(E[tr], y[tr], n) if len(n) else Pb[:0]
        co = f.components(Q, T[ids], Pb, Pn, W, fm, mode)
        mus = torch.cat([mb, raw_means(E[tr], y[tr], n)]) if len(n) else mb
        G = gscale(xb.lda(Q, mus, P), NB, gfix)
        s = fuse(co, G, par['a'], par['u'], par['g'], NB)
        if si == 0:
            base0 = s[:N + C, :NB].clone()
        else:
            assert torch.allclose(s[:N + C, :NB], base0, atol=1e-4), 'base scores changed'
        ss = s[N + C:].clone()
        if si: ss[:, NB:] += par['offset']
        ya = y[active]; br = ya < NB; pred = f.prediction(ss, ids)
        assert np.array_equal(ya[br], y[bt])
        row = dict(method=METHOD, session=si, unconstrained_macro=f.macro(pred == ya, ya, ids), unconstrained_top1=float(np.mean(pred == ya)),
                   base=f.macro(pred[br] == ya[br], ya[br], base))
        if si == 0:
            refs[METHOD + '_cal'] = f.prediction(s[N:N + C], base) == y[cal]; refs[METHOD + '_base_test'] = pred == ya
            np.savez_compressed(out / 'closed_references.npz', **refs)
            # Closed-head comparison with the shared reference (no novel classes yet).
            row['recovery_common'] = dict(cal=rates(rc, refs[METHOD + '_cal'], y[cal], base), base_test=rates(rt, refs[METHOD + '_base_test'], y[bt], base), fallback=False)
        elif track == 'certified':
            vc = f.comp(s[:N], ids, y[tune], base, n); cc = f.comp(s[N:N + C], ids, y[cal], base, n); tc = f.comp(s[N + C:], ids, ya, base, n)
            cand = np.unique(np.quantile(vc['margin'], np.linspace(0, 1, 101))); w = f.risk.weights(y[cal], base)
            for kind in ['common', 'self']:
                ref = rc if kind == 'common' else refs[METHOD + '_cal']
                floor = closed if kind == 'common' else f.macro(refs[METHOD + '_base_test'], y[bt], base)
                detail = f.paired.certify(cc, ref, w, cand, sessions=S - 1); d = detail['offset']
                metrics = dict(base=floor, novel=0.) if d is None else f.evaluate(tc, d)
                row[kind] = dict(**detail, **metrics, closed_base=floor, met=metrics['base'] >= .95 * floor - 1e-10, oracle=f.frontier(tc, .95 * floor))
                if kind == 'common':
                    if d is None:  # fallback keeps the deployed reference
                        ccal, ctest = rc.copy(), rt.copy()
                    else:
                        ccal = cc['base_correct'] & ~f.risk.choose_new(cc, d)
                        ctest = (tc['base_correct'] & ~f.risk.choose_new(tc, d))[br]
                    rec = dict(cal=rates(rc, ccal, y[cal], base), base_test=rates(rt, ctest, y[bt], base), fallback=d is None)
                    if d is not None:
                        assert abs(rec['base_test']['A_C'] - metrics['base']) < 1e-9, (rec, metrics)
                    for part in ['cal', 'base_test']:
                        r = rec[part]; assert abs(r['A_C'] - (r['A_ref'] - r['D'] + r['G'])) < 1e-9, r
                    row['recovery_common'] = rec
                    np.savez_compressed(out / f'recovery_session{si:02d}.npz', cal_ref=rc, cal_readout=ccal, cal_y=y[cal], test_ref=rt, test_readout=ctest, test_y=y[bt])
        old = archived(ds, enc, track, seed, si)
        if old is not None:
            same = abs(old['unconstrained_macro'] - row['unconstrained_macro']) < 1e-6
            if 'common' in row and 'common' in old:
                same = same and abs(old['common']['novel'] - row['common']['novel']) < 1e-6 and abs(old['common']['base'] - row['common']['base']) < 1e-6
            row['matches_archived_anchorg'] = bool(same)
            if not same:
                print('RECOVERY_ARCHIVE_MISMATCH', ds, enc, track, seed, si, flush=True)
        f.atomic(dict(rows=[row], session=si), out / f'session{si:02d}.json'); allrows.append(row)
        rc_ = row.get('recovery_common', {}).get('cal', {})
        print('RECOVERY_SESSION', ds, enc, track, seed, si, '/', S - 1, 'g=%s' % par['g'],
              'D=%.4f G=%.4f' % (rc_.get('D', float('nan')), rc_.get('G', float('nan'))), flush=True)
    f.atomic(dict(complete=True, rows=len(allrows), selection=par, seconds=time.monotonic() - start), out / 'complete.json')
    print('RECOVERY_COMPLETE', VARIANT, TEXT, ds, enc, track, seed, 'g=%s a=%s u=%s' % (par['g'], par['a'], par['u']), round(time.monotonic() - start, 1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(f.CFG)); p.add_argument('--encoder'); p.add_argument('--track', choices=['standard', 'certified'])
    p.add_argument('--seed', type=int); p.add_argument('--all', action='store_true')
    p.add_argument('--variant', choices=list(NAMES), default='full'); p.add_argument('--text', choices=['names', 'desc'], default='names')
    p.add_argument('--encoders', nargs='+', default=['b16', 'l14', 'b32']); p.add_argument('--tracks', nargs='+', default=['certified'])
    a = p.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    global VARIANT, TEXT; VARIANT = a.variant; TEXT = a.text
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
