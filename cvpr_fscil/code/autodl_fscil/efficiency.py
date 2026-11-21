"""Wall-clock cost, peak GPU memory and stored state of every training-free readout (standard track).

For each incremental session we time what a deployed system must do when new classes arrive:
absorb the 5-shot supports and score the full test set of all seen classes. Base fitting (session 0)
is timed separately. Each method keeps only the state it needs and updates it incrementally, as its
paper describes. Timing uses torch.cuda.synchronize; the whole sequence runs once as warm-up and then
REPEAT times, reporting the median. Output: work/efficiency.json and a printed table.
Usage: python efficiency.py --datasets mini cub200 --encoder b16
"""
from pathlib import Path
import json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
import fscil_readouts as f
import extra_baselines as xb

REPEAT = 3


def sync():
    torch.cuda.synchronize()


def nbytes(*ts):
    return int(sum(t.numel() * t.element_size() for t in ts))


class Scatter:
    """Shared within-class covariance kept as an incrementally updated scatter matrix (GDA, KLDA)."""
    def __init__(self, d):
        self.M = torch.zeros(d, d, dtype=torch.float64, device='cuda'); self.n = 0; self.mus = []

    def add(self, X):
        X = X.double(); mu = X.mean(0); C = X - mu
        self.M += C.T @ C; self.n += len(X); self.mus.append(mu)

    def precision(self):
        S = self.M / max(self.n - 1, 1); d = S.shape[0]
        A = (self.n - 1) * S + torch.trace(S) * torch.eye(d, dtype=S.dtype, device=S.device)
        return torch.stack(self.mus), d * torch.cholesky_inverse(torch.linalg.cholesky(A))


def build(ds, enc, seed=0):
    nb, way, sessions, ntune, ncal, nc = f.CFG[ds]; f.NB = nb
    src = f.D / 'readouts' / ds / enc / 'standard' / f'seed{seed}'
    sel = json.loads((src / 'selection.json').read_text())
    z = torch.load(f.D / 'features' / ds / enc / 'frozen.pt', map_location='cpu', weights_only=False)
    E = z['E'].cuda(); T = z['T'].cuda(); y = np.asarray(z['y']); iid = np.asarray(z['ids'])
    index = {int(v): i for i, v in enumerate(iid)}
    fit = np.array([index[int(v)] for v in sel['fit_ids']]); tune = np.array([index[int(v)] for v in sel['tune_ids']])
    test = np.flatnonzero(~z['train'])
    sup = [np.array([index[v] for v in z['sessions'][str(si + 1)]]) for si in range(1, sessions)]
    extra = f.D / 'readouts_extra' / ds / enc / 'standard' / f'seed{seed}' / 'selection_extra.json'
    ex = json.loads(extra.read_text()) if extra.exists() else dict(gda_alpha=1., klda_gamma=1.)
    return dict(key=Path(ds) / enc / 'standard' / f'seed{seed}', nb=nb, way=way, sessions=sessions, nc=nc, sel=sel, E=E, T=T, Ecpu=z['E'], y=y, fit=fit, tune=tune, test=test, sup=sup, ex=ex)


def methods(B):
    """Each entry: (name, base_fit(), session(new_support_idx, ids, queries) -> scores, state())."""
    E, T, y, nb, fit, sel = B['E'], B['T'], B['y'], B['nb'], B['fit'], B['sel']; base = np.arange(nb)
    M = {}

    def anchor(mode):
        st = {}
        def fit_():
            W = f.ridge(E[fit], y[fit], base, sel['ridge_lambda']); Pb = f.means(E[fit], y[fit], base); fm = []
            for v in [E[B['tune']] @ T[base].T, f.lin(E[B['tune']], W), E[B['tune']] @ Pb.T]:
                fm.append((float(v.mean()), float(v.std(1, correction=0).mean().clamp_min(1e-6))))
            st.update(W=W, Pb=Pb, fm=fm, Pn=Pb[:0])
        def sess(new, ids, q):
            st['Pn'] = torch.cat([st['Pn'], f.means(E[new], y[new], np.unique(y[new]))])
            p = sel['selections'][mode]
            co = f.components(E[q], T[ids], st['Pb'], st['Pn'], st['W'], st['fm'], mode)
            return f.fusion(co, p['a'], p['u'], nb, mode == 'dense')
        return fit_, sess, lambda: nbytes(st['W'], st['Pb'], st['Pn'])
    M['AnchorCE-3src'] = anchor('full'); M['Dense'] = anchor('dense')

    # Proposed four-source AnchorCE: the three anchored sources plus the base-frozen LDA source.
    sg = json.loads((f.D / 'readouts_anchorg' / B['key'] / 'selection_anchorg.json').read_text())['selection']
    st0 = {}
    def a4_fit():
        fit3, _, _ = anchor_full
        fit3(); st0['mb'], st0['P'] = xb.precision(E[fit], y[fit], base); st0['mn'] = st0['mb'][:0]
    def a4_sess(new, ids, q):
        _, sess3, _ = anchor_full
        st0['mn'] = torch.cat([st0['mn'], torch.stack([E[new][y[new] == k].double().mean(0) for k in np.unique(y[new])])])
        p3 = sel['selections']['full']; sel['selections']['full'] = dict(p3, a=sg['a'], u=sg['u'])
        s3 = sess3(new, ids, q); sel['selections']['full'] = p3
        v = xb.lda(E[q], torch.cat([st0['mb'], st0['mn']]), st0['P']); ref = v[:, :nb]
        return s3 + sg['g'] * (v - ref.mean(1, keepdim=True)) / ref.std(1, correction=0, keepdim=True).clamp_min(1e-6)
    anchor_full = anchor('full')
    M['AnchorCE'] = (a4_fit, a4_sess, lambda: anchor_full[2]() + nbytes(st0['mb'], st0['mn'], st0['P']))

    st1 = {}
    def sc_fit(): st1['P'] = f.means(E[fit], y[fit], base)
    def sc_sess(new, ids, q):
        st1['P'] = torch.cat([st1['P'], f.means(E[new], y[new], np.unique(y[new]))]); return E[q] @ st1['P'].T
    M['SimpleCIL'] = (sc_fit, sc_sess, lambda: nbytes(st1['P']))

    st2 = {}
    def teen_fit(): st2['P'] = f.means(E[fit], y[fit], base); st2['Pb'] = st2['P'].clone()
    def teen_sess(new, ids, q):
        Pn = f.means(E[new], y[new], np.unique(y[new]))
        Pn = f.norm(f.TEEN_ALPHA * Pn + (1 - f.TEEN_ALPHA) * (torch.softmax(f.TEEN_TAU * (Pn @ st2['Pb'].T), 1) @ st2['Pb']))
        st2['P'] = torch.cat([st2['P'], Pn]); return E[q] @ st2['P'].T
    M['TEEN'] = (teen_fit, teen_sess, lambda: nbytes(st2['P']))

    st3 = {}
    def fecam_fit():
        st3['cache'] = {}; st3['tr'] = list(fit)
        f.fecam_cpu(B['Ecpu'][:1], B['Ecpu'][fit], y[fit], base.copy(), st3['cache'])
    def fecam_sess(new, ids, q):
        st3['tr'].extend(new); tr = np.array(st3['tr'])
        return f.fecam_cpu(B['Ecpu'][q], B['Ecpu'][tr], y[tr], ids.copy(), st3['cache']).cuda()
    M['FeCAM'] = (fecam_fit, fecam_sess, lambda: sum(nbytes(*v[1:]) for k, v in st3['cache'].items() if k != '__pool'))

    st4 = {}
    def rp_fit():
        g = torch.Generator(device='cuda').manual_seed(0); st4['RP'] = torch.randn(E.shape[1], 10000, generator=g, device='cuda')
        H = F.relu(E[fit] @ st4['RP']); Y = F.one_hot(torch.as_tensor(y[fit], device='cuda'), B['nc']).float()
        st4['G'] = H.T @ H; st4['C'] = H.T @ Y; st4['eye'] = torch.eye(10000, device='cuda')
    def rp_sess(new, ids, q):
        H = F.relu(E[new] @ st4['RP']); st4['G'] += H.T @ H
        st4['C'] += H.T @ F.one_hot(torch.as_tensor(y[new], device='cuda'), B['nc']).float()
        coef = torch.linalg.solve(st4['G'] + st4['eye'], st4['C'])
        return (F.relu(E[q] @ st4['RP']) @ coef)[:, :len(ids)]
    M['RanPAC'] = (rp_fit, rp_sess, lambda: nbytes(st4['RP'], st4['G'], st4['C']))

    st5 = {}
    def gda_fit(): st5['S'] = Scatter(E.shape[1]); [st5['S'].add(E[fit][y[fit] == k]) for k in base]
    def gda_sess(new, ids, q):
        for k in np.unique(y[new]): st5['S'].add(E[new][y[new] == k])
        mus, P = st5['S'].precision(); return 100 * E[q] @ T[ids].T + B['ex']['gda_alpha'] * xb.lda(E[q], mus, P)
    M['GDA'] = (gda_fit, gda_sess, lambda: nbytes(st5['S'].M, *st5['S'].mus))

    st6 = {}
    def klda_fit():
        st6['phi'] = xb.rff(B['ex']['klda_gamma'], E.shape[1], 9000); st6['S'] = Scatter(xb.RFF_DIM)
        for k in base: st6['S'].add(st6['phi'](E[fit][y[fit] == k]))
    def klda_sess(new, ids, q):
        for k in np.unique(y[new]): st6['S'].add(st6['phi'](E[new][y[new] == k]))
        mus, P = st6['S'].precision()
        return torch.cat([xb.lda(st6['phi'](E[q][i:i + 4096]), mus, P) for i in range(0, len(q), 4096)])
    M['KLDA'] = (klda_fit, klda_sess, lambda: nbytes(st6['S'].M, *st6['S'].mus))

    st7 = {}
    def hy_fit(): st7['tr'] = list(fit)
    def hy_sess(new, ids, q):
        st7['tr'].extend(new); tr = np.array(st7['tr']); return f.hycal(E[q], T[ids], E[tr], y[tr], ids)
    M['HyCal'] = (hy_fit, hy_sess, lambda: nbytes(E[np.array(st7['tr'])]))

    st8 = {}
    def bimc_fit(): st8['P'] = f.means(E[fit], y[fit], base); st8['Pb'] = st8['P'].clone()
    def bimc_sess(new, ids, q):
        Pn = f.means(E[new], y[new], np.unique(y[new]))
        Pn = f.norm(.9 * Pn + .1 * f.norm(torch.softmax(16 * (Pn @ st8['Pb'].T), 1) @ st8['Pb']))
        st8['P'] = torch.cat([st8['P'], Pn]); fp = f.norm(.3 * T[ids] + .7 * st8['P'])
        return torch.softmax(E[q] @ fp.T, dim=1)
    M['BiMC'] = (bimc_fit, bimc_sess, lambda: nbytes(st8['P']))
    return M


@torch.inference_mode()
def measure(ds, enc):
    B = build(ds, enc); res = {}
    for name, (fit_, sess, state) in methods(B).items():
        runs = []; fits = []; peak = 0
        for rep in range(REPEAT + 1):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
            sync(); t0 = time.perf_counter(); fit_(); sync(); tf = time.perf_counter() - t0; per = []
            for si, new in enumerate(B['sup'], start=1):
                ids = np.arange(B['nb'] + B['way'] * si); q = B['test'][B['y'][B['test']] < len(ids)]
                sync(); t0 = time.perf_counter(); s = sess(new, ids, q); sync(); per.append(time.perf_counter() - t0)
                assert s.shape == (len(q), len(ids)) and bool(torch.isfinite(s).all())
                del s
            if rep:  # first pass is warm-up
                runs.append(float(np.mean(per))); fits.append(tf); peak = max(peak, torch.cuda.max_memory_allocated() - m0)
        res[name] = dict(session_ms=1000 * float(np.median(runs)), base_fit_s=float(np.median(fits)),
                         peak_gpu_mb=peak / 2 ** 20, state_mb=state() / 2 ** 20)
        print('EFFICIENCY %-8s %-6s %-10s session %9.1f ms  base fit %7.2f s  peak GPU %8.1f MB  state %8.1f MB' %
              (ds, enc, name, res[name]['session_ms'], res[name]['base_fit_s'], res[name]['peak_gpu_mb'], res[name]['state_mb']), flush=True)
    return res


def main():
    p = argparse.ArgumentParser(); p.add_argument('--datasets', nargs='+', default=['mini', 'cub200']); p.add_argument('--encoder', default='b16')
    a = p.parse_args(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    out = {'gpu': torch.cuda.get_device_name(0), 'note': 'test queries of all seen classes scored every session; FeCAM runs on 4 CPU threads as in the benchmark'}
    for ds in a.datasets:
        out[f'{ds}/{a.encoder}'] = measure(ds, a.encoder)
    (f.D / 'efficiency.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
