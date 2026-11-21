"""Two-layer MLP external-head experiment derived from on External Session-0 Logistic Probe (A_ref = 87.78%):
ALL methods retain the EXACT SAME external logistic probe on base classes (so G = 0.00% for EVERY method!),
and differ ONLY in how novel-class evidence is scored and aligned against the external probe!
"""
from pathlib import Path
import json, sys
import numpy as np
import torch

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH.parent / 'revision_g_experiments_20261117'))
import probe_all_g as pa
sys.path.insert(0, str(pa.ROOT / "autodl_fscil"))
import extra_baselines as xb

def boot_ci(diffs, n_boot=20000, seed=2026):
    diffs = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi), int(np.sum(diffs > 0)), int(np.sum(diffs < 0))


def fit_mlp(E, y, fit, classes, tune):
    # Fixed architecture and training schedule, regularization selected on base tuning only.
    best = (-1.0, None, None)
    labels = torch.tensor(np.searchsorted(classes, y[fit]), device=E.device)
    weights = 1 / torch.bincount(labels, minlength=len(classes)).float().clamp_min(1)
    weights *= len(classes) / weights.sum()
    for decay in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        torch.manual_seed(2026)
        model = torch.nn.Sequential(
            torch.nn.Linear(E.shape[1], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(classes)),
        ).to(E.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=decay)
        for _ in range(300):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(E[fit].float()), labels, weight=weights)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = pa.macro(pa.prediction(model(E[tune].float()), classes) == y[tune], y[tune], classes)
        if score > best[0]:
            best = (score, model, decay)
    best[1].requires_grad_(False)
    return best[1], best[2]

@torch.no_grad()
def run():
    variants = ['AnchorCE_full', 'Wrap_fixed_mom']
    records = {cap: {m: [] for m in variants} for cap in [5]}

    torch.set_num_threads(4)
    for ds, (NB, WAY, S, NTUNE, NCAL, NC) in pa.CFG.items():
        for seed in range(3):
            print(f'START {ENC} {ds} seed={seed}', flush=True)
            torch.manual_seed(2026 + seed)
            z, E, T, y, iid, sessions, index, base, fit, tune, cal, selectionfit, actualfit, test, _ = pa.load_split(ds, ENC, 'certified', seed)
            bt = test[y[test] < NB]
            with torch.enable_grad():
                deployed, decay = fit_mlp(E, y, actualfit, base, tune)
                pb = np.arange(NB // 2)
                pn = np.arange(NB // 2, NB)
                pfit = selectionfit[np.isin(y[selectionfit], pb)]
                pseudo, pseudo_decay = fit_mlp(E, y, pfit, pb, tune[np.isin(y[tune], pb)])
            ref_cal = (pa.prediction(deployed(E[cal].float()), base) == y[cal])
            ref_closed = pa.macro(pa.prediction(deployed(E[bt].float()), base) == y[bt], y[bt], base)

            torch.manual_seed(2026 + seed)
            rng2 = np.random.default_rng(28012026 + seed)
            for k in base: _ = rng2.permutation(np.array([index[v] for v in sessions['1']])[y[np.array([index[v] for v in sessions['1']])] == k])
            ps = np.concatenate([rng2.choice(selectionfit[y[selectionfit] == k], 5, replace=False) for k in pn])
            Ppb = pa.means(E[pfit], y[pfit], pb); Ppn = pa.means(E[ps], y[ps], pn)
            pids = np.r_[pb, pn]
            mp, Pp = pa.precision(E[pfit], y[pfit], pb)

            rawPr_p = pseudo(E[tune].float())
            rawT_p = E[tune] @ T[pids].T
            rawP_p = E[tune] @ torch.cat([Ppb, Ppn]).T
            rawG_p = pa.lda(E[tune], torch.cat([mp, pa.raw_means(E[ps], y[ps], pn)]), Pp)

            def norm_b(v, nb, mode, fm=None):
                if mode == 'raw': return v
                if mode == 'fixed': return (v - fm[0]) / fm[1]
                ref = v[:, :nb]; sd = ref.std(1, correction=0, keepdim=True).clamp_min(1e-6)
                return v / sd if mode == 'none' else (v - ref.mean(1, keepdim=True)) / sd

            fm_p = [(float(v[:, :len(pb)].mean()), float(v[:, :len(pb)].std(1, correction=0).mean().clamp_min(1e-6)))
                    for v in [rawPr_p, rawT_p, rawP_p, rawG_p]]

            def sel3636(mode):
                cPr = norm_b(rawPr_p, len(pb), mode, fm_p[0])
                cT = norm_b(rawT_p, len(pb), mode, fm_p[1])
                cP = norm_b(rawP_p, len(pb), mode, fm_p[2])
                cG = norm_b(rawG_p, len(pb), mode, fm_p[3])
                best = (-1, None)
                for wt in [0.5, 1.0, 2.0]:
                    for u in [0., 0.5, 1.0, 2.0]:
                        for g in [0., 0.5, 1.0]:
                            s = torch.zeros(len(tune), len(pids), device='cuda')
                            s[:, :len(pb)] = cPr
                            s[:, len(pb):] = wt * cT[:, len(pb):] + u * cP[:, len(pb):] + g * cG[:, len(pb):]
                            cc = pa.comp(s, pids, y[tune], pb, pn)
                            d = np.quantile(cc['margin'], np.linspace(0, 1, 101))
                            ch = (d[:, None] > cc['margin'][None]) | ((d[:, None] == cc['margin'][None]) & (cc['novel_prediction'][None] < cc['base_prediction'][None]))
                            ab = (~ch) @ (cc['weights'] * cc['base_correct']); an = ch @ (cc['weights'] * cc['novel_correct'])
                            h = np.divide(2 * ab * an, ab + an, out=np.zeros_like(ab), where=ab + an > 0); j = int(h.argmax())
                            if h[j] > best[0] + 1e-12: best = (h[j], dict(wt=wt, u=u, g=g))
                return best[1]

            pars = {m: sel3636(md) for m, md in [('AnchorCE_full', 'full'), ('Wrap_fixed_mom', 'fixed')]}

            # Final session
            trainidx = list(actualfit)
            for si in range(1, S): trainidx.extend(index[v] for v in sessions[str(si + 1)])
            n = np.arange(NB, NB + WAY * (S - 1)); ids = np.r_[base, n]
            active = test; queries = np.r_[tune, cal, active]; Q = E[queries]; tr = np.array(trainidx)
            N_t, C_c = len(tune), len(cal)
            Pb = pa.means(E[actualfit], y[actualfit], base); Pn = pa.means(E[tr], y[tr], n)
            P_all_p = pa.means(E[tr], y[tr], ids)
            mb, P_frozen = xb.precision(E[actualfit], y[actualfit], base)
            mus_frozen = torch.cat([mb, pa.raw_means(E[tr], y[tr], n)])
            mu_all, P_all = xb.precision(E[tr], y[tr], ids)

            rawPr = deployed(Q.float())
            rawT = Q @ T[ids].T
            rawP = Q @ torch.cat([Pb, Pn]).T
            rawG = xb.lda(Q, mus_frozen, P_frozen)
            fm_a = [(float(v[:N_t, :NB].mean()), float(v[:N_t, :NB].std(1, correction=0).mean().clamp_min(1e-6)))
                    for v in [rawPr, rawT, rawP, rawG]]

            mscores = {}
            for m, md in [('AnchorCE_full', 'full'), ('Wrap_fixed_mom', 'fixed')]:
                cPr = norm_b(rawPr, NB, md, fm_a[0])
                cT = norm_b(rawT, NB, md, fm_a[1])
                cP = norm_b(rawP, NB, md, fm_a[2])
                cG = norm_b(rawG, NB, md, fm_a[3])
                s = torch.zeros(len(queries), len(ids), device='cuda')
                s[:, :NB] = cPr
                assert np.array_equal(pa.prediction(cPr, base), pa.prediction(rawPr, base)), 'Closed-head decisions changed'
                p = pars[m]
                s[:, NB:] = p['wt'] * cT[:, NB:] + p['u'] * cP[:, NB:] + p['g'] * cG[:, NB:]
                mscores[m] = s

            for m, s in mscores.items():
                vc = pa.comp(s[:N_t], ids, y[tune], base, n)
                tc = pa.comp(s[N_t + C_c:], ids, y[active], base, n)
                cand = np.unique(np.quantile(vc['margin'], np.linspace(0, 1, 101)))
                scal = s[N_t:N_t + C_c]
                cal_base_corr = (pa.prediction(scal[:, :NB], base) == y[cal])
                G_cal = 100.0 * float(pa.risk.weights(y[cal], base) @ (~ref_cal & cal_base_corr))
                assert G_cal == 0.0, f"{m} had G_cal={G_cal}"
                for req_cap in [5]:
                    cap = min(req_cap, NCAL)
                    sub = np.sort(np.concatenate([np.flatnonzero(y[cal] == k)[:cap] for k in base]))
                    cc = pa.comp(scal[sub], ids, y[cal][sub], base, n)
                    w = pa.risk.weights(y[cal][sub], base)
                    det = pa.paired.certify(cc, ref_cal[sub], w, cand, ratio=.95, sessions=S - 1)
                    d = det['offset']
                    ev = dict(base=ref_closed, novel=0.0) if d is None else pa.evaluate(tc, d)
                    records[req_cap][m].append(dict(ds=ds, seed=seed, novel=100.0 * ev['novel'],
                                                    base=100.0 * ev['base'], admitted=(d is not None),
                                                    reference_base=100.0 * ref_closed,
                                                    empirical_floor_met=bool(ev['base'] >= .95 * ref_closed - 1e-12),
                                                    decay=decay, pseudo_decay=pseudo_decay, alignment_parameters=pars[m],
                                                    recovery_percent=G_cal))

            output = dict(encoder=ENC, source_sha256=SOURCE_HASH, records=records)
            (SCRATCH / f'{ENC}_records.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
            print(f'DONE {ENC} {ds} seed={seed}', flush=True)

    for cap in [5]:
        print(f"\n=== ALL G=0% SAME EXTERNAL PROBE (cap={cap}) ===")
        ace = np.array([r['novel'] for r in records[cap]['AnchorCE_full']])
        for m in variants:
            rs = records[cap][m]
            vals = np.array([r['novel'] for r in rs])
            adm = sum(r['admitted'] for r in rs)
            cond = np.mean([r['novel'] for r in rs if r['admitted']]) if adm else 0.0
            ci = boot_ci(ace - vals) if m != 'AnchorCE_full' else None
            per_ds = {ds: round(float(np.mean([r['novel'] for r in rs if r['ds'] == ds])), 2) for ds in pa.CFG}
            print(f"  {m:20s}: Cert={vals.mean():5.2f}% | Cond={cond:5.2f}% | Adm={adm}/9 | per_ds={per_ds} | diff={ci}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', choices=['b16'], required=True)
    args = parser.parse_args()
    ENC = args.encoder
    SOURCE_HASH = '338988be0e52ad2846f6f75bd9069cec1cfae8fba03b482e21c32ccdd3cf189b'
    run()

