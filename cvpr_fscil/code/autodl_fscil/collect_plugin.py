"""Summarise plug-in anchoring (work/readouts_plugin): accuracy, recovery, certification (common = four-source
AnchorCE closed head, self = own head) and the calibration-budget sweep, mean over datasets and seeds."""
from pathlib import Path
import json
import numpy as np
D = Path(__file__).resolve().parent / 'work' / 'readouts_plugin'; S = {'cifar100': 9, 'mini': 9, 'cub200': 11}
M = ['HyCal_transfer', 'HyCal_anchor', 'GDA_transfer', 'GDA_anchor']
res = {}
for enc in sorted({p.name for p in D.glob('*/*') if p.is_dir()}):
    for m in M:
        acc, last, G, Dd, cc, cs, sw, sws = [], [], [], [], [], [], {}, {}
        for ds, n in S.items():
            for s in range(3):
                st = D / ds / enc / 'standard' / f'seed{s}'; ce = D / ds / enc / 'certified' / f'seed{s}'
                if (st / 'complete.json').exists():
                    rows = [r for t in range(n) for r in json.loads((st / f'session{t:02d}.json').read_text())['rows'] if r['method'] == m]
                    a = [100 * r['unconstrained_top1'] for r in sorted(rows, key=lambda r: r['session'])]
                    acc.append(np.mean(a)); last.append(a[-1])
                if (ce / 'complete.json').exists():
                    r = [r for r in json.loads((ce / f'session{n - 1:02d}.json').read_text())['rows'] if r['method'] == m][0]
                    G.append(100 * r['recovery']['G']); Dd.append(100 * r['recovery']['D'])
                    cc.append(100 * r['common']['novel']); cs.append(100 * r['self']['novel'])
                    for k, v in r['sweep'].items(): sw.setdefault(k, []).append(100 * v['novel'])
                    for k, v in r['sweep_self'].items(): sws.setdefault(k, []).append(100 * v['novel'])
        mean = lambda v: float(np.mean(v)) if v else None
        res[f'{enc}/{m}'] = dict(Avg=mean(acc), Last=mean(last), recovery_G=mean(G), destruction_D=mean(Dd), certified_common=mean(cc),
                                 certified_self=mean(cs), sweep_common={k: mean(v) for k, v in sw.items()}, sweep_self={k: mean(v) for k, v in sws.items()},
                                 n_standard=len(acc), n_certified=len(cc))
        r = res[f'{enc}/{m}']; f2 = lambda x: 'NA' if x is None else f'{x:.2f}'
        print(f"PLUGIN {enc}/{m:16s} Avg {f2(r['Avg'])} Last {f2(r['Last'])} G {f2(r['recovery_G'])} D {f2(r['destruction_D'])} "
              f"cert common {f2(r['certified_common'])} self {f2(r['certified_self'])} | r.95 cap5 common {f2(r['sweep_common'].get('0.95_5'))} "
              f"self {f2(r['sweep_self'].get('0.95_5'))} | cap10 common {f2(r['sweep_common'].get('0.95_10'))} self {f2(r['sweep_self'].get('0.95_10'))}")
(D.parent / 'summary_plugin.json').write_text(json.dumps(res, indent=2))
