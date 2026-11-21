"""Summarise the --sweep runs: final-session certified new-class accuracy (common contract) for every
readout, retention ratio r and per-class calibration cap, averaged over seeds (and over datasets)."""
from pathlib import Path
import json
import numpy as np
import sys
D = Path(__file__).resolve().parent / 'work'; S = {'cifar100': 9, 'mini': 9, 'cub200': 11}
REF4 = len(sys.argv) > 1 and sys.argv[1] == '_ref4'
ROOTS = ['readouts_ref4', 'readouts_extra_ref4', 'readouts_anchorg_ref4', 'readouts_plugin'] if REF4 else ['readouts_sweep', 'readouts_extra_sweep', 'readouts_anchorg_sweep']
out = {}
for ds, last in S.items():
    for enc_dir in sorted(e for root in ROOTS for e in (D / root / ds).glob('*')):
        for sd in sorted((enc_dir / 'certified').glob('seed*')):
            if not (sd / 'complete.json').exists():
                continue
            for r in json.loads((sd / f'session{last - 1:02d}.json').read_text())['rows']:
                for key, v in r.get('sweep', {}).items():
                    out.setdefault((enc_dir.name, r['method'], key), {}).setdefault(ds, []).append(v)
res = {}
for (enc, m, key), per in sorted(out.items()):
    row = {ds: dict(novel=100 * np.mean([v['novel'] for v in vs]), admitted=sum(v['admitted'] for v in vs),
                    met=sum(v['met'] for v in vs), n=len(vs)) for ds, vs in per.items()}
    if len(row) == 3:
        row['average_novel'] = float(np.mean([row[ds]['novel'] for ds in S]))
    res[f'{enc}/{m}/{key}'] = row
    print('%-48s' % f'{enc}/{m}/{key}', ' '.join(f"{ds}={row[ds]['novel']:.2f}({row[ds]['admitted']}/{row[ds]['n']},met{row[ds]['met']})" for ds in S if ds in row),
          'avg=%.2f' % row['average_novel'] if 'average_novel' in row else '')
(D / ('summary_sweep_ref4.json' if REF4 else 'summary_sweep.json')).write_text(json.dumps(res, indent=2))
