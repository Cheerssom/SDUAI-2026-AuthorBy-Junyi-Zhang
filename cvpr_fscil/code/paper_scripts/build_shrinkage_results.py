"""All prespecified shrinkage contrasts; no test-driven scheme choice."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_softmax_readout_results import estimate
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=ROOT/'experiments/expansion_shrinkage'
METHODS=['query','none','fixed','half','adaptive_eta','adaptive_scheme']

def main():
    verified=json.loads((OUT/'verification.json').read_text());assert verified['complete']
    summaries=[];comparisons=[]
    for enc in ['l14','b32']:
        for ds in ['pets','cifar','imagenetr']:
            allrows=[];choices={k:Counter() for k in ['adaptive_eta','adaptive_scheme']}
            for part in range(10):
                po=OUT/enc/ds/f'p{part}';selection=json.loads((po/'selection.json').read_text())
                for k in choices:choices[k][selection[k]]+=1
                for order in [0,1]:
                    for session in range(1,6):allrows+=json.loads((po/f'o{order}_t{session}.json').read_text())['records']
            index={(r['method'],r['part'],r['order'],r['session'],r['policy'],r.get('retention')):r for r in allrows}
            def get(m,p,o,s,policy):return index[m,p,o,s,policy,.95 if policy=='matched_test' else None]
            for m in METHODS:
                final=[get(m,p,0,5,'paired_accuracy') for p in range(10)]
                eligible=[p for p in range(10) if all(get(m,p,o,5,'matched_test')['feasible'] for o in [0,1])]
                summaries.append(dict(encoder=enc,dataset=ds,method=m,
                    aia=estimate([np.mean([get(m,p,o,s,'fixed_gate')['overall'] for o in [0,1] for s in range(1,6)]) for p in range(10)]),
                    novel=estimate([r['new'] for r in final]),base=estimate([r['old'] for r in final]),
                    conditional_novel=estimate([r['new'] for r in final if not r['fallback']]),
                    finite=sum(not r['fallback'] for r in final),test_met=sum(r['test_target_met'] for r in final),
                    capacity=estimate([np.mean([get(m,p,o,5,'matched_test')['new'] for o in [0,1]]) for p in eligible]),
                    feasible_parts=eligible,chosen_schemes=dict(choices[m]) if m in choices else None))
            for adaptive in ['half','adaptive_eta','adaptive_scheme']:
                for comparator in ['query','none','fixed']:
                    eligible=[p for p in range(10) if all(get(m,p,o,5,'matched_test')['feasible'] for m in [adaptive,comparator] for o in [0,1])]
                    comparisons.append(dict(encoder=enc,dataset=ds,method=adaptive,comparator=comparator,
                        aia_gain=estimate([np.mean([get(adaptive,p,o,s,'fixed_gate')['overall']-get(comparator,p,o,s,'fixed_gate')['overall'] for o in [0,1] for s in range(1,6)]) for p in range(10)]),
                        policy_gain=estimate([np.mean([get(adaptive,p,o,5,'paired_accuracy')['new']-get(comparator,p,o,5,'paired_accuracy')['new'] for o in [0,1]]) for p in range(10)]),
                        capacity_gain=estimate([np.mean([get(adaptive,p,o,5,'matched_test')['new']-get(comparator,p,o,5,'matched_test')['new'] for o in [0,1]]) for p in eligible]),joint_parts=eligible))
    dest=H/'shrinkage';dest.mkdir(exist_ok=True)
    result=dict(verification=verified,conditions=summaries,comparisons=comparisons,
                budgets=dict(per_fixed_scheme=3636,adaptive_eta=10908,adaptive_scheme=14544),
                scope='Prospective locked post-development extension on previously examined benchmarks; adaptive selection uses base tuning only.')
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Reference-moment shrinkage','',result['scope'],'',
           '| Encoder | Dataset | Scheme | AIA | Capacity (n) | Direct novel | Finite/Met | Selected |',
           '|---|---|---|---:|---:|---:|---:|---|']
    for r in summaries:
        c='--' if r['capacity']['mean'] is None else f"{r['capacity']['mean']:.2f} ({r['capacity']['n']})"
        lines.append(f"| {r['encoder']} | {r['dataset']} | {r['method']} | {r['aia']['mean']:.2f} | {c} | {r['novel']['mean']:.2f} | {r['finite']}/{r['test_met']} | {r['chosen_schemes'] or ''} |")
    lines += ['','Policy means use first order; AIA/capacity and paired contrasts average orders within each partition. Adaptive_eta uses10908 settings; adaptive_scheme14544; each fixed scheme3636. No equal-budget superiority claim is made.','']
    for c in comparisons:
        if c['dataset']!='pets' or c['method']=='half':continue
        vals=[]
        for key in ['capacity_gain','policy_gain']:
            v=c[key];vals.append(f"{key} {v['mean']:.2f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}] n={v['n']}" if v['mean'] is not None else key+' unavailable')
        lines.append(f"{c['encoder']} Pets {c['method']}-minus-{c['comparator']}: "+'; '.join(vals))
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    diag=ROOT/'experiments/expansion_reference_cardinality';assert (diag/'complete.json').exists()
    rows=[]
    for p in diag.glob('*/*/p*.json'):rows+=json.loads(p.read_text())['records']
    groups={}
    for r in rows:
        if r['bank']!='joint':continue
        key=(r['encoder'],r['dataset'],r['stage'],r['subset_count'],r['reference_count'])
        groups.setdefault(key,[]).append(r)
    statistics=[]
    for key,rr in groups.items():
        statistics.append(dict(zip(['encoder','dataset','stage','subset_count','full_count'],key))|{k:float(np.mean([r[k] for r in rr])) for k in ['normalized_mean_rms','relative_scale_rms','margin_rms']}|{'cells':len(rr)})
    full=[r for r in rows if r['reference_count']==r['subset_count']]
    assert max(r['margin_rms'] for r in full)<1e-4
    (dest/'cardinality_diagnostic.json').write_text(json.dumps(dict(records=len(rows),conditions=statistics,
        scope='Reference-subset perturbations of a fixed model on tuning queries. No iid variance law or causal attribution of dataset differences.'),indent=2)+'\n')
if __name__=='__main__':main()
