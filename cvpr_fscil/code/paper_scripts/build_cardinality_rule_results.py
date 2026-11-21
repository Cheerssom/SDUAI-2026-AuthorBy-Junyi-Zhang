"""Report both specified cardinality interpretations without test-driven choice."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_softmax_readout_results import estimate
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=ROOT/'experiments/expansion_cardinality_rule_v2'
METHODS=['primary','catalog_frozen','query','none','fixed','dense']
NAMES={'primary':'Stage rule','catalog_frozen':'Catalog rule','query':'Query','none':'None','fixed':'Fixed','dense':'Dense/Query'}

def main():
    verified=json.loads((OUT/'verification.json').read_text());assert verified['complete']
    conditions=[];comparisons=[]
    for enc in ['l14','b32']:
        for ds in ['pets','cifar','imagenetr']:
            rows=[]
            for p in (OUT/enc/ds).glob('p*/o*_t*.json'):rows+=json.loads(p.read_text())['records']
            index={(r['method'],r['part'],r['order'],r['session'],r['policy'],r.get('retention')):r for r in rows}
            def get(m,p,o,s,policy):return index[m,p,o,s,policy,.95 if policy=='matched_test' else None]
            for method in METHODS:
                final=[get(method,p,0,5,'paired_accuracy') for p in range(10)]
                eligible=[p for p in range(10) if all(get(method,p,o,5,'matched_test')['feasible'] for o in [0,1])]
                conditions.append(dict(encoder=enc,dataset=ds,method=method,
                    aia=estimate([np.mean([get(method,p,o,s,'fixed_gate')['overall'] for o in [0,1] for s in range(1,6)]) for p in range(10)]),
                    capacity=estimate([np.mean([get(method,p,o,5,'matched_test')['new'] for o in [0,1]]) for p in eligible]),feasible_parts=eligible,
                    novel=estimate([r['new'] for r in final]),base=estimate([r['old'] for r in final]),
                    conditional_novel=estimate([r['new'] for r in final if not r['fallback']]),finite=sum(not r['fallback'] for r in final),test_met=sum(r['test_target_met'] for r in final)))
            for method,comparator in [(m,c) for m in ['primary','catalog_frozen'] for c in ['query','none','fixed','dense']]+[('primary','catalog_frozen')]:
                eligible=[p for p in range(10) if all(get(m,p,o,5,'matched_test')['feasible'] for m in [method,comparator] for o in [0,1])]
                comparisons.append(dict(encoder=enc,dataset=ds,method=method,comparator=comparator,joint_parts=eligible,
                    aia_gain=estimate([np.mean([get(method,p,o,s,'fixed_gate')['overall']-get(comparator,p,o,s,'fixed_gate')['overall'] for o in [0,1] for s in range(1,6)]) for p in range(10)]),
                    capacity_gain=estimate([np.mean([get(method,p,o,5,'matched_test')['new']-get(comparator,p,o,5,'matched_test')['new'] for o in [0,1]]) for p in eligible]),
                    policy_gain=estimate([np.mean([get(method,p,o,5,'paired_accuracy')['new']-get(comparator,p,o,5,'paired_accuracy')['new'] for o in [0,1]]) for p in range(10)])))
    dest=H/'cardinality_rule';dest.mkdir(exist_ok=True)
    result=dict(verification=verified,feature_witness=json.loads((OUT/'feature_witness.json').read_text()),
                restart_verification=json.loads((OUT/'restart_verification.json').read_text()),
                design_constant=30,conditions=conditions,comparisons=comparisons,
                scope='Post-development fixed rule; primary recomputes eta at each fitting stage, secondary freezes catalog eta throughout tuning/deployment. Same original reference and cap40 calibration.')
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    report=['# Cardinality rule validation','',result['scope'],'',
            '| Encoder | Dataset | Scheme | AIA | Capacity (n) | Base | Novel (conditional) | Finite/Met |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for r in conditions:
        c=r['capacity'];cond=r['conditional_novel']['mean'];cs='--' if cond is None else f'{cond:.2f}'
        cap='--' if c['mean'] is None else f"{c['mean']:.2f} ({c['n']})"
        report.append(f"| {r['encoder']} | {r['dataset']} | {NAMES[r['method']]} | {r['aia']['mean']:.2f} | {cap} | {r['base']['mean']:.2f} | {r['novel']['mean']:.2f} ({cs}) | {r['finite']}/{r['test_met']} |")
    report += ['','Policy table: first order; AIA/capacity and paired contrasts: both orders averaged within partitions.','']
    for r in comparisons:
        if r['dataset']!='pets' or r['method']!='primary':continue
        parts=[]
        for k in ['capacity_gain','policy_gain']:
            v=r[k];parts.append(k+f": {v['mean']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}], n={v['n']}")
        report.append(f"{r['encoder']} Pets Stage-minus-{NAMES[r['comparator']]}: "+'; '.join(parts))
    (dest/'report.md').write_text('\n'.join(report)+'\n')
    for ds in ['pets','cifar','imagenetr']:
        lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Encoder & Scheme & AIA & Capacity ($n$) & Base & Novel (cond.) & Finite/Met \\',r'\midrule']
        for enc in ['l14','b32']:
            for r in conditions:
                if (r['encoder'],r['dataset'])!=(enc,ds):continue
                c=r['capacity'];cv='--' if c['mean'] is None else f"{c['mean']:.2f}";n=r['conditional_novel']['mean'];ns='--' if n is None else f'{n:.2f}'
                lines.append(f"{enc.replace('l14','L/14').replace('b32','B/32')} & {NAMES[r['method']]} & {r['aia']['mean']:.2f} & {cv} ({c['n']}) & {r['base']['mean']:.2f} & {r['novel']['mean']:.2f} ({ns}) & {r['finite']}/{r['test_met']}"+r' \\')
            lines.append(r'\midrule')
        lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}')
        (H/f'table_cardinality_{ds}.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',r'Encoder & Weight rule & Capacity & Novel & Finite \\',r'\midrule']
    for enc in ['l14','b32']:
        for method,name in [('primary','Stage'),('catalog_frozen','Catalog'),('query','Query')]:
            r=next(r for r in conditions if (r['encoder'],r['dataset'],r['method'])==(enc,'pets',method))
            lines.append(f"{enc.replace('l14','L/14').replace('b32','B/32')} & {name} & {r['capacity']['mean']:.2f} & {r['novel']['mean']:.2f} & {r['finite']}/10"+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}');(H/'table_cardinality_main.tex').write_text('\n'.join(lines)+'\n')
    print('\n'.join(report))
if __name__=='__main__':main()
