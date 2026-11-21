"""Summarize every prespecified frozen SigLIP readout comparison."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_softmax_readout_results import estimate
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=ROOT/'experiments/expansion_cross_encoder'
METHODS=[a+'_'+b for b in ['query','none','fixed'] for a in ['retained','dense']]+['hycal']

def main():
    v=json.loads((OUT/'verification.json').read_text());assert v['complete']
    rows=[]
    for p in (OUT/'evaluation').glob('p*/o*_t*.json'):rows+=json.loads(p.read_text())['records']
    index={(r['method'],r['part'],r['order'],r['session'],r['policy'],r.get('retention')):r for r in rows}
    def get(m,p,o,s,policy):return index[m,p,o,s,policy,.95 if policy=='matched_test' else None]
    conditions=[];comparisons=[]
    for m in METHODS:
        final=[get(m,p,0,5,'paired_accuracy') for p in range(10)]
        eligible=[p for p in range(10) if all(get(m,p,o,5,'matched_test')['feasible'] for o in [0,1])]
        conditions.append(dict(method=m,aia=estimate([np.mean([get(m,p,o,s,'fixed_gate')['overall'] for o in [0,1] for s in range(1,6)]) for p in range(10)]),
                               capacity=estimate([np.mean([get(m,p,o,5,'matched_test')['new'] for o in [0,1]]) for p in eligible]),
                               capacity_feasible_parts=eligible,novel=estimate([r['new'] for r in final]),base=estimate([r['old'] for r in final]),
                               finite=sum(not r['fallback'] for r in final),test_met=sum(r['test_target_met'] for r in final),
                               conditional_novel=estimate([r['new'] for r in final if not r['fallback']])))
    for comparator in [m for m in METHODS if m!='retained_query']:
        eligible=[p for p in range(10) if all(get(m,p,o,5,'matched_test')['feasible'] for m in ['retained_query',comparator] for o in [0,1])]
        comparisons.append(dict(comparator=comparator,
            aia_gain=estimate([np.mean([get('retained_query',p,o,s,'fixed_gate')['overall']-get(comparator,p,o,s,'fixed_gate')['overall'] for o in [0,1] for s in range(1,6)]) for p in range(10)]),
            policy_gain=estimate([np.mean([get('retained_query',p,o,5,'paired_accuracy')['new']-get(comparator,p,o,5,'paired_accuracy')['new'] for o in [0,1]]) for p in range(10)]),
            capacity_gain=estimate([np.mean([get('retained_query',p,o,5,'matched_test')['new']-get(comparator,p,o,5,'matched_test')['new'] for o in [0,1]]) for p in eligible]),joint_parts=eligible))
    dest=H/'cross_encoder';dest.mkdir(exist_ok=True)
    result=dict(encoder='google/siglip-base-patch16-224',revision='7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed',verification=v,
                feature_verification=json.loads((OUT/'features/verification.json').read_text()),
                conditions=conditions,comparisons=comparisons,
                scope='Cross-pretraining model family; ViT backbone remains. Encoder-specific common reference, original source image identities and calibration protocol.')
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# SigLIP cross-pretraining validation','',result['scope'],'',
           '| Readout | AIA | Capacity (n) | Direct base | Direct novel | Finite/Met |',
           '|---|---:|---:|---:|---:|---:|']
    for r in conditions:
        cap='--' if r['capacity']['mean'] is None else f"{r['capacity']['mean']:.2f} ({r['capacity']['n']})"
        lines.append(f"| {r['method']} | {r['aia']['mean']:.2f} | {cap} | {r['base']['mean']:.2f} | {r['novel']['mean']:.2f} | {r['finite']}/{r['test_met']} |")
    lines += ['','AIA and capacity average both orders within partitions; direct-policy means and counts are final-session first-order. Fallback-inclusive means and conditional values are retained.','']
    for c in comparisons:
        parts=[]
        for key in ['aia_gain','capacity_gain','policy_gain']:
            r=c[key];parts.append(f"{key}: {r['mean']:.2f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}], n={r['n']}" if r['mean'] is not None else key+': no paired feasible set')
        lines.append('Retained query minus '+c['comparator']+': '+'; '.join(parts))
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    table=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Base head & Alignment & AIA & Capacity ($n$) & Direct base & Direct novel & Finite/Met \\',r'\midrule']
    for r in conditions:
        cap='--' if r['capacity']['mean'] is None else f"{r['capacity']['mean']:.2f}"
        name={'retained_query':'Retained & Query','dense_query':'Dense & Query','retained_none':'Retained & None','dense_none':'Dense & None','retained_fixed':'Retained & Fixed','dense_fixed':'Dense & Fixed','hycal':'HyCal transfer & --'}[r['method']]
        table.append(f"{name} & {r['aia']['mean']:.2f} & {cap} ({r['capacity']['n']}) & {r['base']['mean']:.2f} & {r['novel']['mean']:.2f} & {r['finite']}/{r['test_met']}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}'];(H/'table_cross_encoder.tex').write_text('\n'.join(table)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
