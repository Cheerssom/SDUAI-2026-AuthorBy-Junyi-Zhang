"""Summarize every prespecified frozen RN50 readout comparison."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_softmax_readout_results import estimate
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=ROOT/'experiments/expansion_rn50'
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
    dest=H/'rn50';dest.mkdir(exist_ok=True)
    result=dict(encoder='OpenAI CLIP RN50',revision='afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762',verification=v,
                feature_verification=json.loads((OUT/'features/verification.json').read_text()),
                score_witness=json.loads((OUT/'model_witness_verification.json').read_text()),
                restart_verification=json.loads((OUT/'restart_verification.json').read_text()),
                conditions=conditions,comparisons=comparisons,
                scope='Convolutional ModifiedResNet backbone with native attention pooling; no ViT image backbone. Encoder-specific common reference, original source image identities and calibration protocol.')
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# RN50 convolutional-backbone validation','',result['scope'],'',
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
    lines += ['',f"Verification: {v['metric_records']} metric records, {v['prediction_archives']} prediction archives, {v['reconstructed_tests']} independently reconstructed threshold tests.",
              f"CPU score witness: {result['score_witness']['score_blocks']} blocks, {result['score_witness']['predictions_checked']} predictions, maximum margin error {result['score_witness']['max_margin_error']:.8g}.",
              f"Warm restart: {result['restart_verification']['unchanged_checkpoint_files']} model/feature/evaluation files retain hashes and modification times."]
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    table=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Base head & Alignment & AIA & Capacity ($n$) & Direct base & Direct novel & Finite/Met \\',r'\midrule']
    for r in conditions:
        cap='--' if r['capacity']['mean'] is None else f"{r['capacity']['mean']:.2f}"
        name={'retained_query':'Retained & Query','dense_query':'Dense & Query','retained_none':'Retained & None','dense_none':'Dense & None','retained_fixed':'Retained & Fixed','dense_fixed':'Dense & Fixed','hycal':'HyCal transfer & --'}[r['method']]
        table.append(f"{name} & {r['aia']['mean']:.2f} & {cap} ({r['capacity']['n']}) & {r['base']['mean']:.2f} & {r['novel']['mean']:.2f} & {r['finite']}/{r['test_met']}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}'];(H/'table_rn50.tex').write_text('\n'.join(table)+'\n')
    pair=[r'\begin{tabular}{@{}lrrrc@{}}',r'\toprule',r'Comparator & AIA gain & Capacity gain & Policy gain & $n$ \\',r'\midrule']
    names={'dense_query':'Dense/Query','retained_none':'Retained/None','dense_none':'Dense/None',
           'retained_fixed':'Retained/Fixed','dense_fixed':'Dense/Fixed','hycal':'HyCal transfer'}
    for r in comparisons:
        vals=[names[r['comparator']]]
        for key in ['aia_gain','capacity_gain','policy_gain']:
            v=r[key];vals.append('--' if v['mean'] is None else f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
        vals.append(str(len(r['joint_parts'])));pair.append(' & '.join(vals)+r' \\')
    pair += [r'\bottomrule',r'\end{tabular}'];(H/'table_rn50_pairs.tex').write_text('\n'.join(pair)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
