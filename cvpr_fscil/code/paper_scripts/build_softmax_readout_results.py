"""Summarize the complete softmax-head control; retain all predefined contrasts."""
from pathlib import Path
import json,csv
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OUT=ROOT/'experiments/expansion_softmax_readout'

def estimate(values,scale=100):
    a=np.asarray(values,float);rng=np.random.default_rng(27102026)
    if not len(a):return dict(n=0,mean=None,sd=None,ci95=None,values=[])
    boot=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)*scale
    return dict(n=len(a),mean=float(a.mean()*scale),sd=float(a.std(ddof=1)*scale) if len(a)>1 else 0.,
                ci95=np.quantile(boot,[.025,.975]).tolist(),values=(a*scale).tolist())

def main():
    verification=json.loads((OUT/'verification.json').read_text());assert verification['complete']
    rows=[]
    for e in ['l14','b32']:
        for part in range(10):
            for order in [0,1]:
                for session in range(1,6):
                    p=OUT/e/f'imagenetr/p{part}/o{order}_t{session}.json'
                    rows+=json.loads(p.read_text())['records']
    summary=[];comparisons=[]
    for e in ['l14','b32']:
        for mode in ['query','none','fixed']:
            rr=[r for r in rows if r['encoder']==e and r['method']=='softmax_'+mode]
            pick=lambda p,o,s,pol:next(r for r in rr if (r['part'],r['order'],r['session'],r['policy'])==(p,o,s,pol) and (pol!='matched_test' or r['retention']==.95))
            aia=estimate([np.mean([pick(p,o,s,'fixed_gate')['overall'] for o in [0,1] for s in range(1,6)]) for p in range(10)])
            final=[pick(p,0,5,'paired_accuracy') for p in range(10)]
            cap=[pick(p,o,5,'matched_test') for p in range(10) for o in [0,1]]
            feasible=[p for p in range(10) if all(pick(p,o,5,'matched_test')['feasible'] for o in [0,1])]
            pairedcap=estimate([np.mean([pick(p,o,5,'matched_test')['new'] for o in [0,1]]) for p in feasible])
            pairs_policy=estimate([np.mean([pick(p,o,5,'paired_accuracy')['new'] for o in [0,1]]) for p in range(10)])
            headroom=estimate([np.mean([pick(p,o,5,'matched_test')['new']-pick(p,o,5,'paired_accuracy')['new'] for o in [0,1]]) for p in feasible])
            summary.append(dict(encoder=e,mode=mode,aia=aia,novel=estimate([r['new'] for r in final]),
                                base=estimate([r['old'] for r in final]),finite=sum(not r['fallback'] for r in final),
                                test_met=sum(r['test_target_met'] for r in final),both_order_policy=pairs_policy,
                                capacity=pairedcap,capacity_feasible_parts=feasible,headroom=headroom,
                                conditional_novel=estimate([r['new'] for r in final if not r['fallback']])))
        for other in ['none','fixed']:
            def get(mode,p,o,s,policy):
                return next(r for r in rows if (r['encoder'],r['method'],r['part'],r['order'],r['session'],r['policy'])==(e,'softmax_'+mode,p,o,s,policy) and (policy!='matched_test' or r['retention']==.95))
            eligible=[p for p in range(10) if all(get(m,p,o,5,'matched_test')['feasible'] for m in ['query',other] for o in [0,1])]
            comparisons.append(dict(encoder=e,comparator=other,
                aia_gain=estimate([np.mean([get('query',p,o,s,'fixed_gate')['overall']-get(other,p,o,s,'fixed_gate')['overall'] for o in [0,1] for s in range(1,6)]) for p in range(10)]),
                final_direct_gain=estimate([np.mean([get('query',p,o,5,'paired_accuracy')['new']-get(other,p,o,5,'paired_accuracy')['new'] for o in [0,1]]) for p in range(10)]),
                capacity_gain=estimate([np.mean([get('query',p,o,5,'matched_test')['new']-get(other,p,o,5,'matched_test')['new'] for o in [0,1]]) for p in eligible]),
                capacity_joint_parts=eligible))
    result=dict(scope='New softmax readout control on ImageNet-R; old CLIP features, splits/supports/calibration; original common ridge accuracy reference.',
                verification={k:v for k,v in verification.items() if k!='source_reference_sha256'},conditions=summary,comparisons=comparisons)
    dest=H/'softmax_readout';dest.mkdir(exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Cross-entropy readout control','',result['scope'],'',
           '| Encoder | Normalization | Average incremental | Final novel | Finite | Test met | Common-floor capacity | Headroom |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda x:'--' if x['mean'] is None else f"{x['mean']:.2f}"
        lines.append(f"| {r['encoder']} | {r['mode']} | {fmt(r['aia'])} | {fmt(r['novel'])} | {r['finite']}/10 | {r['test_met']}/10 | {fmt(r['capacity'])} (n={r['capacity']['n']}) | {fmt(r['headroom'])} |")
    lines+=['','Final novel/admission columns use first order; capacity/headroom and paired contrasts average two orders within each partition. All percentages include fallback. Intervals characterize class-assignment variation on fixed images. Cross-head recovery is allowed relative to the original ridge reference.']
    for c in comparisons:
        lines += ['',f"{c['encoder']} query-minus-{c['comparator']}: "+'; '.join(f"{name}: {c[key]['mean']:.2f} [{c[key]['ci95'][0]:.2f}, {c[key]['ci95'][1]:.2f}], n={c[key]['n']}" if c[key]['mean'] is not None else f'{name}: no jointly feasible partitions' for name,key in [('AIA','aia_gain'),('capacity','capacity_gain'),('direct policy','final_direct_gain')])]
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    table=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Encoder & Alignment & AIA (\%) & Capacity (\%) & Direct novel (\%) & Finite/Met \\',r'\midrule']
    for r in summary:
        cap='--' if r['capacity']['mean'] is None else f"{r['capacity']['mean']:.2f}"
        table.append(f"{r['encoder'].replace('l14','L/14').replace('b32','B/32')} & {r['mode'].capitalize()} & {r['aia']['mean']:.2f} & {cap} & {r['novel']['mean']:.2f} & {r['finite']}/{r['test_met']}"+r' \\')
    table += [r'\bottomrule',r'\end{tabular}']
    (H/'table_softmax_readout.tex').write_text('\n'.join(table)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
