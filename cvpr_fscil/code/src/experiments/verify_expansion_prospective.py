"""Independent prediction-archive reconstruction for the four requested blocks."""
from pathlib import Path
import argparse,json,sys,math,time,subprocess
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
from kbs_runtime import digest_file,atomic_json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_prospective_validation'

def compact(z,m,pool,y,base,new):
    key=pool+'_'+m+'_';margin=z[key+'margin'];bp=z[key+'base_prediction'];npred=z[key+'novel_prediction'];w=np.zeros(len(y))
    for group in [base,new]:
        for cl in group:
            q=y==cl
            if q.any():w[q]=1/(len(group)*q.sum())
    return dict(margin=margin,base_prediction=bp,novel_prediction=npred,base_correct=(bp==y)&np.isin(y,base),novel_correct=(npred==y)&np.isin(y,new),weights=w,nb=len(base),nn=len(new))

def accuracy(c,d):
    ch=(d>c['margin'])|((d==c['margin'])&(c['novel_prediction']<c['base_prediction']))
    return float(c['weights'][c['base_correct']&~ch].sum()),float(c['weights'][c['novel_correct']&ch].sum())

def checkpolicy(c,ref,w,candidates,saved,tc,closed):
    got=paired.certify(c,ref,w,candidates);assert got['offset']==saved['offset'] and got['fallback']==saved['fallback']
    assert len(got['trace'])==len(saved['trace'])
    for a,b in zip(got['trace'],saved['trace']):
        assert abs(a['pvalue']-b['pvalue'])<1e-11 and a['accepted']==b['accepted']
        # Independently recompute the literal per-observation product,without
        # the runner's equal-weight grouping or its p-value helper.
        offset=b['offset'];novel=(offset>c['margin'])|((offset==c['margin'])&(c['novel_prediction']<c['base_prediction']))
        correct=c['base_correct']&~novel;surplus=(correct.astype(float)-.95*ref)*(w/w.max())
        logs=np.log1p(paired.LAMBDAS[:,None]*surplus[None,:]).sum(axis=1)
        log_e=float(np.logaddexp.reduce(logs)-math.log(len(paired.LAMBDAS)))
        raw_p=math.exp(-max(0.,log_e));assert abs(raw_p-b['pvalue'])<1e-10
    if got['fallback']:ab,an=closed,0.
    else:ab,an=accuracy(tc,got['offset'])
    assert abs(ab-saved['old'])<1e-10 and abs(an-saved['new'])<1e-10
    return len(got['trace'])

def main():
    a=argparse.ArgumentParser();a.add_argument('--partial',action='store_true');a.add_argument('--block',choices=['external','mechanism','cardinality']);a=a.parse_args();start=time.monotonic()
    assert a.partial or a.block is None,'A block filter requires --partial'
    mf=json.loads((OUT/'manifest.json').read_text())
    for p,h in mf['sources'].items():assert digest_file(ROOT/p)==h,p
    counts=dict(archives=0,records=0,threshold_tests=0,frontiers=0,selections=0,shift_records=0)
    for block in ([a.block] if a.block else ['mechanism','cardinality','external']):
        folders=sorted((OUT/block).glob('*/*/*/p*'))
        for folder in folders:
            jp=folder/'selection.json'
            if not jp.exists():continue
            sel=json.loads(jp.read_text());assert digest_file(folder/'weights.pt')==sel['weights_sha256'];counts['selections']+=1
            for v in sel['selections'].values():
                assert v['evaluated_settings']==3636 and len(v['trials'])==36
                maximum=max(max(t['harmonics']) for t in v['trials']);assert abs(maximum-v['selected']['pseudo_harmonic'])<1e-10
            if block=='external':
                for state in sel['gates'].values():assert state['selection_settings']==3636 and state['candidate_models']==len(state['trials'])==36
            for p in sorted(folder.glob('o*_t*.json')):
                v=json.loads(p.read_text());assert v['selection_sha256']==digest_file(jp);assert v['archive_sha256']==digest_file(p.with_suffix('.npz'))
                with np.load(p.with_suffix('.npz')) as data:z={k:data[k] for k in data.files}
                b,n=z['base_classes'],z['novel_classes'];yc,yt=z['cal_labels'],z['test_labels'];w=risk.weights(yc,b);ref=z['cal_reference_correct'];closed=float(np.mean([z['test_reference_correct'][yt==c].mean() for c in b]))
                assert abs(closed-v['reference_closed'])<1e-12
                for row in v['records']:
                    m=row['method'];cc=compact(z,m,'cal',yc,b,n);tc=compact(z,m,'test',yt,b,n)
                    counts['threshold_tests']+=checkpolicy(cc,ref,w,z[m+'_candidates'],row['policy'],tc,closed)
                    for target,fr in [(closed*.95,row['capacity95'])]+[(next(rr['policy']['old'] for rr in v['records'] if rr['method']==k),f) for k,f in row['matched_actual_base'].items()]:
                        ceiling=float(tc['weights'][tc['base_correct']].sum());assert abs(ceiling-fr['ceiling'])<1e-10
                        if fr['feasible']:
                            ab,an=accuracy(tc,fr['offset']);assert ab>=target-1e-10;assert abs(ab-fr['old'])<1e-10 and abs(an-fr['new'])<1e-10
                        else:assert ceiling<target+1e-10
                        counts['frontiers']+=1
                    counts['records']+=1
                counts['archives']+=1
    for p in ([] if a.block else (OUT/'shift/results').glob('*/p*/*.json')):
        if p.name=='complete.json':continue
        v=json.loads(p.read_text());assert v['archive_sha256']==digest_file(p.with_suffix('.npz'))
        with np.load(p.with_suffix('.npz')) as data:z={k:data[k] for k in data.files}
        b,n=z['base_classes'],z['novel_classes'];yt=z['test_labels'];ref=z['test_reference_correct'];closed=float(np.mean([ref[yt==c].mean() for c in b]))
        for row in v['records']:
            if row['method']=='reference':assert row['old']==closed;continue
            m,k=row['method'],row['calibration'];yc=z[k+'_cal_labels'];tc=compact(z,m,'test',yt,b,n);cc=compact(z,m,k+'_cal',yc,b,n)
            counts['threshold_tests']+=checkpolicy(cc,z[k+'_cal_reference_correct'],risk.weights(yc,b),z[m+'_candidates'],row,tc,closed);counts['shift_records']+=1
    externalcells=0
    for ds in ['food','sun']:
        p=OUT/f'external/{ds}/data_manifest.json'
        if not p.exists():continue
        d=json.loads(p.read_text());hashes=[r['sha256'] for rr in d['pools'].values() for r in rr];assert len(hashes)==len(set(hashes));externalcells+=1
        for enc in ['l14','b32']:
            p=OUT/f'external/{ds}/{enc}/verification.json'
            if p.exists():
                check=json.loads(p.read_text());assert check['min_float32_witness_cosine']>.999;assert digest_file(p.parent/'frozen.pt')==check['cache_sha256']
    if not a.partial:
        assert counts['selections']==60+260+40,counts
        assert counts['archives']==3600 and counts['shift_records']==720,counts
        assert externalcells==2
        extra=json.loads((OUT/'external/equal_budget_manifest.json').read_text())
        for f,sha in extra['sources'].items():assert digest_file(ROOT/f)==sha
        subprocess.run([sys.executable,str(ROOT/'src/experiments/verify_expansion_prospective_features.py')],cwd=ROOT,check=True)
        witness=json.loads((OUT/'feature_witness.json').read_text());assert witness['complete'] and witness['max_error']<.005
    receipt=f'{a.block}_partial_verification.json' if a.block else ('partial_verification.json' if a.partial else 'verification.json')
    atomic_json(dict(complete=not a.partial,partial=a.partial,blocks_verified=[a.block] if a.block else ['mechanism','cardinality','external','shift'],**counts,source_hashes_verified=True,identity_disjoint=True,seconds=time.monotonic()-start),OUT/receipt)
    print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
