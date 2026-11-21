"""Reconstruct frozen-head controls from saved predictions and test traces."""
from pathlib import Path
import json,sys,math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_retention_value as rv
import kbs_risk_control as risk
from kbs_runtime import atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_softmax_readout'
LAMBDAS=np.array([.01,.025,.05,.1,.2,.3,.4,.5,.65,.8,.95,1.])

def compact(z,pool,mode):
    y=z[pool+'_labels'];base,new=z['base_classes'],z['novel_classes']
    bp=z[pool+'_'+mode+'_base_prediction'];npred=z[pool+'_'+mode+'_novel_prediction']
    b=np.isin(y,base);n=np.isin(y,new);w=np.zeros(len(y))
    for group in [base,new]:
        for c in group:
            mask=y==c
            if mask.any():w[mask]=1/(len(group)*mask.sum())
    return dict(margin=z[pool+'_'+mode+'_margin'],base_prediction=bp,novel_prediction=npred,
                base_correct=b&(bp==y),novel_correct=n&(npred==y),base_mask=b,novel_mask=n,
                weights=w,nb=len(base),nn=len(new))

def direct_p(reference,correct,w):
    # Direct product calculation independent of grouped sufficient-statistic code.
    Z=correct.astype(float)-.95*reference
    logs=np.log1p(LAMBDAS[:,None]*(w/w.max())[None]*Z[None]).sum(1)
    logE=np.logaddexp.reduce(logs)-math.log(len(LAMBDAS))
    return float(np.exp(-max(0,logE)))

def main():
    assert (OUT/'complete.json').exists(),'Full experiment has not completed.'
    records=tests=archives=0;source_hashes={}
    for e in ['l14','b32']:
        folder=OUT/e/'imagenetr';manifest=json.loads((folder/'manifest.json').read_text())
        for n,h in manifest['sources'].items():assert digest_file(ROOT/n)==h,n
        for n,h in json.loads((folder/'data_manifest.json').read_text()).items():assert digest_file(ROOT/n)==h,n
        for part in range(10):
            po=folder/f'p{part}';sp=po/'selection.json';sel=json.loads(sp.read_text());assert (po/'complete.json').exists()
            for v in sel['selections'].values():assert v['evaluated_settings']==3636 and len(v['trials'])==36
            for order in [0,1]:
                for session in range(1,6):
                    p=po/f'o{order}_t{session}.json';r=json.loads(p.read_text());raw=p.with_suffix('.npz')
                    assert r['selection_sha256']==digest_file(sp)
                    assert r['raw_sha256']==digest_file(raw)
                    ref=ROOT/r['common_reference_archive'];assert digest_file(ref)==r['common_reference_archive_sha256']
                    source_hashes[ref.relative_to(ROOT).as_posix()]=r['common_reference_archive_sha256']
                    with np.load(raw) as z,np.load(ref) as old:
                        for key in ['base_classes','novel_classes','cal_labels','test_labels','cal_reference_correct','test_reference_correct']:
                            assert np.array_equal(z[key],old[key]),(p,key)
                        closed=float(np.mean([z['test_reference_correct'][z['test_labels']==c].mean() for c in z['base_classes']]))
                        for mode in ['query','none','fixed']:
                            c,tc=compact(z,'cal',mode),compact(z,'test',mode)
                            w=risk.weights(z['cal_labels'],z['base_classes']);refc=z['cal_reference_correct']
                            rows=[x for x in r['records'] if x['method']=='softmax_'+mode]
                            assert len(rows)==8
                            for row in rows:
                                records+=1;assert abs(row['reference_closed_test']-closed)<1e-12
                                if row['policy']=='matched_test':
                                    off,ceil=rv.threshold(tc,row['retention']*closed)
                                    assert (off is not None)==row['feasible'] and abs(ceil-row['ceiling'])<1e-12
                                    if off is None:continue
                                    assert row['offset']==off
                                if row['policy'] in ['protection','paired_accuracy']:
                                    chosen=None
                                    for j,step in enumerate(row['trace']):
                                        assert step['offset']==z[mode+'_candidates'][j]
                                        correct=c['base_correct']&~risk.choose_new(c,step['offset'])
                                        pv=direct_p(refc,correct,w) if row['policy']=='paired_accuracy' else risk.pvalue(risk.protection_loss(refc,correct),w)[0]
                                        assert abs(pv-step['pvalue'])<1e-8,(p,mode,pv,step)
                                        assert step['accepted']==(pv<=.01)
                                        if step['accepted']:chosen=step['offset']
                                        else:assert j==len(row['trace'])-1
                                        tests+=1
                                    assert chosen==row['offset']
                                if row.get('fallback'):
                                    expected=dict(old=closed,new=0.,overall=closed*tc['nb']/(tc['nb']+tc['nn']),harmonic=0.)
                                else:expected=rv.evaluate(tc,row['offset'])
                                for k,v in expected.items():assert abs(row[k]-v)<1e-10,(p,k)
                                if row['policy'] in ['protection','paired_accuracy']:
                                    assert row['test_target_met']==(row['old']>=.95*closed-1e-10)
                                    assert abs(row['old']-(closed-row['destroyed_test']+row['recovered_test']))<1e-10
                        archives+=1
    assert records==4800 and archives==200
    report=dict(complete=True,metric_records=records,prediction_archives=archives,reconstructed_tests=tests,
                common_reference_archives_identical=True,selection_settings_per_scheme=3636,
                source_reference_sha256=source_hashes,scope='Deterministic prediction/selection/trace reconstruction; not independent semantic peer review.')
    atomic_json(report,OUT/'verification.json');print(json.dumps({k:v for k,v in report.items() if k!='source_reference_sha256'},indent=2))
if __name__=='__main__':main()
