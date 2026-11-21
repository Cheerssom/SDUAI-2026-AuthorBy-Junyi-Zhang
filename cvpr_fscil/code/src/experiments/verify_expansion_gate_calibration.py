"""Independent reconstruction of the four additional experiment blocks."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import expansion_gate_calibration as ex
import fcs_retention_value as rv
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
from kbs_runtime import atomic_json,digest_file

ROOT=ex.ROOT;OUT=ex.OUT

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    for name,h in manifest['sources'].items():assert digest_file(ROOT/name)==h,name
    matching=gate_records=archives=tests=stability=simulation=0;max_margin=0.;variance_error=0.
    for enc in ex.ENCODERS:
        for ds in ex.DATASETS:
            inputs=json.loads((OUT/f'gates/{enc}/{ds}/inputs.json').read_text())
            for name,h in inputs.items():assert digest_file(ROOT/name)==h
            for p in range(10):
                mp=OUT/f'matched/{enc}/{ds}/p{p}.json';m=json.loads(mp.read_text())
                for name,h in m['inputs'].items():assert digest_file(ROOT/name)==h
                assert len(m['records'])==240
                # Reconstruct all matched frontier values from the original archives.
                for order in [0,1]:
                    for session in range(1,6):
                        old=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p{p}/o{order}_t{session}.npz'
                        z=ex.loadz(old);base=z['base_classes'];new=z['novel_classes'];y=z['test_labels']
                        for r in [r for r in m['records'] if r['order']==order and r['session']==session]:
                            c=ex.cm(z,r['method'],y,base,new,'test_');d,ceiling=rv.threshold(c,r['target']);f=r['frontier']
                            assert f['feasible']==(d is not None)
                            if d is not None:
                                a=rv.evaluate(c,d);assert abs(a['old']-f['base'])<1e-12 and abs(a['new']-f['novel'])<1e-12
                                assert -1e-10<=f['overshoot']
                            st=r['error_structure']
                            assert abs(st['base']-r['selected_base'])<1e-10
                            assert abs(st['reference']-st['destroyed']+st['recovered']-st['base'])<1e-10
                            matching+=1
                gp=OUT/f'gates/{enc}/{ds}/p{p}';sp=gp/'selection.json';s=json.loads(sp.read_text())
                assert s['fit_before_calibration'] and 'base-only' in s['gates']['logistic']['tuning_labels']
                for gate in ex.GATES:
                    state=s['gates'][gate];assert state['features']==ex.FEATURES and len(state['trials'])==4
                    assert state['selected']==max(state['trials'],key=lambda r:r['harmonic'])
                for order in [0,1]:
                    for session in range(1,6):
                        jp=gp/f'o{order}_t{session}.json';obj=json.loads(jp.read_text());zp=jp.with_suffix('.npz')
                        assert obj['selection_sha256']==digest_file(sp) and obj['raw_sha256']==digest_file(zp)
                        assert len(obj['records'])==10;gate_records+=10;archives+=1
                        a=ex.loadz(zp);base=a['base_classes'];new=a['novel_classes'];yc=a['cal_labels'];yt=a['test_labels']
                        calref=a['cal_reference_correct'];testref=a['test_reference_correct'];closed=float(np.mean([testref[yt==cl].mean() for cl in base]))
                        max_margin=max(max_margin,*obj['witness'].values())
                        for gate in ex.GATES:
                            c=ex.cm(a,gate,yc,base,new,'cal_');tc=ex.cm(a,gate,yt,base,new,'test_');w=risk.weights(yc,base)
                            for row in [r for r in obj['records'] if r['method']==gate]:
                                if row['policy'] in ['protection','paired_accuracy']:
                                    for trace in row['trace']:
                                        corr=c['base_correct']&~risk.choose_new(c,trace['offset'])
                                        pp=(paired.pvalue(calref,corr,w)[0] if row['policy']=='paired_accuracy' else risk.pvalue(risk.protection_loss(calref,corr),w)[0])
                                        assert abs(pp-trace['pvalue'])<1e-11
                                        assert trace['accepted']==(pp<=.01);tests+=1
                                    accepted=[t['offset'] for t in row['trace'] if t['accepted']]
                                    assert row['offset']==(accepted[-1] if accepted else None)
                                    if row['offset'] is None:assert row['new']==0 and abs(row['old']-closed)<1e-12
                                    else:
                                        v=rv.evaluate(tc,row['offset']);assert abs(v['old']-row['old'])<1e-12 and abs(v['new']-row['new'])<1e-12
                                    assert row['recovered_test']<1e-12
                                elif row['policy']=='matched_test':
                                    off,_=rv.threshold(tc,row['retention']*closed);assert row['feasible']==(off is not None)
                                    if off is not None:assert abs(rv.evaluate(tc,off)['new']-row['novel'])<1e-12
                st=json.loads((OUT/f'stability/{enc}/{ds}/p{p}.json').read_text());assert len(st['records'])==1452
                for name,h in st['inputs'].items():assert digest_file(ROOT/name)==h
                stability+=len(st['records'])
                # Independently rerun selected deterministic subset witnesses, using scalar certification.
                cal=ex.loadz(ROOT/f'experiments/kbs_calibration_budget/{enc}/{ds}/p{p}/o0_t5.npz')
                test=ex.loadz(ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{p}/o0_t5.npz');gate=ex.loadz(gp/'o0_t5.npz')
                base=cal['base_classes'];new=cal['novel_classes'];y=cal['labels']
                for cap,seed in [(2,100),(10,117),(40,100)]:
                    rng=np.random.default_rng(771000+seed);idx=[rng.permutation(np.flatnonzero(y==cl))[:min(cap,int((y==cl).sum()))] for cl in base]
                    take=np.sort(np.concatenate(idx));w=risk.weights(y[take],base);rr=cal['reference_correct'][take]
                    for method in ['retained','anchored_tp','logistic','mlp']:
                        learned=method in ex.GATES;src=gate if learned else cal
                        c=ex.cm(src,method,y,base,new,'cal_' if learned else '')
                        c={k:(v[take] if isinstance(v,np.ndarray) else v) for k,v in c.items()};c['weights']=w
                        cand=(gate if learned else test)[method+'_candidates']
                        for policy,module in [('paired_accuracy',paired),('protection',risk)]:
                            got=module.certify(c,rr,w,cand)
                            row=next(r for r in st['records'] if r['cap']==cap and r['subset_seed']==seed and r['method']==method and r['policy']==policy)
                            assert got['offset']==row['offset'];tests+=len(got['trace'])
            sim=json.loads((OUT/'coupling'/f'{enc}_{ds}.json').read_text());assert len(sim['records'])==20
            ar=np.array(sim['reference_class_rates']);ac=.975*ar;B=len(ar)
            for row in sim['records']:
                g=row['recovery_fraction']*np.minimum(1-ar,ac)
                v=ac+.95**2*ar-2*.95*(ac-g)-(ac-.95*ar)**2
                expected=float(v.sum()/row['cap']/B**2);variance_error=max(variance_error,abs(expected-row['exact_mean_variance']))
                assert abs(expected-row['exact_mean_variance'])<1e-15
                assert 0<=row['acceptance']<=1 and row['draws']==500
                simulation+=row['draws']
    receipt=dict(complete=True,matched_records=matching,gate_metric_records=gate_records,gate_prediction_archives=archives,
        reconstructed_threshold_tests=tests,stability_metric_records=stability,coupling_simulation_draws=simulation,
        gate_reconstruction_max_margin_error=max_margin,coupling_variance_max_error=variance_error,
        all_input_hashes_verified=True,selection_excludes_calibration_test=True)
    atomic_json(receipt,OUT/'verification.json');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
