"""Reconstruct shrinkage outcomes and enforce tuning-only adaptive choices."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_retention_value as rv
import kbs_risk_control as risk
from verify_expansion_softmax_readout import compact,direct_p
from kbs_runtime import atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_shrinkage'

def main():
    assert (OUT/'complete.json').exists()
    tests=records=archives=0
    for enc in ['l14','b32']:
        for ds in ['pets','cifar','imagenetr']:
            folder=OUT/enc/ds
            for n,h in json.loads((folder/'manifest.json').read_text())['sources'].items():assert digest_file(ROOT/n)==h
            for n,h in json.loads((folder/'data_manifest.json').read_text()).items():assert digest_file(ROOT/n)==h
            for part in range(10):
                po=folder/f'p{part}';sp=po/'selection.json';sel=json.loads(sp.read_text())
                assert max(sel['endpoint_max_errors'].values())<1e-6
                for name,opts in [('adaptive_eta',['fixed','half','query']),('adaptive_scheme',['fixed','half','query','none'])]:
                    assert sel[name]==max(opts,key=lambda k:sel['selections'][k]['selected']['pseudo_harmonic'])
                for order in [0,1]:
                    for session in range(1,6):
                        p=po/f'o{order}_t{session}.json';r=json.loads(p.read_text());raw=p.with_suffix('.npz');frozen=ROOT/r['frozen_source']
                        assert r['raw_sha256']==digest_file(raw) and r['selection_sha256']==digest_file(sp)
                        assert r['frozen_source_sha256']==digest_file(frozen) and r['frozen_raw_sha256']==digest_file(frozen.with_suffix('.npz'))
                        old=json.loads(frozen.read_text())['records'];rows=r['records'];assert len(rows)==48
                        for mode in ['query','none','fixed']:
                            want=[dict(x,method=mode) for x in old if x['method']=='retained_'+mode]
                            assert [x for x in rows if x['method']==mode]==want
                        for adaptive in ['adaptive_eta','adaptive_scheme']:
                            chosen=sel[adaptive]
                            assert [x for x in rows if x['method']==adaptive]==[dict(x,method=adaptive,chosen_scheme=chosen) for x in rows if x['method']==chosen]
                        with np.load(raw) as z,np.load(frozen.with_suffix('.npz')) as orig:
                            for k in ['base_classes','novel_classes','cal_labels','test_labels','cal_reference_correct','test_reference_correct']:
                                assert np.array_equal(z[k],orig[k])
                            c,tc=compact(z,'cal','half'),compact(z,'test','half');w=risk.weights(z['cal_labels'],z['base_classes'])
                            closed=float(np.mean([z['test_reference_correct'][z['test_labels']==cl].mean() for cl in z['base_classes']]))
                            for row in [x for x in rows if x['method']=='half']:
                                if row['policy']=='matched_test':
                                    off,ceil=rv.threshold(tc,row['retention']*closed)
                                    assert row['feasible']==(off is not None) and row['offset']==off
                                    if off is None:continue
                                if row['policy'] in ['paired_accuracy','protection']:
                                    chosen=None
                                    for j,step in enumerate(row['trace']):
                                        assert step['offset']==z['half_candidates'][j]
                                        cor=c['base_correct']&~risk.choose_new(c,step['offset']);ref=z['cal_reference_correct']
                                        pv=direct_p(ref,cor,w) if row['policy']=='paired_accuracy' else risk.pvalue(risk.protection_loss(ref,cor),w)[0]
                                        assert abs(pv-step['pvalue'])<1e-8 and step['accepted']==(pv<=.01)
                                        if step['accepted']:chosen=step['offset']
                                        else:assert j==len(row['trace'])-1
                                        tests+=1
                                    assert row['offset']==chosen
                                expected=dict(old=closed,new=0.,overall=closed*tc['nb']/(tc['nb']+tc['nn']),harmonic=0.) if row.get('fallback') else rv.evaluate(tc,row['offset'])
                                for k,v in expected.items():assert abs(row[k]-v)<1e-10
                        records+=len(rows);archives+=1
    assert records==28800 and archives==600
    report=dict(complete=True,metric_records=records,prediction_archives=archives,new_threshold_tests=tests,
                endpoints_reconstructed=True,original_controls_unchanged=True,tuning_only_adaptive_choices_verified=True)
    atomic_json(report,OUT/'verification.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
