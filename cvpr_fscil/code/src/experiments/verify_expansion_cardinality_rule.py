"""Verify the stage-consistent rule and all untouched comparison controls."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_retention_value as rv
import kbs_risk_control as risk
from verify_expansion_softmax_readout import compact,direct_p
from kbs_runtime import atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_cardinality_rule_v2'

def main():
    assert (OUT/'complete.json').exists()
    tests=records=archives=0
    for enc in ['l14','b32']:
        for ds in ['pets','cifar','imagenetr']:
            folder=OUT/enc/ds
            for n,h in json.loads((folder/'manifest.json').read_text())['sources'].items():assert digest_file(ROOT/n)==h,n
            for n,h in json.loads((folder/'data_manifest.json').read_text()).items():assert digest_file(ROOT/n)==h,n
            entries=json.loads((ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json').read_text())
            for part in range(10):
                po=folder/f'p{part}';sp=po/'selection.json';sel=json.loads(sp.read_text());e=entries[part]
                assert sel['eta_pseudo']==min(1.,len(e['pseudo_base'])/30.)
                assert sel['eta_full']==min(1.,len(e['base'])/30.)
                expected_frozen='half' if ds=='pets' else 'query'
                assert sel['catalog_frozen_method']==expected_frozen
                chosen=sel['selections']['primary'];assert chosen['evaluated_settings']==3636
                best=max(float(v) for t in chosen['trials'] for v in t['harmonics'])
                assert abs(best-chosen['selected']['pseudo_harmonic'])<1e-12
                if ds=='imagenetr':
                    old=json.loads((ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p{part}/selection.json').read_text())
                    assert chosen==old['selections']['retained_query']
                for order in [0,1]:
                    for session in range(1,6):
                        p=po/f'o{order}_t{session}.json';r=json.loads(p.read_text());raw=p.with_suffix('.npz');frozen=ROOT/r['frozen_source']
                        assert r['raw_sha256']==digest_file(raw) and r['selection_sha256']==digest_file(sp)
                        assert r['frozen_source_sha256']==digest_file(frozen) and r['frozen_raw_sha256']==digest_file(frozen.with_suffix('.npz'))
                        assert r['catalog_frozen_sha256']==digest_file(ROOT/r['catalog_frozen_source'])
                        old=json.loads(frozen.read_text())['records'];rows=r['records'];assert len(rows)==48
                        for mode in ['query','none','fixed','dense']:
                            name='dense_query' if mode=='dense' else 'retained_'+mode
                            assert [x for x in rows if x['method']==mode]==[dict(x,method=mode) for x in old if x['method']==name]
                        catalog=json.loads((ROOT/r['catalog_frozen_source']).read_text())['records']
                        assert [x for x in rows if x['method']=='catalog_frozen']==[dict(x,method='catalog_frozen',chosen_scheme=expected_frozen) for x in catalog if x['method']==expected_frozen]
                        with np.load(raw) as z,np.load(frozen.with_suffix('.npz')) as orig:
                            for k in ['base_classes','novel_classes','cal_labels','test_labels','cal_reference_correct','test_reference_correct']:assert np.array_equal(z[k],orig[k]),k
                            c,tc=compact(z,'cal','primary'),compact(z,'test','primary');w=risk.weights(z['cal_labels'],z['base_classes'])
                            closed=float(np.mean([z['test_reference_correct'][z['test_labels']==cl].mean() for cl in z['base_classes']]))
                            for row in [x for x in rows if x['method']=='primary']:
                                if row['policy']=='matched_test':
                                    off,ceil=rv.threshold(tc,row['retention']*closed)
                                    assert row['feasible']==(off is not None) and row['offset']==off
                                    if off is None:continue
                                if row['policy'] in ['paired_accuracy','protection']:
                                    accepted=None
                                    for j,step in enumerate(row['trace']):
                                        assert step['offset']==z['primary_candidates'][j]
                                        cor=c['base_correct']&~risk.choose_new(c,step['offset']);ref=z['cal_reference_correct']
                                        pv=direct_p(ref,cor,w) if row['policy']=='paired_accuracy' else risk.pvalue(risk.protection_loss(ref,cor),w)[0]
                                        assert abs(pv-step['pvalue'])<1e-8 and step['accepted']==(pv<=.01)
                                        if step['accepted']:accepted=step['offset']
                                        else:assert j==len(row['trace'])-1
                                        tests+=1
                                    assert row['offset']==accepted
                                want=dict(old=closed,new=0.,overall=closed*tc['nb']/(tc['nb']+tc['nn']),harmonic=0.) if row.get('fallback') else rv.evaluate(tc,row['offset'])
                                for k,v in want.items():assert abs(row[k]-v)<1e-10
                        records+=len(rows);archives+=1
    assert records==28800 and archives==600
    result=dict(complete=True,metric_records=records,prediction_archives=archives,new_threshold_tests=tests,
                stage_cardinality_checked=True,untouched_controls_checked=True,catalog_frozen_control_checked=True,
                imagenetr_selection_identical_to_query=True,design_constant_postdevelopment=30)
    atomic_json(result,OUT/'verification.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
