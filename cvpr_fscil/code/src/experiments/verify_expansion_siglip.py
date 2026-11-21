"""Deterministic prediction/threshold reconstruction of the SigLIP transfer."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_retention_value as rv
import kbs_risk_control as risk
from verify_expansion_softmax_readout import compact,direct_p
from kbs_runtime import atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_cross_encoder'

def main():
    folder=OUT/'evaluation';assert (folder/'complete.json').exists()
    cfg=json.loads((folder/'manifest.json').read_text())
    assert digest_file(OUT/'features/frozen.pt')==cfg['feature_sha256']
    for n,h in cfg['sources'].items():assert digest_file(ROOT/n)==h
    native=json.loads((OUT/'features/native_witness.json').read_text());assert native['complete']
    assert native['cache_sha256']==cfg['feature_sha256']
    methods=[a+'_'+b for b in ['query','none','fixed'] for a in ['retained','dense']]+['hycal']
    records=tests=archives=0
    for part in range(10):
        po=folder/f'p{part}';sp=po/'selection.json';sel=json.loads(sp.read_text())
        for s in sel['selections'].values():assert s['evaluated_settings']==3636
        for order in [0,1]:
            for session in range(1,6):
                p=po/f'o{order}_t{session}.json';r=json.loads(p.read_text());raw=p.with_suffix('.npz')
                assert r['selection_sha256']==digest_file(sp) and r['raw_sha256']==digest_file(raw)
                assert len(r['records'])==56
                with np.load(raw) as z:
                    w=risk.weights(z['cal_labels'],z['base_classes']);ref=z['cal_reference_correct']
                    closed=float(np.mean([z['test_reference_correct'][z['test_labels']==c].mean() for c in z['base_classes']]))
                    for method in methods:
                        c,tc=compact(z,'cal',method),compact(z,'test',method)
                        for row in [r for r in r['records'] if r['method']==method]:
                            records+=1;assert abs(row['reference_closed_test']-closed)<1e-12
                            if row['policy']=='matched_test':
                                offset,ceiling=rv.threshold(tc,row['retention']*closed)
                                assert row['offset']==offset and row['feasible']==(offset is not None)
                                if offset is None:continue
                            if row['policy'] in ['protection','paired_accuracy']:
                                accepted=None
                                for j,step in enumerate(row['trace']):
                                    assert step['offset']==z[method+'_candidates'][j]
                                    cor=c['base_correct']&~risk.choose_new(c,step['offset'])
                                    pv=direct_p(ref,cor,w) if row['policy']=='paired_accuracy' else risk.pvalue(risk.protection_loss(ref,cor),w)[0]
                                    assert abs(pv-step['pvalue'])<1e-8 and step['accepted']==(pv<=.01)
                                    if step['accepted']:accepted=step['offset']
                                    else:assert j==len(row['trace'])-1
                                    tests+=1
                                assert row['offset']==accepted
                            expected=dict(old=closed,new=0.,overall=closed*tc['nb']/(tc['nb']+tc['nn']),harmonic=0.) if row.get('fallback') else rv.evaluate(tc,row['offset'])
                            for k,v in expected.items():assert abs(row[k]-v)<1e-10
                            if row['policy'] in ['protection','paired_accuracy']:
                                assert abs(row['old']-(closed-row['destroyed_test']+row['recovered_test']))<1e-10
                                assert row['test_target_met']==(row['old']>=.95*closed-1e-10)
                archives+=1
    assert records==5600 and archives==100
    report=dict(complete=True,metric_records=records,prediction_archives=archives,reconstructed_tests=tests,
                source_hashes_verified=True,native_model_witness=native,
                scope='Deterministic reconstruction; matched transfer under encoder-specific common reference.')
    atomic_json(report,OUT/'verification.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
