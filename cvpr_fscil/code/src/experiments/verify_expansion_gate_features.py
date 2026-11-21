"""CPU feature/model witness for the new learned gates, six frozen conditions."""
from pathlib import Path
import sys,json,gc
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import expansion_gate_calibration as ex
import fcs_expansion as core
from kbs_runtime import atomic_json,digest_file

def main():
    torch.set_num_threads(2);rows=[]
    for enc in ex.ENCODERS:
        for ds in ex.DATASETS:
            d,_=ex.previous.load_data(ds,enc)
            entry=json.loads((ex.ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json').read_text())[0]
            sel=json.loads((ex.ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p0/selection.json').read_text())
            saved=json.loads((ex.OUT/f'gates/{enc}/{ds}/p0/selection.json').read_text())
            base=np.array(entry['base']);new=np.random.default_rng(1000).permutation(np.array(entry['novel']));ids=np.r_[base,new]
            W=core.fit_ridge(d['E_fit'],d['fit_y'],base,sel['ridge_lambda'])
            Pb=core.means(d['E_fit'],d['fit_y'],base);si=np.array(sel['support_indices'])
            Pn=core.means(d['E_support'][si],d['support_y'][si],new)
            z=ex.loadz(ex.OUT/f'gates/{enc}/{ds}/p0/o0_t5.npz')
            ix=np.random.default_rng(20261028).choice(len(d['test_y']),128,replace=False)
            X,c=ex.features(d['E_test'][ix],d['T'][ids],Pb,Pn,W,sel['parameters']['retained'],ids,d['test_y'][ix],base,new)
            jp=json.loads((ex.OUT/f'gates/{enc}/{ds}/p0/o0_t5.json').read_text())
            for kind in ex.GATES:
                margin=ex.apply_gate(saved['gates'][kind],X);reference=z['test_'+kind+'_margin'][ix]
                error=float(np.max(abs(margin-reference)));assert error<.005,(enc,ds,kind,error)
                for key in ['base_prediction','novel_prediction']:
                    assert np.array_equal(c[key],z['test_'+kind+'_'+key][ix])
                policy=next(r for r in jp['records'] if r['method']==kind and r['policy']=='paired_accuracy')
                pred_match=128
                if policy['offset'] is not None:
                    cc=dict(c,margin=margin);original=dict(c,margin=reference)
                    pred_match=int((ex.choose(cc,policy['offset'])==ex.choose(original,policy['offset'])).sum())
                    assert pred_match==128,(enc,ds,kind,pred_match)
                rows.append(dict(encoder=enc,dataset=ds,method=kind,queries=128,cpu_margin_max_error=error,admission_predictions_matched=pred_match))
            del d;gc.collect()
    result=dict(complete=True,rows=rows,total_admission_predictions=sum(r['queries'] for r in rows),
                verifier_sha256=digest_file(Path(__file__)))
    atomic_json(result,ex.OUT/'feature_witness.json');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
