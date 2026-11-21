"""CPU feature-to-score witness independent of the cardinality runner."""
from pathlib import Path
import json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import kbs_reviewer_revision as previous
from kbs_runtime import atomic_json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_cardinality_rule_v2'

@torch.no_grad()
def main():
    torch.set_num_threads(2);witnesses=[]
    for enc in ['l14','b32']:
        for ds in ['pets','cifar','imagenetr']:
            d,_=previous.load_data(ds,enc)
            entry=json.loads((ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json').read_text())[0]
            original=json.loads((ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p0/selection.json').read_text())
            selection=json.loads((OUT/enc/ds/'p0/selection.json').read_text())
            base=np.asarray(entry['base']);W=core.fit_ridge(d['E_fit'],d['fit_y'],base,original['ridge_lambda'])
            si=np.asarray(original['support_indices'])
            with np.load(OUT/enc/ds/'p0/o0_t5.npz') as z:
                novel=z['novel_classes'];ids=np.r_[base,novel];nb=len(base);P0=core.means(d['E_fit'],d['fit_y'],base);Pn=core.means(d['E_support'][si],d['support_y'][si],novel)
                ix=np.linspace(0,len(d['E_test'])-1,128,dtype=int);E=d['E_test'][ix];T=d['T'][ids]
                branches=[E@T.T,core.linear(E,W),E@torch.cat([P0,Pn]).T]
                eta=min(1.,nb/30.);normalized=[]
                for j,q in enumerate(branches):
                    ref=q if j==1 else q[:,:nb];mu=eta*ref.mean(1,keepdim=True)+(1-eta)*selection['full_moments'][j][0]
                    sigma=eta*ref.std(1,correction=0,keepdim=True).clamp_min(1e-6)+(1-eta)*selection['full_moments'][j][1]
                    normalized.append((q-mu)/sigma.clamp_min(1e-6))
                par=selection['selections']['primary']['selected'];scores=normalized[0].clone()
                scores[:,:nb]+=par['a']*normalized[1];scores[:,nb:]+=par['u']*normalized[2][:,nb:]
                bp=scores[:,:nb].argmax(1);npred=scores[:,nb:].argmax(1)
                margin=(scores[:,:nb].max(1).values-scores[:,nb:].max(1).values).numpy()
                err=float(np.max(np.abs(margin-z['test_primary_margin'][ix])))
                assert err<2e-4,(enc,ds,err)
                assert np.array_equal(base[bp.numpy()],z['test_primary_base_prediction'][ix])
                assert np.array_equal(novel[npred.numpy()],z['test_primary_novel_prediction'][ix])
                witnesses.append(dict(encoder=enc,dataset=ds,queries=len(ix),maximum_margin_error=err))
    report=dict(complete=True,scope='Sampled CPU feature reconstruction on partition0, final first-order session',conditions=witnesses)
    atomic_json(report,OUT/'feature_witness.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
