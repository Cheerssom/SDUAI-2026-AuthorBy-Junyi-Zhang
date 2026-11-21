"""Recompute sampled scores from saved CE weights and original frozen features."""
from pathlib import Path
import sys,json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kbs_reviewer_revision as data
from kbs_runtime import atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_softmax_readout'

@torch.no_grad()
def main():
    torch.set_num_threads(2);errors=[];predictions=0
    for enc in ['l14','b32']:
        d,_=data.load_data('imagenetr',enc);po=OUT/enc/'imagenetr/p0'
        sel=json.loads((po/'selection.json').read_text());source=json.loads((ROOT/f'experiments/kbs_independent_risk/{enc}/imagenetr/p0/selection.json').read_text())
        with np.load(po/'selected_weights.npz') as z:W=torch.from_numpy(z['full_weights']);base=z['base_classes']
        state=torch.load(po/f"full_fits/lambda_{sel['full_fit']['selected_regularization']:g}.pt",map_location='cpu',weights_only=False)
        assert np.array_equal(W.numpy(),state['W'].numpy())
        Pb=torch.stack([d['E_fit'][d['fit_y']==c].mean(0) for c in base]);Pb=Pb/Pb.norm(dim=1,keepdim=True)
        si=np.array(source['support_indices']);Es=d['E_support'][si];ys=d['support_y'][si]
        mask=np.isin(d['cal_y'],base);rng=np.random.default_rng(27102026)
        for session in [1,5]:
            with np.load(po/f'o0_t{session}.npz') as z:
                novel=z['novel_classes'];ids=np.r_[base,novel]
                Pn=torch.stack([Es[ys==c].mean(0) for c in novel]);Pn=Pn/Pn.norm(dim=1,keepdim=True)
                for pool in ['cal','test']:
                    allE=d['E_cal'][mask] if pool=='cal' else d['E_test']
                    ix=rng.choice(len(allE),min(128,len(allE)),replace=False);E=allE[ix]
                    text=E@d['T'][ids].T;learned=E@W[:-1]+W[-1];proto=E@torch.cat([Pb,Pn]).T
                    for mode in ['query','none','fixed']:
                        out=[]
                        for j,(raw,ref) in enumerate([(text,text[:,:len(base)]),(learned,learned),(proto,proto[:,:len(base)])]):
                            mu=ref.mean(1,keepdim=True) if mode=='query' else (0 if mode=='none' else sel['full_moments'][j][0])
                            sd=ref.std(1,correction=0,keepdim=True).clamp_min(1e-6) if mode!='fixed' else sel['full_moments'][j][1]
                            out.append((raw-mu)/sd)
                        params=sel['selections'][mode]['selected'];score=out[0]
                        score[:,:len(base)]+=params['a']*out[1]
                        score[:,len(base):]+=params['u']*out[2][:,len(base):]
                        margin=(score[:,:len(base)].max(1).values.double()-score[:,len(base):].max(1).values.double()).numpy()
                        error=float(np.max(np.abs(margin-z[pool+'_'+mode+'_margin'][ix])))
                        assert error<3e-4,(enc,pool,mode,error)
                        errors.append(error)
                        for group,cols,key in [(base,score[:,:len(base)],'base'),(novel,score[:,len(base):],'novel')]:
                            order=np.argsort(group);pred=group[order][cols[:,order].argmax(1).numpy()]
                            assert np.array_equal(pred,z[pool+'_'+mode+'_'+key+'_prediction'][ix]),(enc,pool,mode,key)
                            predictions+=len(pred)
    report=dict(complete=True,score_blocks=len(errors),predictions_checked=predictions,max_margin_error=max(errors),
                recomputed_on='CPU from saved selected weights, original fitting/support features and fixed prompts',
                tolerance=3e-4)
    atomic_json(report,OUT/'model_witness_verification.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
