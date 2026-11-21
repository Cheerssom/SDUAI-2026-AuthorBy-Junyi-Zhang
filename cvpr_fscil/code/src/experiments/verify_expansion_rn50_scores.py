"""Independent sampled CNN readout scores from saved weights and embeddings."""
from pathlib import Path
import sys,json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import expansion_rn50_evaluate as experiment
from kbs_runtime import atomic_json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_rn50'

@torch.no_grad()
def main():
    torch.set_num_threads(2);d=experiment.load();errors=[];checked=0
    for part in [0,9]:
        po=OUT/'evaluation'/f'p{part}';sel=json.loads((po/'selection.json').read_text())
        with np.load(po/'selected_weights.npz') as z:W=torch.from_numpy(z['full_weights']);base=z['base_classes']
        Pb=torch.stack([d['E_fit'][d['fit_y']==c].mean(0) for c in base]);Pb=Pb/Pb.norm(dim=1,keepdim=True)
        si=np.array(sel['support_indices']);Es=d['E_support'][si];ys=d['support_y'][si]
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
                        for head in ['retained','dense']:
                            name=head+'_'+mode;params=sel['selections'][name]['selected'];score=out[0].clone()
                            score[:,:len(base)]+=params['a']*(out[1] if head=='retained' else out[2][:,:len(base)])
                            score[:,len(base):]+=params['u']*out[2][:,len(base):]
                            margin=(score[:,:len(base)].max(1).values.double()-score[:,len(base):].max(1).values.double()).numpy()
                            error=float(np.max(np.abs(margin-z[pool+'_'+name+'_margin'][ix])))
                            assert error<3e-4,(part,pool,name,error);errors.append(error)
                            for group,cols,key in [(base,score[:,:len(base)],'base'),(novel,score[:,len(base):],'novel')]:
                                order=np.argsort(group);pred=group[order][cols[:,order].argmax(1).numpy()]
                                assert np.array_equal(pred,z[pool+'_'+name+'_'+key+'_prediction'][ix]),(part,pool,name,key)
                                checked+=len(pred)
    report=dict(complete=True,score_blocks=len(errors),predictions_checked=checked,max_margin_error=max(errors),
                recomputed_on='CPU from saved selected RN50 ridge weights and original frozen features',tolerance=3e-4)
    atomic_json(report,OUT/'model_witness_verification.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
