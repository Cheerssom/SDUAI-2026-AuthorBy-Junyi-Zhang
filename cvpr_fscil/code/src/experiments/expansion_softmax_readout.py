"""Resumable cross-entropy base-head control on frozen ImageNet-R features."""
from pathlib import Path
import argparse,gc,json,sys,time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_full_fusion_retuning as tuning
import kbs_reviewer_revision as previous
import kbs_reviewer_readouts as readouts
import kbs_calibration_fairness as fairness
import kbs_risk_control as risk
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,atomic_torch,digest_file
ROOT=Path(__file__).resolve().parents[2]
REG=(1e-5,1e-4,1e-3,1e-2)
MODES=('query','none','fixed')
STEPS=300

@torch.enable_grad()
def fit_softmax(E,y,ids,lam,path,max_steps=STEPS):
    keep=np.isin(y,ids);x=E[keep].detach();yy=np.searchsorted(ids,y[keep])
    counts=np.bincount(yy,minlength=len(ids));assert (counts>0).all()
    target=torch.as_tensor(yy,device=E.device,dtype=torch.long)
    weight=torch.as_tensor(1./(len(ids)*counts[yy]),device=E.device,dtype=x.dtype)
    W=torch.zeros((x.shape[1]+1,len(ids)),device=E.device,requires_grad=True)
    opt=torch.optim.Adam([W],lr=.05,betas=(.9,.999),eps=1e-8)
    start=0;initial=None;losses=[]
    if path.exists():
        state=torch.load(path,map_location=E.device,weights_only=False)
        assert state['lambda']==lam and state['classes']==ids.tolist()
        with torch.no_grad():W.copy_(state['W'])
        opt.load_state_dict(state['optimizer']);start=state['step']
        initial=state['initial_loss'];losses=state['losses']
    for step in range(start,max_steps):
        opt.zero_grad(set_to_none=True)
        scores=core.linear(x,W)
        loss=(F.cross_entropy(scores,target,reduction='none')*weight).sum()+lam/2*W[:-1].square().sum()
        assert torch.isfinite(loss)
        if initial is None:initial=float(loss.detach())
        loss.backward();opt.step()
        if (step+1)%50==0 or step+1==max_steps:
            losses.append([step+1,float(loss.detach())])
            atomic_torch(dict(W=W.detach(),optimizer=opt.state_dict(),step=step+1,
                              initial_loss=initial,losses=losses,lambda_=lam,**{'lambda':lam},
                              classes=ids.tolist()),path)
            check_pause()
    assert losses[-1][1]<initial,(initial,losses[-1])
    return W.detach(),dict(initial_loss=initial,losses=losses,steps=max_steps)

@torch.no_grad()
def select_head(Ef,yf,Ev,yv,ids,folder):
    keep=np.isin(yv,ids);best=(-1.,None,None);trials=[]
    for lam in REG:
        check_pause();W,fit=fit_softmax(Ef,yf,ids,lam,folder/f'lambda_{lam:g}.pt')
        pred=ids[core.linear(Ev[keep],W).argmax(1).cpu().numpy()]
        accuracy=float(np.mean([np.mean(pred[yv[keep]==c]==c) for c in ids]))
        trials.append(dict(regularization=lam,tuning_macro=accuracy,fit=fit))
        if accuracy>best[0]+1e-12:best=(accuracy,lam,W)
    return best[2],dict(selected_regularization=best[1],tuning_macro=best[0],trials=trials)

@torch.no_grad()
def run(enc,parts,out,device):
    started=time.perf_counter();folder=out/enc/'imagenetr';folder.mkdir(parents=True,exist_ok=True)
    code=[Path(__file__),Path(tuning.__file__),Path(core.__file__),Path(rv.__file__),
          Path(previous.__file__),Path(readouts.__file__),Path(fairness.__file__),Path(risk.__file__),
          ROOT/'src/experiments/kbs_paired_accuracy.py',ROOT/'src/experiments/kbs_runtime.py']
    config=dict(protocol_sha256=digest_file(ROOT/'experiments/expansion_softmax_readout_protocol.md'),
                sources={p.relative_to(ROOT).as_posix():digest_file(p) for p in code})
    mf=folder/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==config,'Source changed: use a new output directory.'
    else:atomic_json(config,mf)
    if (folder/'complete.json').exists():return
    d,cache=previous.load_data('imagenetr',enc)
    inputs=[cache,ROOT/'experiments/kbs_independent_risk/splits/imagenetr.json',
            ROOT/'experiments/kbs_calibration_budget/pools/imagenetr.json',
            ROOT/f'experiments/kbs_calibration_budget/features/{enc}/imagenetr/extra.pt',
            ROOT/f'experiments/kbs_revision/{enc}/imagenetr/partitions.json']
    data_hashes={p.relative_to(ROOT).as_posix():digest_file(p) for p in inputs}
    dp=folder/'data_manifest.json'
    if dp.exists():assert json.loads(dp.read_text())==data_hashes
    else:atomic_json(data_hashes,dp)
    entries=json.loads(inputs[-1].read_text());Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']]
    yf,ys=d['fit_y'],d['support_y']
    for entry in entries[:parts]:
        check_pause();part=entry['seed'];po=folder/f'p{part}';po.mkdir(exist_ok=True)
        if (po/'complete.json').exists():continue
        base,new,pb,pn=[np.asarray(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        source_selection=ROOT/f'experiments/kbs_independent_risk/{enc}/imagenetr/p{part}/selection.json'
        original=json.loads(source_selection.read_text())
        mask=np.isin(d['tune_y'],base);Ev=d['E_tune'][mask].to(device);yv=d['tune_y'][mask]
        W,fullfit=select_head(Ef,yf,Ev,yv,base,po/'full_fits')
        Wp,pseudofit=select_head(Ef,yf,Ev,yv,pb,po/'pseudo_fits')
        Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
        psi=core.support_rows(ys,pn,5,7);Ppn=core.means(Es[psi],ys[psi],pn)
        ps=po/'selection.json'
        if ps.exists():selection=json.loads(ps.read_text());assert selection['original_selection_sha256']==digest_file(source_selection)
        else:
            pm=readouts.fixed_moments(Ev,T[pb],Ppb,Wp);fm=readouts.fixed_moments(Ev,T[base],Pb,W)
            selections={}
            for mode in MODES:
                comp=tuning.components(Ev,T[np.r_[pb,pn]],Ppb,Ppn,Wp,pm,mode)
                selections[mode]=tuning.select(comp,'retained',np.r_[pb,pn],yv,pb,pn)
            selection=dict(encoder=enc,dataset='imagenetr',part=part,full_fit=fullfit,pseudo_fit=pseudofit,
                           original_selection_sha256=digest_file(source_selection),full_moments=fm,
                           pseudo_moments=pm,selections=selections,
                           selection_uses='original fitting and base-only tuning pools; no calibration/test outcomes')
            atomic_json(selection,ps)
            atomic_npz(po/'selected_weights.npz',full_weights=W.cpu().numpy(),pseudo_weights=Wp.cpu().numpy(),
                       base_classes=base,pseudo_base_classes=pb)
        maskc=np.isin(d['cal_y'],base)
        queries={'tune':Ev,'cal':d['E_cal'][maskc].to(device),'test':d['E_test'].to(device)}
        labels={'tune':yv,'cal':d['cal_y'][maskc],'test':d['test_y']};wc=risk.weights(labels['cal'],base)
        si=np.asarray(original['support_indices']);count=0
        for order in [0,1]:
            ordered=np.random.default_rng(1000+part+100*order).permutation(new)
            for session in range(1,6):
                check_pause();dest=po/f'o{order}_t{session}.json'
                if dest.exists():continue
                source=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/imagenetr/p{part}/o{order}_t{session}.npz'
                with np.load(source) as z:refc=z['cal_reference_correct'];reft=z['test_reference_correct']
                closed=float(np.mean([reft[labels['test']==c].mean() for c in base]))
                novel=ordered[:len(new)*session//5];ids=np.r_[base,novel];Pn=core.means(Es[si],ys[si],novel)
                archive=dict(base_classes=base,novel_classes=novel,cal_labels=labels['cal'],test_labels=labels['test'],
                             cal_reference_correct=refc,test_reference_correct=reft);results=[]
                for mode in MODES:
                    param=selection['selections'][mode]['selected'];systems={}
                    for pool in ['tune','cal','test']:
                        comp=tuning.components(queries[pool],T[ids],Pb,Pn,W,selection['full_moments'],mode)
                        c=rv.compact(tuning.scores(comp,'retained',param['a'],param['u']),ids,labels[pool],base,novel)
                        systems[pool]=c
                        if pool!='tune':archive.update({pool+'_'+mode+'_'+k:v for k,v in tuning.pack(c).items()})
                    tc,cc=systems['test'],systems['cal'];candidates=np.unique(np.quantile(systems['tune']['margin'],np.linspace(0,1,101)))
                    archive[mode+'_candidates']=candidates
                    common=dict(encoder=enc,dataset='imagenetr',part=part,order=order,session=session,
                                method='softmax_'+mode,reference_closed_test=closed)
                    results.append(dict(**common,policy='fixed_gate',offset=param['offset'],fallback=False,**rv.evaluate(tc,param['offset'])))
                    for policy in ['protection','paired_accuracy']:
                        detail=fairness.evaluate_policy(cc,refc,wc,candidates,tc,reft,closed,policy)
                        assert abs(detail['old']-(closed-detail['destroyed_test']+detail['recovered_test']))<1e-9
                        results.append(dict(**common,**detail))
                    for ratio in [.80,.85,.90,.95,.99]:
                        offset,ceiling=rv.threshold(tc,ratio*closed)
                        results.append(dict(**common,policy='matched_test',retention=ratio,feasible=offset is not None,
                                            ceiling=ceiling,offset=offset,**(rv.evaluate(tc,offset) if offset is not None else {})))
                raw=dest.with_suffix('.npz');atomic_npz(raw,**archive)
                atomic_json(dict(selection_sha256=digest_file(ps),common_reference_archive=source.relative_to(ROOT).as_posix(),
                                 common_reference_archive_sha256=digest_file(source),raw_sha256=digest_file(raw),records=results),dest)
                count+=len(results)
        atomic_json(dict(complete=True,selection_sha256=digest_file(ps),part=part,reference_closed_test=closed),po/'complete.json')
        print('SOFTMAX COMPLETE',enc,part,'seconds',round(time.perf_counter()-started,1),'records',count,flush=True)
    if parts==10:atomic_json(dict(complete=True,partitions=10,seconds=time.perf_counter()-started,
                                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0),folder/'complete.json')
    del d,Ef,Es,T;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--parts',type=int,default=10)
    ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    device=configure();check_pause();torch.manual_seed(27102026)
    out=ROOT/('experiments/expansion_softmax_readout_pilot' if args.pilot else 'experiments/expansion_softmax_readout')
    for enc in ([args.encoder] if args.encoder else ['l14','b32']):run(enc,args.parts,out,device)
    if args.parts==10 and not args.encoder:atomic_json(dict(complete=True,conditions=2,partitions=20),out/'complete.json')
if __name__=='__main__':main()
