"""Matched cross-pretraining readouts and common-reference risk calibration."""
from pathlib import Path
import json,sys,time,argparse
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_full_fusion_retuning as tuning
import kbs_reviewer_revision as previous
import kbs_reviewer_readouts as readouts
import kbs_calibration_fairness as fairness
import kbs_risk_control as risk
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_cross_encoder'

def load():
    cache=OUT/'features/frozen.pt';raw=torch.load(cache,map_location='cpu',weights_only=False)
    split=json.loads((ROOT/'experiments/kbs_independent_risk/splits/imagenetr.json').read_text())
    pool=json.loads((ROOT/'experiments/kbs_calibration_budget/pools/imagenetr.json').read_text())
    E=torch.cat([raw['E_fit'],raw['E_val']]);y=np.r_[raw['fit_y'],raw['val_y']];pieces=[];j=0
    for r in pool['rows']:
        if 'cached_index' in r:pieces.append(E[r['cached_index']])
        else:pieces.append(raw['E_extra'][j]);j+=1
    return dict(E_fit=E[split['indices']['fit']],fit_y=y[split['indices']['fit']],
                E_tune=E[split['indices']['tune']],tune_y=y[split['indices']['tune']],
                E_cal=torch.stack(pieces),cal_y=np.array([r['class_id'] for r in pool['rows']]),
                E_test=raw['E_test'],test_y=raw['test_y'],E_support=raw['E_support'],support_y=raw['support_y'],T=raw['T100'])

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parts',type=int,default=10);args=ap.parse_args()
    device=configure();check_pause();folder=OUT/'evaluation';folder.mkdir(exist_ok=True);started=time.perf_counter()
    code=[Path(__file__),Path(core.__file__),Path(rv.__file__),Path(tuning.__file__),Path(previous.__file__),Path(readouts.__file__),Path(fairness.__file__),Path(risk.__file__),ROOT/'src/experiments/kbs_paired_accuracy.py']
    manifest=dict(protocol_sha256=digest_file(ROOT/'experiments/expansion_cross_encoder_protocol.md'),
                  feature_sha256=digest_file(OUT/'features/frozen.pt'),sources={p.relative_to(ROOT).as_posix():digest_file(p) for p in code})
    mp=folder/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:atomic_json(manifest,mp)
    if (folder/'complete.json').exists():return
    d=load();Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']];yf,ys=d['fit_y'],d['support_y']
    hycal=previous.prepare_hycal(d,folder,device)
    entries=json.loads((ROOT/'experiments/kbs_revision/l14/imagenetr/partitions.json').read_text())
    for entry in entries[:args.parts]:
        check_pause();part=entry['seed'];po=folder/f'p{part}';po.mkdir(exist_ok=True)
        if (po/'complete.json').exists():continue
        base,new,pb,pn=[np.asarray(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        Evall=d['E_tune'].to(device);_,lam=core.select_ridge(Ef,yf,Evall,d['tune_y'],base)
        _,plam=core.select_ridge(Ef,yf,Evall,d['tune_y'],pb)
        W=core.fit_ridge(Ef,yf,base,lam);Wp=core.fit_ridge(Ef,yf,pb,plam)
        Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
        tm=np.isin(d['tune_y'],base);Ev=Evall[tm];yv=d['tune_y'][tm]
        psi=core.support_rows(ys,pn,5,7);Ppn=core.means(Es[psi],ys[psi],pn)
        sp=po/'selection.json'
        if sp.exists():selection=json.loads(sp.read_text())
        else:
            pm=readouts.fixed_moments(Ev,T[pb],Ppb,Wp);fm=readouts.fixed_moments(Ev,T[base],Pb,W)
            selections={}
            for mode in ['query','none','fixed']:
                comp=tuning.components(Ev,T[np.r_[pb,pn]],Ppb,Ppn,Wp,pm,mode)
                for head in ['retained','dense']:selections[head+'_'+mode]=tuning.select(comp,head,np.r_[pb,pn],yv,pb,pn)
            best=(-1.,None);trials=[]
            for reg in readouts.HYCAL_LAMBDAS:
                co=rv.compact(hycal(reg,'tune',pb,pn)[tm],np.r_[pb,pn],yv,pb,pn)
                offsets=np.arange(-1,1.001,.05);hs=tuning.tuning_scores(co,offsets);j=int(hs.argmax())
                trials.append(dict(regularization=reg,harmonics=hs.tolist(),offsets=offsets.tolist()))
                if hs[j]>best[0]+1e-12:best=(float(hs[j]),dict(regularization=reg,offset=float(offsets[j]),pseudo_harmonic=float(hs[j])))
            selection=dict(encoder='siglip_b16',dataset='imagenetr',part=part,ridge_lambda=lam,pseudo_ridge_lambda=plam,
                           full_moments=fm,pseudo_moments=pm,selections=selections,hycal=dict(selected=best[1],trials=trials),
                           support_indices=core.support_rows(ys,new,5,7).tolist(),
                           selection_uses='base-only tuning; reference is this encoder learned/query readout')
            atomic_json(selection,sp)
        mc=np.isin(d['cal_y'],base);queries={'tune':Ev,'cal':d['E_cal'][mc].to(device),'test':d['E_test'].to(device)}
        labels={'tune':yv,'cal':d['cal_y'][mc],'test':d['test_y']};w=risk.weights(labels['cal'],base)
        reference_a=selection['selections']['retained_query']['selected']['a']
        def reference(E,y):
            s=core.z(E@T[base].T)+reference_a*core.z(core.linear(E,W))
            return base[s.argmax(1).cpu().numpy()]==y
        refc=reference(queries['cal'],labels['cal']);reft=reference(queries['test'],labels['test'])
        closed=float(np.mean([reft[labels['test']==c].mean() for c in base]));si=np.asarray(selection['support_indices'])
        for order in [0,1]:
            ordered=np.random.default_rng(1000+part+100*order).permutation(new)
            for session in range(1,6):
                check_pause();dest=po/f'o{order}_t{session}.json'
                if dest.exists():continue
                novel=ordered[:len(new)*session//5];ids=np.r_[base,novel];Pn=core.means(Es[si],ys[si],novel)
                systems={};archive=dict(base_classes=base,novel_classes=novel,cal_labels=labels['cal'],test_labels=labels['test'],cal_reference_correct=refc,test_reference_correct=reft);records=[]
                for mode in ['query','none','fixed']:
                    for pool in ['tune','cal','test']:
                        comp=tuning.components(queries[pool],T[ids],Pb,Pn,W,selection['full_moments'],mode)
                        for head in ['retained','dense']:
                            name=head+'_'+mode;par=selection['selections'][name]['selected']
                            systems[pool,name]=rv.compact(tuning.scores(comp,head,par['a'],par['u']),ids,labels[pool],base,novel)
                hp=selection['hycal']['selected']
                for pool,mask in [('tune',tm),('cal',mc),('test',slice(None))]:
                    systems[pool,'hycal']=rv.compact(hycal(hp['regularization'],pool,base,novel)[mask],ids,labels[pool],base,novel)
                for name in list(selection['selections'])+['hycal']:
                    par=hp if name=='hycal' else selection['selections'][name]['selected']
                    tc,cc=systems['test',name],systems['cal',name]
                    for pool in ['cal','test']:archive.update({pool+'_'+name+'_'+k:v for k,v in tuning.pack(systems[pool,name]).items()})
                    candidates=np.unique(np.quantile(systems['tune',name]['margin'],np.linspace(0,1,101)));archive[name+'_candidates']=candidates
                    common=dict(encoder='siglip_b16',dataset='imagenetr',part=part,order=order,session=session,method=name,reference_closed_test=closed)
                    records.append(dict(**common,policy='fixed_gate',offset=par['offset'],fallback=False,**rv.evaluate(tc,par['offset'])))
                    for policy in ['protection','paired_accuracy']:records.append(dict(**common,**fairness.evaluate_policy(cc,refc,w,candidates,tc,reft,closed,policy)))
                    for ratio in [.80,.85,.90,.95,.99]:
                        off,ceiling=rv.threshold(tc,ratio*closed)
                        records.append(dict(**common,policy='matched_test',retention=ratio,feasible=off is not None,ceiling=ceiling,offset=off,**(rv.evaluate(tc,off) if off is not None else {})))
                raw=dest.with_suffix('.npz');atomic_npz(raw,**archive)
                atomic_json(dict(selection_sha256=digest_file(sp),raw_sha256=digest_file(raw),records=records),dest)
        atomic_json(dict(complete=True,selection_sha256=digest_file(sp)),po/'complete.json')
        print('SIGLIP EVALUATION COMPLETE',part,'seconds',round(time.perf_counter()-started,1),flush=True)
    if args.parts==10:atomic_json(dict(complete=True,parts=10,seconds=time.perf_counter()-started),folder/'complete.json')
if __name__=='__main__':main()
