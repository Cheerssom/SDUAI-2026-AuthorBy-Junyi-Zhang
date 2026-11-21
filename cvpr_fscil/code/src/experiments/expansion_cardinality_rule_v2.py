"""Stage-consistent cardinality rule with a catalog-frozen secondary control."""
from pathlib import Path
import argparse,gc,json,sys,time
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
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_cardinality_rule_v2'

@torch.no_grad()
def components(E,T,Pb,Pn,W,moments,eta):
    nb=len(Pb);result={}
    for i,(name,values,reference) in enumerate([
        ('text',E@T.T,E@T[:nb].T),('learned',core.linear(E,W),None),
        ('prototype',E@torch.cat([Pb,Pn]).T,E@Pb.T)]):
        if reference is None:reference=values
        mu=eta*reference.mean(1,keepdim=True)+(1-eta)*moments[i][0]
        sd=(eta*reference.std(1,correction=0,keepdim=True).clamp_min(1e-6)+(1-eta)*moments[i][1]).clamp_min(1e-6)
        result[name]=(values-mu)/sd
    result['nb']=nb;return result

@torch.no_grad()
def run(enc,ds,parts,device):
    started=time.perf_counter();folder=OUT/enc/ds;folder.mkdir(parents=True,exist_ok=True)
    code=[Path(__file__),Path(core.__file__),Path(rv.__file__),Path(tuning.__file__),Path(previous.__file__),Path(fairness.__file__),Path(readouts.__file__),Path(risk.__file__),ROOT/'src/experiments/kbs_paired_accuracy.py']
    config=dict(protocol_sha256=digest_file(ROOT/'experiments/expansion_cardinality_rule_protocol.md'),sources={p.relative_to(ROOT).as_posix():digest_file(p) for p in code})
    mf=folder/'manifest.json'
    if mf.exists():assert json.loads(mf.read_text())==config
    else:atomic_json(config,mf)
    if (folder/'complete.json').exists():return
    d,cache=previous.load_data(ds,enc)
    inputs=[cache,ROOT/f'experiments/kbs_independent_risk/splits/{ds}.json',ROOT/f'experiments/kbs_calibration_budget/pools/{ds}.json',ROOT/f'experiments/kbs_calibration_budget/features/{enc}/{ds}/extra.pt',ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json']
    hashes={p.relative_to(ROOT).as_posix():digest_file(p) for p in inputs};dp=folder/'data_manifest.json'
    if dp.exists():assert json.loads(dp.read_text())==hashes
    else:atomic_json(hashes,dp)
    Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']];yf,ys=d['fit_y'],d['support_y']
    for entry in json.loads(inputs[-1].read_text())[:parts]:
        check_pause();part=entry['seed'];po=folder/f'p{part}';po.mkdir(exist_ok=True)
        if (po/'complete.json').exists():continue
        base,new,pb,pn=[np.asarray(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
        oldp=ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{part}/selection.json';orig=json.loads(oldp.read_text())
        qp=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p{part}/selection.json';old=json.loads(qp.read_text())
        W=core.fit_ridge(Ef,yf,base,orig['ridge_lambda']);Wp=core.fit_ridge(Ef,yf,pb,orig['pseudo_ridge_lambda'])
        Pb=core.means(Ef,yf,base);Ppb=core.means(Ef,yf,pb)
        mask=np.isin(d['tune_y'],base);Ev=d['E_tune'][mask].to(device);yv=d['tune_y'][mask]
        si=core.support_rows(ys,pn,5,7);Ppn=core.means(Es[si],ys[si],pn);ids=np.r_[pb,pn]
        sp=po/'selection.json'
        if sp.exists():selection=json.loads(sp.read_text());assert selection['source_selection_sha256']==digest_file(qp)
        else:
            pm=old['pseudo_moments'];fm=old['full_moments']
            eta_pseudo=min(1.,len(pb)/30.);eta_full=min(1.,len(base)/30.)
            primary=tuning.select(components(Ev,T[ids],Ppb,Ppn,Wp,pm,eta_pseudo),'retained',ids,yv,pb,pn)
            frozen=ROOT/f'experiments/expansion_shrinkage/{enc}/{ds}/p{part}/selection.json'
            frozen_sel=json.loads(frozen.read_text())
            frozen_method='half' if eta_full==.5 else 'query'
            selection=dict(encoder=enc,dataset=ds,part=part,source_selection_sha256=digest_file(qp),
                           original_selection_sha256=digest_file(oldp),full_moments=fm,pseudo_moments=pm,
                           eta_pseudo=eta_pseudo,eta_full=eta_full,design_constant=30,
                           selections=dict(primary=primary),catalog_frozen_method=frozen_method,
                           frozen_selection_sha256=digest_file(frozen),settings=3636,
                           development_scope='Rule constant proposed after previous fixed-eta results; no prior-optimality claim')
            atomic_json(selection,sp)
        maskc=np.isin(d['cal_y'],base);queries={'tune':Ev,'cal':d['E_cal'][maskc].to(device),'test':d['E_test'].to(device)}
        labels={'tune':yv,'cal':d['cal_y'][maskc],'test':d['test_y']};w=risk.weights(labels['cal'],base);si=np.array(orig['support_indices'])
        for order in [0,1]:
            ordered=np.random.default_rng(1000+part+100*order).permutation(new)
            for session in range(1,6):
                check_pause();dest=po/f'o{order}_t{session}.json'
                if dest.exists():continue
                source=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p{part}/o{order}_t{session}.json';oldrows=json.loads(source.read_text())
                rawsource=source.with_suffix('.npz')
                with np.load(rawsource) as z:refc=z['cal_reference_correct'];reft=z['test_reference_correct']
                novel=ordered[:len(new)*session//5];ids=np.r_[base,novel];Pn=core.means(Es[si],ys[si],novel)
                param=selection['selections']['primary']['selected'];systems={}
                archive=dict(base_classes=base,novel_classes=novel,cal_labels=labels['cal'],test_labels=labels['test'],cal_reference_correct=refc,test_reference_correct=reft)
                for pool in ['tune','cal','test']:
                    comp=components(queries[pool],T[ids],Pb,Pn,W,selection['full_moments'],selection['eta_full'])
                    c=rv.compact(tuning.scores(comp,'retained',param['a'],param['u']),ids,labels[pool],base,novel);systems[pool]=c
                    if pool!='tune':archive.update({pool+'_primary_'+k:v for k,v in tuning.pack(c).items()})
                tc,cc=systems['test'],systems['cal'];candidates=np.unique(np.quantile(systems['tune']['margin'],np.linspace(0,1,101)));archive['primary_candidates']=candidates
                closed=float(np.mean([reft[labels['test']==c].mean() for c in base]))
                common=dict(encoder=enc,dataset=ds,part=part,order=order,session=session,reference_closed_test=closed)
                records=[dict(**common,method='primary',policy='fixed_gate',offset=param['offset'],fallback=False,**rv.evaluate(tc,param['offset']))]
                for policy in ['protection','paired_accuracy']:
                    records.append(dict(**common,method='primary',**fairness.evaluate_policy(cc,refc,w,candidates,tc,reft,closed,policy)))
                for ratio in [.80,.85,.90,.95,.99]:
                    off,ceiling=rv.threshold(tc,ratio*closed)
                    records.append(dict(**common,method='primary',policy='matched_test',retention=ratio,feasible=off is not None,ceiling=ceiling,offset=off,**(rv.evaluate(tc,off) if off is not None else {})))
                for mode in ['query','none','fixed']:
                    records += [dict(r,method=mode) for r in oldrows['records'] if r['method']=='retained_'+mode]
                records += [dict(r,method='dense') for r in oldrows['records'] if r['method']=='dense_query']
                frozen=ROOT/f'experiments/expansion_shrinkage/{enc}/{ds}/p{part}/o{order}_t{session}.json'
                frozen_rows=json.loads(frozen.read_text())['records']
                records += [dict(r,method='catalog_frozen',chosen_scheme=selection['catalog_frozen_method']) for r in frozen_rows if r['method']==selection['catalog_frozen_method']]
                raw=dest.with_suffix('.npz');atomic_npz(raw,**archive)
                atomic_json(dict(selection_sha256=digest_file(sp),raw_sha256=digest_file(raw),
                                 frozen_source=source.relative_to(ROOT).as_posix(),frozen_source_sha256=digest_file(source),
                                 frozen_raw_sha256=digest_file(rawsource),catalog_frozen_source=frozen.relative_to(ROOT).as_posix(),catalog_frozen_sha256=digest_file(frozen),records=records),dest)
        atomic_json(dict(complete=True,selection_sha256=digest_file(sp)),po/'complete.json')
        print('CARDINALITY RULE COMPLETE',enc,ds,part,'seconds',round(time.perf_counter()-started,1),flush=True)
    if parts==10:atomic_json(dict(complete=True,parts=10,seconds=time.perf_counter()-started),folder/'complete.json')
    del d,Ef,Es,T;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--dataset',choices=['pets','cifar','imagenetr']);ap.add_argument('--parts',type=int,default=10);args=ap.parse_args()
    device=configure();check_pause()
    for enc in ([args.encoder] if args.encoder else ['l14','b32']):
        for ds in ([args.dataset] if args.dataset else ['pets','cifar','imagenetr']):run(enc,ds,args.parts,device)
    if args.parts==10 and not args.encoder and not args.dataset:atomic_json(dict(complete=True,conditions=6),OUT/'complete.json')
if __name__=='__main__':main()
