"""Author-requested matched-base, learned-gate and calibration experiments."""
from pathlib import Path
import argparse, json, sys, time, gc, math, warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_reviewer_revision as previous
import kbs_calibration_budget as budget
import kbs_calibration_fairness as fairness
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,digest_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/expansion_gate_calibration'
PROTOCOL=ROOT/'experiments/expansion_gate_calibration_protocol.md'
ENCODERS=['l14','b32'];DATASETS=['cifar','pets','imagenetr']
GATES=['logistic','mlp'];CAPS=[2,5,10,20,40]
FEATURES=['aligned_margin','uncentered_margin','text_gap','prototype_gap','base_winner','novel_winner',
          'learned_max_z','learned_log_sigma','prototype_mean','prototype_log_sigma','normalized_entropy','base_top_gap']

def loadz(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

def source_manifest():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),PROTOCOL,Path(core.__file__),Path(rv.__file__),Path(previous.__file__),
           Path(budget.__file__),Path(fairness.__file__),Path(paired.__file__),Path(risk.__file__)]
    manifest={'sources':{p.relative_to(ROOT).as_posix():digest_file(p) for p in files},'protocol':'four-block-v1'}
    path=OUT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Sources changed; use a new output root.'
    else:atomic_json(manifest,path)

def choose(c,d):return (d>c['margin'])|((d==c['margin'])&(c['novel_prediction']<c['base_prediction']))

def cm(z,method,labels,base,new,prefix=''):
    return budget.compact(z,method,labels,base,new,prefix)

def operating(c,target):
    d,ceiling=rv.threshold(c,target)
    if d is None:return dict(feasible=False,ceiling=ceiling)
    v=rv.evaluate(c,d)
    return dict(feasible=True,base=v['old'],novel=v['new'],overshoot=v['old']-target)

def accuracy_stats(c,ref,y,base,d):
    b=c['base_mask'];rc=np.asarray(ref)[b]
    cc=(rc if d is None else (c['base_correct']&~choose(c,d))[b])
    labels=y[b];B=len(base);ar=[];ac=[];gr=[];vr=[];nc=[]
    for cl in base:
        m=labels==cl;r=rc[m];v=cc[m];a=float(r.mean());q=float(v.mean());g=float((~r&v).mean())
        actual=float(np.var(v.astype(float)-.95*r))
        predicted=q+.95**2*a-2*.95*(q-g)-(q-.95*a)**2
        assert abs(actual-predicted)<1e-12
        ar.append(a);ac.append(q);gr.append(g);vr.append(actual);nc.append(len(r))
    return dict(base=float(np.mean(ac)),reference=float(np.mean(ar)),recovered=float(np.mean(gr)),
                destroyed=float(np.mean(ar)-np.mean(ac)+np.mean(gr)),
                paired_variance=float(np.mean(vr)),calibration_mean_variance=float(np.sum(np.array(vr)/np.array(nc))/B**2),
                class_reference=ar,class_candidate=ac,class_recovered=gr,class_counts=nc)

def match_archives(parts=10):
    """All targets are empirical diagnostics, never threshold-selection inputs."""
    methods=['retained_query','retained_none','retained_fixed','dense_query','dense_none','dense_fixed']
    for enc in ENCODERS:
        for ds in DATASETS:
            for p in range(parts):
                dest=OUT/'matched'/enc/ds/f'p{p}.json'
                if dest.exists():continue
                rows=[];inputs={}
                for order in [0,1]:
                    for session in range(1,6):
                        check_pause();folder=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p{p}'
                        jp=folder/f'o{order}_t{session}.json';zp=jp.with_suffix('.npz');a=loadz(zp)
                        inputs[jp.relative_to(ROOT).as_posix()]=digest_file(jp);inputs[zp.relative_to(ROOT).as_posix()]=digest_file(zp)
                        records=json.loads(jp.read_text())['records'];pol={r['method']:r for r in records if r['policy']=='paired_accuracy'}
                        y=a['test_labels'];base=a['base_classes'];new=a['novel_classes'];ref=a['test_reference_correct']
                        cs={m:cm(a,m,y,base,new,'test_') for m in methods}
                        closed=float(np.mean([ref[y==cl].mean() for cl in base]))
                        for target_name,target in [('dense_realized',pol['dense_query']['old']),('retained_realized',pol['retained_query']['old']),('floor95',.95*closed),('floor99',.99*closed)]:
                            values={m:operating(cs[m],target) for m in methods}
                            for m in methods:
                                policy=pol[m];stats=accuracy_stats(cs[m],ref,y,base,policy['offset'])
                                rows.append(dict(encoder=enc,dataset=ds,part=p,order=order,session=session,target_kind=target_name,target=target,
                                    method=m,selected_base=policy['old'],selected_novel=policy['new'],finite=not policy['fallback'],
                                    test_met=policy['test_target_met'],frontier=values[m],error_structure=stats))
                atomic_json(dict(inputs=inputs,records=rows),dest)
                print('MATCHED',enc,ds,p,flush=True)

@torch.no_grad()
def features(E,T,Pb,Pn,W,params,ids,y,base,new):
    nb=len(Pb);a,u,_=params
    qt=E@T.T;ql=core.linear(E,W);qp=E@torch.cat([Pb,Pn]).T
    mt=qt[:,:nb].mean(1,keepdim=True);st=qt[:,:nb].std(1,correction=0,keepdim=True).clamp_min(1e-6)
    ml=ql.mean(1,keepdim=True);sl=ql.std(1,correction=0,keepdim=True).clamp_min(1e-6)
    mp=qp[:,:nb].mean(1,keepdim=True);sp=qp[:,:nb].std(1,correction=0,keepdim=True).clamp_min(1e-6)
    zt=(qt-mt)/st;zl=(ql-ml)/sl;zp=(qp-mp)/sp
    sb=zt[:,:nb]+a*zl;sn=zt[:,nb:]+u*zp[:,nb:]
    score=torch.cat([sb,sn],1);c=rv.compact(score,ids,y,base,new)
    bmax=sb.max(1).values;nmax=sn.max(1).values
    rawb=(qt[:,:nb]/st+a*ql/sl).max(1).values
    rawn=(qt[:,nb:]/st+u*qp[:,nb:]/sp).max(1).values
    prob=torch.softmax(zl,1);ent=-(prob*torch.log(prob.clamp_min(1e-12))).sum(1)/math.log(nb)
    top=sb.topk(2,dim=1).values
    X=torch.stack([nmax-bmax,rawn-rawb,zt[:,nb:].max(1).values-zt[:,:nb].max(1).values,
        zp[:,nb:].max(1).values-zp[:,:nb].max(1).values,bmax,nmax,zl.max(1).values,
        torch.log(sl[:,0]),mp[:,0],torch.log(sp[:,0]),ent,top[:,0]-top[:,1]],1).cpu().numpy().astype(np.float64)
    assert np.isfinite(X).all()
    return X,c

def macro_weights(y):
    values,idx,cnt=np.unique(y,return_inverse=True,return_counts=True)
    return 1/(len(values)*cnt[idx])

def group_h(c,margin):
    cc=dict(c,margin=margin);cand=np.quantile(margin,np.linspace(0,1,101))
    vals=[rv.evaluate(cc,float(d))['harmonic'] for d in cand];j=int(np.argmax(vals))
    return float(vals[j]),float(cand[j])

def subcompact(c,mask,y,base,new):
    out={k:(v[mask] if isinstance(v,np.ndarray) else v) for k,v in c.items()}
    out['weights']=np.zeros(mask.sum())
    yy=y[mask]
    for group in [base,new]:
        for cl in group:
            ix=yy==cl
            if ix.any():out['weights'][ix]=1/(len(group)*ix.sum())
    return out

def fit_model(kind,X,labels,weight,reg,seed):
    if kind=='logistic':
        model=LogisticRegression(C=reg,max_iter=2000,solver='lbfgs',tol=1e-8)
        model.fit(X,labels,sample_weight=weight*len(weight))
        return dict(kind=kind,coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]))
    torch.manual_seed(seed);model=torch.nn.Sequential(torch.nn.Linear(X.shape[1],16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    xx=torch.tensor(X,dtype=torch.float32);yy=torch.tensor(labels,dtype=torch.float32);ww=torch.tensor(weight,dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    for step in range(250):
        opt.zero_grad();logit=model(xx).flatten();loss=(torch.nn.functional.binary_cross_entropy_with_logits(logit,yy,reduction='none')*ww).sum()
        penalty=sum(v.square().sum() for n,v in model.named_parameters() if 'weight' in n)
        (loss+reg*penalty).backward();opt.step()
    return dict(kind=kind,state={k:v.detach().numpy().tolist() for k,v in model.state_dict().items()})

def predict_model(model,X):
    if model['kind']=='logistic':return X@np.array(model['coef'])+model['intercept']
    z=model['state'];h=np.tanh(X@np.array(z['0.weight']).T+np.array(z['0.bias']))
    return (h@np.array(z['2.weight']).T+np.array(z['2.bias'])).ravel()

def select_gate(kind,X,c,y,base,new,seed):
    rng=np.random.default_rng(281000+seed);train=np.zeros(len(y),bool)
    for cl in np.unique(y):
        ix=rng.permutation(np.flatnonzero(y==cl));train[ix[:len(ix)//2]]=True
    valid=~train;sc=StandardScaler().fit(X[train]);xx=sc.transform(X);labels=np.isin(y,new).astype(int)
    vc=subcompact(c,valid,y,base,new);trials=[]
    grid=[.01,.1,1.,10.] if kind=='logistic' else [.0001,.001,.01,.1]
    for reg in grid:
        model=fit_model(kind,xx[train],labels[train],macro_weights(y[train]),reg,seed)
        h,d=group_h(vc,-predict_model(model,xx[valid]));trials.append(dict(reg=reg,harmonic=h,offset=d))
    best=max(range(len(trials)),key=lambda j:trials[j]['harmonic']);chosen=trials[best]
    full=StandardScaler().fit(X);model=fit_model(kind,full.transform(X),labels,macro_weights(y),chosen['reg'],seed)
    trainmargin=-predict_model(model,full.transform(X));h,offset=group_h(c,trainmargin)
    return dict(model=model,mean=full.mean_.tolist(),scale=full.scale_.tolist(),trials=trials,selected=chosen,
                refit_offset=offset,refit_harmonic=h,features=FEATURES,training_rows=int(train.sum()),validation_rows=int(valid.sum()),
                training_ids_sha256=core.digest(np.flatnonzero(train)),tuning_labels='base-only pseudo episode')

def apply_gate(state,X):
    return -predict_model(state['model'],(X-np.array(state['mean']))/np.array(state['scale']))

def pack(c):return {k:c[k] for k in ['margin','base_prediction','novel_prediction']}

def train_gates(parts,device):
    for enc in ENCODERS:
        for ds in DATASETS:
            folder=OUT/'gates'/enc/ds
            if parts==10 and (folder/'complete.json').exists():continue
            d,cache=previous.load_data(ds,enc)
            partitions=ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json'
            entries=json.loads(partitions.read_text())[:parts]
            inputs={str(p.relative_to(ROOT)):digest_file(p) for p in [cache,partitions,
                ROOT/f'experiments/kbs_independent_risk/splits/{ds}.json',ROOT/f'experiments/kbs_calibration_budget/pools/{ds}.json',
                ROOT/f'experiments/kbs_calibration_budget/features/{enc}/{ds}/extra.pt']}
            folder.mkdir(parents=True,exist_ok=True);mf=folder/'inputs.json'
            if mf.exists():assert json.loads(mf.read_text())==inputs
            else:atomic_json(inputs,mf)
            Ef=d['E_fit'].to(device);Es=d['E_support'].to(device);T=d['T'].to(device);yf=d['fit_y'];ys=d['support_y']
            for entry in entries:
                check_pause();p=entry['seed'];po=folder/f'p{p}';po.mkdir(exist_ok=True)
                if (po/'complete.json').exists():continue
                original=ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{p}'
                selp=original/'selection.json';sel=json.loads(selp.read_text());params=sel['parameters']['retained']
                base,new,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
                W=core.fit_ridge(Ef,yf,base,sel['ridge_lambda']);Pb=core.means(Ef,yf,base)
                tunemask=np.isin(d['tune_y'],base);Ev=d['E_tune'][tunemask].to(device);yv=d['tune_y'][tunemask]
                sp=po/'selection.json'
                if sp.exists():selection=json.loads(sp.read_text());assert selection['original_sha256']==digest_file(selp)
                else:
                    Wp=core.fit_ridge(Ef,yf,pb,sel['pseudo_ridge_lambda']);Ppb=core.means(Ef,yf,pb)
                    si=core.support_rows(ys,pn,5,7);Ppn=core.means(Es[si],ys[si],pn)
                    X,c=features(Ev,T[np.r_[pb,pn]],Ppb,Ppn,Wp,params,np.r_[pb,pn],yv,pb,pn)
                    # CPU fits retain gradients even though frozen features have none.
                    states={kind:select_gate(kind,X,c,yv,pb,pn,72000+p) for kind in GATES}
                    selection=dict(original_sha256=digest_file(selp),gates=states,parameters=params,
                                   input_sha256=inputs,fit_before_calibration=True)
                    atomic_json(selection,sp)
                calmask=np.isin(d['cal_y'],base);Ec=d['E_cal'][calmask].to(device);yc=d['cal_y'][calmask]
                Et=d['E_test'].to(device);yt=d['test_y'];si=np.array(sel['support_indices'])
                for order in [0,1]:
                    ordered=np.random.default_rng(1000+p+100*order).permutation(new)
                    for session in range(1,6):
                        check_pause();dest=po/f'o{order}_t{session}.json'
                        if dest.exists():continue
                        novel=ordered[:len(new)*session//5];ids=np.r_[base,novel];Pn=core.means(Es[si],ys[si],novel)
                        cp=ROOT/f'experiments/kbs_calibration_budget/{enc}/{ds}/p{p}/o{order}_t{session}.npz'
                        tp=original/f'o{order}_t{session}.npz';cal=loadz(cp);test=loadz(tp)
                        assert np.array_equal(yc,cal['labels']) and np.array_equal(yt,test['test_labels'])
                        ref=cal['reference_correct'];tref=test['test_reference_correct'];closed=float(np.mean([tref[yt==cl].mean() for cl in base]))
                        x={};cs={};witness={}
                        for pool,E,y in [('tune',Ev,yv),('cal',Ec,yc),('test',Et,yt)]:
                            xx,cc=features(E,T[ids],Pb,Pn,W,params,ids,y,base,novel);x[pool]=xx
                            if pool!='tune':
                                legacy=cm(cal if pool=='cal' else test,'retained',y,base,novel,'' if pool=='cal' else 'test_')
                                err=float(np.max(abs(legacy['margin']-cc['margin'])));assert err<5e-4,(enc,ds,p,pool,err)
                                for key in ['base_prediction','novel_prediction']:assert np.array_equal(legacy[key],cc[key]),(p,pool,key)
                                witness[pool]=err;cc=legacy
                            cs[pool]=cc
                        archive=dict(base_classes=base,novel_classes=novel,cal_labels=yc,test_labels=yt,
                                     cal_reference_correct=ref,test_reference_correct=tref);records=[]
                        wc=risk.weights(yc,base)
                        for kind in GATES:
                            gc={pool:dict(cs[pool],margin=apply_gate(selection['gates'][kind],x[pool])) for pool in x}
                            candidates=np.unique(np.quantile(gc['tune']['margin'],np.linspace(0,1,101)))
                            archive[kind+'_candidates']=candidates
                            for pool in ['cal','test']:
                                archive.update({pool+'_'+kind+'_'+k:v for k,v in pack(gc[pool]).items()})
                            common=dict(encoder=enc,dataset=ds,part=p,order=order,session=session,method=kind)
                            records.append(dict(**common,policy='fixed_gate',**rv.evaluate(gc['test'],selection['gates'][kind]['refit_offset'])))
                            for policy in ['protection','paired_accuracy']:
                                result=fairness.evaluate_policy(gc['cal'],ref,wc,candidates,gc['test'],tref,closed,policy)
                                assert result['recovered_test']<1e-12
                                records.append(dict(**common,**result))
                            for ratio in [.95,.99]:
                                records.append(dict(**common,policy='matched_test',retention=ratio,**operating(gc['test'],ratio*closed)))
                        raw=dest.with_suffix('.npz');atomic_npz(raw,**archive)
                        atomic_json(dict(selection_sha256=digest_file(sp),raw_sha256=digest_file(raw),
                            original_archive_sha256=digest_file(tp),calibration_archive_sha256=digest_file(cp),records=records,witness=witness),dest)
                atomic_json(dict(complete=True,selection_sha256=digest_file(sp)),po/'complete.json')
                print('GATES',enc,ds,p,flush=True)
            if parts==10:atomic_json(dict(complete=True,parts=10),folder/'complete.json')
            del d,Ef,Es,T;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()

def fast_direct(c,reference,w,candidates):
    """Batch exact fixed-sample p-values from counts, then fixed-sequence stop."""
    weights=w/w.max();levels=np.unique(weights);counts=[]
    tie=c['novel_prediction']<c['base_prediction']
    events=np.where(tie,c['margin'],np.nextafter(c['margin'],np.inf))
    for lev in levels:
        mask=weights==lev;r=reference[mask];bc=c['base_correct'][mask];ev=events[mask]
        ks=np.sort(ev[r&bc]);gs=np.sort(ev[~r&bc])
        k=len(ks)-np.searchsorted(ks,candidates,side='right');g=len(gs)-np.searchsorted(gs,candidates,side='right');h=int(r.sum())-k
        counts.append((h,g,k))
    h,g,k=[np.stack([r[i] for r in counts],1) for i in range(3)]
    a=paired.LAMBDAS[:,None]*levels[None]
    logs=h@np.log1p(-.95*a).T+g@np.log1p(a).T+k@np.log1p(.05*a).T
    loge=np.logaddexp.reduce(logs,axis=1)-math.log(len(paired.LAMBDAS))
    pv=np.exp(-np.maximum(0,loge));bad=np.flatnonzero(pv>.01);end=int(bad[0]) if len(bad) else len(candidates)
    return (None if end==0 else float(candidates[end-1])),pv

def stability(parts):
    methods=['retained','anchored_tp','raw','fixed_temperature','logistic','mlp']
    for enc in ENCODERS:
        for ds in DATASETS:
            for p in range(parts):
                check_pause();dest=OUT/'stability'/enc/ds/f'p{p}.json'
                if dest.exists():continue
                cp=ROOT/f'experiments/kbs_calibration_budget/{enc}/{ds}/p{p}/o0_t5.npz'
                tp=ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{p}/o0_t5.npz'
                gp=OUT/f'gates/{enc}/{ds}/p{p}/o0_t5.npz';cal=loadz(cp);test=loadz(tp);gate=loadz(gp)
                y=cal['labels'];yt=test['test_labels'];base=cal['base_classes'];new=cal['novel_classes'];ref=cal['reference_correct'];tref=test['test_reference_correct']
                assert np.array_equal(y,gate['cal_labels']);closed=float(np.mean([tref[yt==cl].mean() for cl in base]))
                cc={};tc={};cand={}
                for m in methods:
                    learned=m in GATES
                    cc[m]=cm(gate if learned else cal,m,y,base,new,'cal_' if learned else '')
                    tc[m]=cm(gate if learned else test,m,yt,base,new,'test_')
                    cand[m]=(gate if learned else test)[m+'_candidates']
                rows=[]
                # Class permutations shared by all methods, nested across caps.
                indices={}
                for seed in range(100,130):
                    rng=np.random.default_rng(771000+seed)
                    indices[seed]={int(cl):rng.permutation(np.flatnonzero(y==cl)) for cl in base}
                for cap in CAPS:
                    for seed in ([100] if cap==40 else range(100,130)):
                        take=np.sort(np.concatenate([ix[:min(cap,len(ix))] for ix in indices[seed].values()]));yy=y[take];w=risk.weights(yy,base);rr=ref[take]
                        for m in methods:
                            sub={k:(v[take] if isinstance(v,np.ndarray) else v) for k,v in cc[m].items()};sub['weights']=w
                            offset,pv=fast_direct(sub,rr,w,cand[m])
                            # Independent scalar reconstruction for every subset at first candidate and selected boundary.
                            for j in set([0,max(0,int(np.searchsorted(cand[m],offset)) if offset is not None else 0)]):
                                corr=sub['base_correct']&~choose(sub,float(cand[m][j]))
                                pp,_=paired.pvalue(rr,corr,w);assert abs(pp-pv[j])<2e-11
                            for policy,d in [('paired_accuracy',offset),('protection',risk.certify(sub,rr,w,cand[m])['offset'])]:
                                result=dict(old=closed,new=0.) if d is None else rv.evaluate(tc[m],d)
                                rows.append(dict(encoder=enc,dataset=ds,part=p,cap=cap,subset_seed=int(seed),method=m,policy=policy,
                                    labels=len(take),finite=d is not None,offset=d,base=result['old'],novel=result['new'],
                                    test_met=result['old']>=.95*closed-1e-10,target_gap_pp=100*(result['old']-.95*closed),
                                    subset_sha256=core.digest(take)))
                atomic_json(dict(inputs={str(q.relative_to(ROOT)):digest_file(q) for q in [cp,tp,gp]},records=rows),dest)
                print('STABILITY',enc,ds,p,flush=True)

def simulations(parts):
    for enc in ENCODERS:
        for ds in DATASETS:
            check_pause();dest=OUT/'coupling'/f'{enc}_{ds}.json'
            if dest.exists():continue
            class_rates=[]
            for p in range(parts):
                z=loadz(ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{p}/o0_t5.npz')
                class_rates += [float(z['test_reference_correct'][z['test_labels']==c].mean()) for c in z['base_classes']]
            # Fix B strata with the empirical reference-quality distribution; no test-driven method selection.
            B=len(z['base_classes']);rates=np.array(class_rates).reshape(parts,B).mean(0)
            candidate=.975*rates;limit=np.minimum(1-rates,candidate);rows=[];rng=np.random.default_rng(28102026+ENCODERS.index(enc)*10+DATASETS.index(ds))
            for cap in [5,10,20,40]:
                for frac in [0.,.25,.5,.75,1.]:
                    g=frac*limit;both=candidate-g;destroy=rates-candidate+g;neither=1-rates-g
                    probs=np.stack([neither,destroy,g,both],1);assert np.all(probs>=-1e-12) and np.allclose(probs.sum(1),1)
                    # Exact stratified-mean variance and independent Monte Carlo power.
                    means=candidate-.95*rates
                    var=candidate+.95**2*rates-2*.95*(candidate-g)-means**2
                    count=np.stack([rng.multinomial(cap,q,size=500) for q in probs],axis=1)
                    # category [00,10,01,11], equal per-class cap => equal weights.
                    h=count[:,:,1].sum(1);gg=count[:,:,2].sum(1);k=count[:,:,3].sum(1)
                    logs=h[:,None]*np.log1p(-.95*paired.LAMBDAS)+gg[:,None]*np.log1p(paired.LAMBDAS)+k[:,None]*np.log1p(.05*paired.LAMBDAS)
                    le=np.logaddexp.reduce(logs,axis=1)-math.log(len(paired.LAMBDAS));passed=le>=math.log(100)
                    pw=float(passed.mean());emp=(count[:,:,2]+.05*count[:,:,3]-.95*count[:,:,1]).sum(1)/(B*cap)
                    rows.append(dict(encoder=enc,dataset=ds,cap=cap,recovery_fraction=frac,draws=500,
                        reference_accuracy=float(rates.mean()),candidate_accuracy=float(candidate.mean()),recovered=float(g.mean()),
                        exact_mean_variance=float(var.sum()/cap/B**2),empirical_mean_variance=float(emp.var(ddof=1)),
                        acceptance=pw,mc_standard_error=math.sqrt(pw*(1-pw)/500),positive_surplus=float(means.mean())))
            atomic_json(dict(scope='controlled fixed-correctness coupling simulation',reference_class_rates=rates.tolist(),records=rows),dest)
            print('COUPLING',enc,ds,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','matched','gates','stability','coupling'],default='all');ap.add_argument('--parts',type=int,default=10);args=ap.parse_args()
    source_manifest();start=time.perf_counter()
    if args.stage in ['all','matched']:match_archives(args.parts)
    if args.stage in ['all','gates']:
        device=configure();train_gates(args.parts,device)
    if args.stage in ['all','stability']:stability(args.parts)
    if args.stage in ['all','coupling']:simulations(args.parts)
    atomic_json(dict(stage=args.stage,parts=args.parts,seconds=time.perf_counter()-start,complete=True),OUT/f'complete_{args.stage}.json')

if __name__=='__main__':main()
