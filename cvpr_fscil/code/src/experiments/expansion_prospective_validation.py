"""Locked four-block evidence extension; atomic frozen-feature checkpoints."""
from pathlib import Path
import argparse,json,sys,gc,time,math,hashlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_full_fusion_retuning as tune
import kbs_reviewer_revision as previous
import kbs_reviewer_readouts as readouts
import kbs_calibration_fairness as fairness
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
import expansion_gate_calibration as gates
from kbs_runtime import configure,check_pause,atomic_json,atomic_npz,atomic_torch,digest_file
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/expansion_prospective_validation'
PROTOCOL=ROOT/'experiments/expansion_prospective_validation_protocol.md'
MECHANISMS=['full','pm_only','scale_only','minus_learned_scale','minus_text_scale','shuffle3101','shuffle3102','shuffle3103']

def freeze():
    files=[Path(__file__),PROTOCOL,*[Path(m.__file__) for m in [core,rv,tune,previous,readouts,fairness,paired,risk,gates]]]
    manifest=dict(protocol='prospective-four-blocks-v1',sources={p.relative_to(ROOT).as_posix():digest_file(p) for p in files})
    p=OUT/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Frozen code changed; preserve provenance before resuming.'
    else:atomic_json(manifest,p)

def hashids(y):return hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()

@torch.no_grad()
def raw(E,T,Pb,Pn,W):
    b=len(Pb);t=E@T.T;l=core.linear(E,W);p=E@torch.cat([Pb,Pn]).T
    values=[t,l,p];mom=[(v[:,:b].mean(1,keepdim=True),v[:,:b].std(1,correction=0,keepdim=True).clamp_min(1e-6)) for v in values]
    return dict(values=values,mom=mom,nb=b)

def comps(r,fixed,method,C=30,pool_seed=0):
    b=r['nb'];eta=min(1.,b/C);out={};used=[]
    perm=None
    if method.startswith('shuffle'):
        seed=int(method[7:]);perm=np.random.default_rng(seed+pool_seed).permutation(len(r['values'][0]))
        perm=torch.as_tensor(perm,device=r['values'][0].device)
    for i,(name,v) in enumerate(zip(['text','learned','prototype'],r['values'])):
        mu,sd=r['mom'][i]
        if perm is not None:mu,sd=mu[perm],sd[perm]
        mu=eta*mu+(1-eta)*fixed[i][0];sd=eta*sd+(1-eta)*fixed[i][1]
        if method=='none':mu=0.
        if method=='fixed':mu,sd=fixed[i]
        if method=='pm_only':
            sd=fixed[i][1]
            if i!=2:mu=fixed[i][0]
        if method=='scale_only' and i==2:mu=fixed[i][0]
        if method=='minus_learned_scale' and i==1:sd=fixed[i][1]
        if method=='minus_text_scale' and i==0:sd=fixed[i][1]
        sd=sd.clamp_min(1e-6) if torch.is_tensor(sd) else max(sd,1e-6)
        out[name]=(v-mu)/sd;used.append((mu,sd))
    out['nb']=b;out['used']=used;return out

def score(c,method,par):return tune.scores(c,'dense' if method=='dense' else 'retained',par['a'],par['u'])

def makecompact(r,fm,method,par,ids,y,b,n,pool_seed=0):
    C=float(method[1:]) if method.startswith('C') else 30
    comp=comps(r,fm,'full' if method in ['dense'] or method.startswith('C') else method,C,pool_seed)
    return rv.compact(score(comp,method,par),ids,y,b,n)

def gate_features(r,comp,par,c):
    b=r['nb'];a,u=par['a'],par['u'];qt,ql,qp=r['values'];zt,zl,zp=[comp[k] for k in ['text','learned','prototype']]
    sb=zt[:,:b]+a*zl;sn=zt[:,b:]+u*zp[:,b:];bmax=sb.max(1).values;nmax=sn.max(1).values
    st,sl,sp=[z[1] for z in comp['used']];mp=comp['used'][2][0]
    prob=torch.softmax(zl,1);ent=-(prob*prob.clamp_min(1e-12).log()).sum(1)/math.log(b);top=sb.topk(2,dim=1).values
    expand=lambda x:torch.as_tensor(x,device=qt.device).expand(len(qt),1)[:,0]
    un=(qt[:,b:]/st+u*qp[:,b:]/sp).max(1).values-(qt[:,:b]/st+a*ql/sl).max(1).values
    X=torch.stack([nmax-bmax,un,zt[:,b:].max(1).values-zt[:,:b].max(1).values,zp[:,b:].max(1).values-zp[:,:b].max(1).values,bmax,nmax,zl.max(1).values,expand(sl).log(),expand(mp),expand(sp).log(),ent,top[:,0]-top[:,1]],1)
    return X.detach().cpu().numpy().astype(np.float64)

def auc(c):
    m=c['base_mask']|c['novel_mask'];y=c['novel_mask'][m];w=c['weights'][m].copy()
    return float(roc_auc_score(y,-c['margin'][m],sample_weight=w))

def frontier(c,target):
    off,ceiling=rv.threshold(c,target)
    return dict(feasible=off is not None,ceiling=ceiling,offset=off,**(rv.evaluate(c,off) if off is not None else {}))

def load_data(ds,enc):
    if ds in ['food','sun']:
        p=OUT/'external'/ds/enc/'frozen.pt';return torch.load(p,map_location='cpu',weights_only=False),p
    return previous.load_data(ds,enc)

def stratified_classes(K,seed,B,N,counts):
    rng=np.random.default_rng(seed);rank=np.argsort(-counts,kind='stable');pools=[rng.permutation(v) for v in np.array_split(rank,3)]
    # Round-robin interleaving preserves broad frequency representation at each B.
    seq=np.array([v[j] for j in range(max(map(len,pools))) for v in pools if j<len(v)])
    new=np.sort(seq[-N:]);base=np.sort(seq[:B]);return base,new

def assignments(ds,d,block):
    K=len(d['T']);counts=np.bincount(d['fit_y'],minlength=K)
    if block=='mechanism':
        return json.loads((ROOT/f'experiments/kbs_revision/l14/{ds}/partitions.json').read_text())
    rows=[]
    for part in range(10):
        sizes=([5,10,15,20,30,50] if ds=='cifar' else [5,10,15,20,30,50,100]) if block=='cardinality' else [50 if ds=='food' else 195]
        N=(20 if ds=='cifar' else 50) if block=='cardinality' else (50 if ds=='food' else 195)
        for B in sizes:
            base,new=stratified_classes(K,28102026+part,B,N,counts)
            pb=np.sort(np.random.default_rng(6000+part+B).permutation(base)[:B//2]);pn=np.setdiff1d(base,pb)
            rows.append(dict(seed=part,base=base.tolist(),novel=new.tolist(),pseudo_base=pb.tolist(),pseudo_novel=pn.tolist(),B=B))
    return rows

def run_task(ds,enc,block,entry,d,cache,device):
    part=entry['seed'];b,n,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']]
    dest=OUT/block/enc/ds/(f'B{len(b)}' if block=='cardinality' else 'catalog')/f'p{part}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    started=time.monotonic();Ef=d['E_fit'].to(device);Es=d['E_support'].to(device);T=d['T'].to(device);yf=d['fit_y'];ys=d['support_y']
    m=np.isin(d['tune_y'],b);Ev=d['E_tune'][m].to(device);yv=d['tune_y'][m]
    selectionpath=dest/'selection.json';weights=dest/'weights.pt'
    methods=MECHANISMS if block=='mechanism' else (['C15','C20','C30','C40','C60'] if block=='cardinality' else ['full','none','fixed','dense'])
    if selectionpath.exists():
        selection=json.loads(selectionpath.read_text());ww=torch.load(weights,map_location=device,weights_only=False);W,Wp=ww['W'],ww['Wp']
    else:
        if block=='mechanism':
            old=ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{part}/selection.json';orig=json.loads(old.read_text())
            lam,plam=orig['ridge_lambda'],orig['pseudo_ridge_lambda'];W=core.fit_ridge(Ef,yf,b,lam);Wp=core.fit_ridge(Ef,yf,pb,plam)
        else:
            W,lam=core.select_ridge(Ef,yf,Ev,yv,b);Wp,plam=core.select_ridge(Ef,yf,Ev,yv,pb)
        Pb=core.means(Ef,yf,b);Ppb=core.means(Ef,yf,pb);psi=core.support_rows(ys,pn,5,7);Ppn=core.means(Es[psi],ys[psi],pn)
        fm=readouts.fixed_moments(Ev,T[b],Pb,W);pm=readouts.fixed_moments(Ev,T[pb],Ppb,Wp)
        pr=raw(Ev,T[np.r_[pb,pn]],Ppb,Ppn,Wp);selections={}
        with torch.no_grad():
            for method in methods:
                check_pause();C=float(method[1:]) if method.startswith('C') else 30
                pc=comps(pr,pm,'full' if method=='dense' or method.startswith('C') else method,C,11+part*100)
                selections[method]=tune.select(pc,'dense' if method=='dense' else 'retained',np.r_[pb,pn],yv,pb,pn)
                print('SELECT',block,enc,ds,len(b),part,method,flush=True)
        gate_states={}
        if block=='external':
            par=selections['full']['selected'];pc=comps(pr,pm,'full');c=rv.compact(score(pc,'full',par),np.r_[pb,pn],yv,pb,pn)
            with torch.no_grad():X=gate_features(pr,pc,par,c)
            with torch.enable_grad():gate_states={g:gates.select_gate(g,X,c,yv,pb,pn,72000+part) for g in ['logistic','mlp']}
        atomic_torch(dict(W=W.cpu(),Wp=Wp.cpu()),weights);W=W.to(device);Wp=Wp.to(device)
        selection=dict(entry=entry,cache_sha256=digest_file(cache),ridge_lambda=lam,pseudo_ridge_lambda=plam,full_moments=fm,pseudo_moments=pm,selections=selections,gates=gate_states,settings_per_readout=3636,selection_access='fit and base-only tuning/support;no calibration/test labels',weights_sha256=digest_file(weights))
        atomic_json(selection,selectionpath)
        del pr;gc.collect()
    Pb=core.means(Ef,yf,b);fm=selection['full_moments'];key='C30' if block=='cardinality' else 'full';refpar=selection['selections'][key]['selected']
    # Reference depends only on the locked Full parameters and base fitting/tuning.
    def reference(E,y):
        r=raw(E,T[b],Pb,torch.empty((0,Ef.shape[1]),device=device),W)
        z=comps(r,fm,'full');sc=z['text']+refpar['a']*z['learned'];return b[sc.argmax(1).cpu().numpy()]==y
    maskc=np.isin(d['cal_y'],b);Ec=d['E_cal'][maskc].to(device);yc=d['cal_y'][maskc]
    maskt=np.isin(d['test_y'],np.r_[b,n]);Et=d['E_test'][maskt].to(device);yt=d['test_y'][maskt]
    with torch.no_grad():refc=reference(Ec,yc);reft=reference(Et,yt)
    closed=float(np.mean([reft[yt==cl].mean() for cl in b]));w=risk.weights(yc,b);si=core.support_rows(ys,n,5,7)
    ridge_error=float((core.linear(Ev[:128],W).mean(1)-1/len(b)).abs().max());assert ridge_error<1e-4,ridge_error
    for order in [0,1]:
        ordered=np.random.default_rng(1000+part+100*order).permutation(n)
        for session in range(1,6):
            jp=dest/f'o{order}_t{session}.json'
            if jp.exists():continue
            check_pause();novel=ordered[:len(n)*session//5];ids=np.r_[b,novel]
            with torch.no_grad():Pn=core.means(Es[si],ys[si],novel)
            systems={};archive=dict(base_classes=b,novel_classes=novel,cal_labels=yc,test_labels=yt,cal_reference_correct=refc,test_reference_correct=reft)
            with torch.no_grad():
                for pool,E,y,ps in [('tune',Ev,yv,11),('cal',Ec,yc,23),('test',Et,yt,37)]:
                    r=raw(E,T[ids],Pb,Pn,W)
                    for method in methods:
                        par=selection['selections'][method]['selected'];c=makecompact(r,fm,method,par,ids,y,b,novel,ps+part*100);systems[pool,method]=c
                        if pool!='tune':archive.update({pool+'_'+method+'_'+k:v for k,v in tune.pack(c).items()})
                    if block=='external':
                        cc=comps(r,fm,'full');c=systems[pool,'full'];X=gate_features(r,cc,refpar,c)
                        for g,state in selection['gates'].items():
                            gcpt=dict(c,margin=gates.apply_gate(state,X));systems[pool,g]=gcpt
                            if pool!='tune':archive.update({pool+'_'+g+'_'+k:v for k,v in tune.pack(gcpt).items()})
            allmethods=list(methods)+list(selection['gates']);rows=[]
            for method in allmethods:
                c,tc,vc=[systems[k,method] for k in ['cal','test','tune']];candidates=np.unique(np.quantile(vc['margin'],np.linspace(0,1,101)));archive[method+'_candidates']=candidates
                policy=fairness.evaluate_policy(c,refc,w,candidates,tc,reft,closed,'paired_accuracy')
                par=selection['selections'].get(method,{}).get('selected',{})
                fixed=par.get('offset',selection['gates'].get(method,{}).get('refit_offset'))
                rows.append(dict(method=method,policy=policy,capacity95=frontier(tc,.95*closed),auc=auc(tc),fixed_gate=rv.evaluate(tc,fixed)))
            policies={r['method']:r['policy'] for r in rows}
            for row in rows:
                tc=systems['test',row['method']]
                row['matched_actual_base']={m:frontier(tc,pol['old']) for m,pol in policies.items()}
            zp=jp.with_suffix('.npz');atomic_npz(zp,**archive)
            atomic_json(dict(encoder=enc,dataset=ds,block=block,part=part,B=len(b),order=order,session=session,reference_closed=closed,selection_sha256=digest_file(selectionpath),archive_sha256=digest_file(zp),records=rows),jp)
    atomic_json(dict(complete=True,ridge_mean_error=ridge_error,seconds=time.monotonic()-started),dest/'complete.json')
    print('COMPLETE',block,enc,ds,len(b),part,round(time.monotonic()-started,1),'s',flush=True)
    del Ef,Es,T,Ev,Et,Ec,W,Wp;gc.collect();torch.cuda.empty_cache()

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--block',choices=['mechanism','cardinality','external'],required=True);ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--dataset');ap.add_argument('--parts',type=int,default=10);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.smoke:OUT=ROOT/'experiments/expansion_prospective_validation_smoke'
    device=configure();torch.manual_seed(28102026);freeze();check_pause()
    datasets=[a.dataset] if a.dataset else {'mechanism':['cifar','pets','imagenetr'],'cardinality':['cifar','imagenetr'],'external':['food','sun']}[a.block]
    for enc in ([a.encoder] if a.encoder else ['l14','b32']):
        for ds in datasets:
            d,cache=load_data(ds,enc);entries=assignments(ds,d,a.block);dp=OUT/a.block/enc/ds/'tasks.json'
            if dp.exists():assert json.loads(dp.read_text())==entries
            else:atomic_json(entries,dp)
            for entry in entries:
                if entry['seed']<a.parts:run_task(ds,enc,a.block,entry,d,cache,device)
            del d;gc.collect();torch.cuda.empty_cache()
    if a.parts==10 and not a.encoder and not a.dataset:atomic_json(dict(complete=True),OUT/a.block/'complete.json')
if __name__=='__main__':main()
