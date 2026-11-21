"""Standard FSCIL benchmarks (CIFAR-100, miniImageNet, CUB-200; CEC indices) with frozen CLIP.

Readouts, 3,636-setting selection, transfers and certification follow expansion_cub_readouts.py /
expansion_cifar_fscil_readouts.py. Added training-free baselines used by BiMC (CVPR25) and recent
CLIP-FSCIL work: CLIP zero-shot, TEEN-style prototype calibration and FeCAM-style shrunk-covariance
Mahalanobis scoring, all on the same frozen features and supports (matched transfers).
"""
from pathlib import Path
import json,argparse,hashlib,time,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
R=Path(__file__).resolve().parent;D=R/'work';GRID=[0.,.25,.5,1.,2.,4.]
CFG={'cifar100':(60,5,9,10,40,100),'mini':(60,5,9,10,40,100),'cub200':(100,10,11,5,10,200)}
NB=WAY=S=NTUNE=NCAL=NC=None;ONLY='';TEXT='names';TAG=''
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def pause():
 if (D/'PAUSE').exists():raise SystemExit('PAUSED')
def norm(x):return F.normalize(x,dim=-1)
def means(E,y,classes):return norm(torch.stack([E[y==c].mean(0) for c in classes]))
def ridge(E,y,classes,lam):
 x=torch.cat([E.double(),torch.ones(len(E),1,device='cuda')],1);t=torch.tensor(np.searchsorted(classes,y),device='cuda');Y=F.one_hot(t,len(classes)).double();w=1/torch.bincount(t)[t].double();reg=torch.eye(x.shape[1],device='cuda')*lam;reg[-1,-1]=0
 return torch.linalg.solve(x.T@(x*w[:,None])+reg,x.T@(Y*w[:,None])).float()
def lin(E,W):return E@W[:-1]+W[-1]
def macro(c,y,classes):return float(np.mean([np.mean(c[y==k]) for k in classes]))
def prediction(s,ids):return ids[s.argmax(1).cpu().numpy()]
def comp(scores,ids,y,b,n):
 nb=len(b);bp=prediction(scores[:,:nb],b);pn=prediction(scores[:,nb:],n);w=np.zeros(len(y));bm=np.isin(y,b);nm=np.isin(y,n)
 for group in [b,n]:
  for k in group:
   mask=y==k
   if mask.any():w[mask]=1/(len(group)*mask.sum())
 return dict(margin=(scores[:,:nb].max(1).values.double()-scores[:,nb:].max(1).values.double()).cpu().numpy(),base_prediction=bp,novel_prediction=pn,base_correct=bm&(bp==y),novel_correct=nm&(pn==y),weights=w,nb=len(b),nn=len(n))
def evaluate(c,d):
 ch=risk.choose_new(c,d);a=float(c['weights']@(c['base_correct']&~ch));n=float(c['weights']@(c['novel_correct']&ch));return dict(base=a,novel=n,overall=(a*c['nb']+n*c['nn'])/(c['nb']+c['nn']))
def frontier(c,target):
 ix=np.flatnonzero(c['base_correct']);order=ix[np.argsort(-c['margin'][ix],kind='stable')];v=np.cumsum(c['weights'][order])
 if not len(v) or target>v[-1]+1e-10:return dict(feasible=False)
 j=min(np.searchsorted(v,target-1e-12),len(order)-1);d=float(c['margin'][order[j]])
 if evaluate(c,d)['base']<target-1e-10:d=float(np.nextafter(d,-np.inf))
 return dict(feasible=True,offset=d,**evaluate(c,d))
def components(E,T,Pb,Pn,W,fm,mode):
 out=[];nb=len(Pb)
 for i,v in enumerate([E@T.T,lin(E,W),E@torch.cat([Pb,Pn]).T]):
  ref=v if i==1 else v[:,:nb];mu=ref.mean(1,keepdim=True);sd=ref.std(1,correction=0,keepdim=True).clamp_min(1e-6)
  if mode=='fixed':mu,sd=fm[i]
  if mode=='none':mu=0.
  out.append((v-mu)/sd)
 return out
def fusion(c,a,u,nb,dense=False):
 s=c[0].clone();s[:,:nb]+=a*(c[2][:,:nb] if dense else c[1]);s[:,nb:]+=u*c[2][:,nb:];return s
def select(c,ids,y,b,n,dense=False):
 best=(-1,None)
 for a in GRID:
  for u in GRID:
   cc=comp(fusion(c,a,u,len(b),dense),ids,y,b,n);d=np.quantile(cc['margin'],np.linspace(0,1,101));ch=(d[:,None]>cc['margin'][None])|((d[:,None]==cc['margin'][None])&(cc['novel_prediction'][None]<cc['base_prediction'][None]));ab=(~ch)@(cc['weights']*cc['base_correct']);an=ch@(cc['weights']*cc['novel_correct']);h=np.divide(2*ab*an,ab+an,out=np.zeros_like(ab),where=ab+an>0);j=int(h.argmax())
   if h[j]>best[0]+1e-12:best=(h[j],dict(a=a,u=u,offset=float(d[j]),harmonic=float(h[j])))
 return best[1]
TEEN_ALPHA,TEEN_TAU=.9,16.
@torch.inference_mode()
def fecam_cpu(Q,train,y,classes,cache,g1=1.,g2=1.):
 # FeCAM-style Mahalanobis scoring (Goswami et al., NeurIPS 2023): shrunk covariance (diagonal and
 # off-diagonal mean shrinkage), correlation normalisation, negated squared Mahalanobis distance.
 # Every class uses the same regularised estimator so that distances share one scale:
 #   Sigma_c = (n_c * S_c + m * S_pool) / (n_c + m),  m = feature dimension,
 # where S_pool is the mean base-class covariance (regularised discriminant analysis). Classes with
 # few images (CUB base classes, all 5-shot novel classes) lean on S_pool; data-rich classes keep
 # their own structure.
 def finish(mu,C):
  d=torch.diagonal(C);v1=d.mean();off=C-torch.diag(d);v2=off.sum()/max(C.numel()-len(d),1);I=torch.eye(len(d),dtype=C.dtype)
  C=C+g1*v1*I+g2*v2*(1-I);sc=torch.sqrt(torch.diagonal(C)).clamp_min(1e-8);C=C/(sc[:,None]*sc[None]);return mu,sc,torch.linalg.pinv(C)
 def own(x):
  xc=x-x.mean(0);return xc.T@xc/max(len(x)-1,1)
 if '__pool' not in cache:
  cache['__pool']=torch.stack([own(train[y==k].double()) for k in classes if int(k)<NB]).mean(0)
 pool=cache['__pool'];m=float(pool.shape[0])
 out=[]
 for k in classes:
  key=int(k);count=int(np.sum(y==k))
  if key not in cache:
   x=train[y==k].double();mu=x.mean(0);n=float(len(x))
   C=(n*own(x)+m*pool)/(n+m) if n>1 else pool.clone()
   cache[key]=(count,)+finish(mu,C)
  cached_count,mu,sc,P=cache[key];assert cached_count==count
  z=(Q.double()-mu)/sc
  out.append(-(z@P*z).sum(1))
 return torch.stack(out,1).float()
def cdf(v):return .5*(1+torch.erf((v-v.mean(1,keepdim=True))/v.std(1,correction=1,keepdim=True).clamp_min(1e-6)/2**.5))
def hycal(E,T,train,y,classes,reg=.001):
 cs=[];md=[];counts=[]
 for j,k in enumerate(classes):
  x=norm(train[y==k]+T[j]);mu=norm(x.mean(0));A=(x-mu).double();L=torch.linalg.cholesky(A@A.T+reg*(len(x)-1)*torch.eye(len(x),device='cuda'));B=torch.linalg.solve_triangular(L,A,upper=False).float();z=norm(E+T[j]);diff=z-mu;cs.append(z@mu);md.append(-((diff*diff).sum(1)-(diff@B.T).square().sum(1)).clamp_min(0)/reg);counts.append(len(x))
 co=cdf(torch.stack(cs,1));ma=cdf(torch.stack(md,1));w=torch.sigmoid(torch.tensor(counts,device='cuda')-5.)
 return (1-w)*co+w*ma
@torch.inference_mode()
def run(ds,enc,track,seed):
 global NB,WAY,S,NTUNE,NCAL,NC
 NB,WAY,S,NTUNE,NCAL,NC=CFG[ds]
 torch.manual_seed(2026+seed);out=D/f'readouts{TAG}'/ds/enc/track/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists() and not ONLY:return
 z=torch.load(D/'features'/ds/enc/'frozen.pt',map_location='cpu',weights_only=False);E=z['E'].cuda();T=z['T'].cuda();Torig=T
 dpath=D/'description_features'/ds/f'{enc}.pt'
 if TEXT=='desc':
  assert dpath.exists(),f'missing {dpath}: run encode_descriptions.py first'
  T=norm(.5*Torig+.5*torch.load(dpath,map_location='cuda',weights_only=False).float())
 print('TEXT_ANCHORS',TEXT,ds,enc,flush=True)
 y=np.asarray(z['y']);iid=np.asarray(z['ids']);sessions=z['sessions'];base=np.arange(NB);rng=np.random.default_rng(28012026+seed);index={int(v):i for i,v in enumerate(iid)};official=np.array([index[v] for v in sessions['1']]);fit=[];tune=[];cal=[]
 for k in base:
  q=rng.permutation(official[y[official]==k]);tune.extend(q[:NTUNE]);cal.extend(q[NTUNE:NTUNE+NCAL]);fit.extend(q[NTUNE+NCAL:])
 fit,tune,cal=[np.array(v) for v in [fit,tune,cal]];selectionfit=fit if track=='certified' else np.setdiff1d(official,tune);actualfit=fit if track=='certified' else official
 assert not set(fit)&set(tune) and not set(fit)&set(cal) and not set(tune)&set(cal)
 def fitted(ix,classes,valid):
  best=(-1,None,None)
  for lam in [.01,.1,1.,10.]:
   w=ridge(E[ix],y[ix],classes,lam);acc=macro(prediction(lin(E[valid],w),classes)==y[valid],y[valid],classes)
   if acc>best[0]:best=(acc,w,lam)
  return best[1:]
 Ws,lam=fitted(selectionfit,base,tune);pb=np.arange(NB//2);pn=np.arange(NB//2,NB);pfit=selectionfit[np.isin(y[selectionfit],pb)];Wp,plam=fitted(pfit,pb,tune[np.isin(y[tune],pb)]);ps=np.concatenate([rng.choice(selectionfit[y[selectionfit]==k],5,replace=False) for k in pn]);Ppb=means(E[pfit],y[pfit],pb);Ppn=means(E[ps],y[ps],pn);fm=[]
 for v in [E[tune]@T[pb].T,lin(E[tune],Wp),E[tune]@Ppb.T]:fm.append((float(v.mean()),float(v.std(1,correction=0).mean().clamp_min(1e-6))))
 pars={}
 for mode in ['full','none','fixed','dense']:
  c=components(E[tune],T[np.r_[pb,pn]],Ppb,Ppn,Wp,fm,mode);pars[mode]=select(c,np.r_[pb,pn],y[tune],pb,pn,mode=='dense')
 W=ridge(E[actualfit],y[actualfit],base,lam);Pb=means(E[actualfit],y[actualfit],base);fm=[]
 for v in [E[tune]@T[base].T,lin(E[tune],W),E[tune]@Pb.T]:fm.append((float(v.mean()),float(v.std(1,correction=0).mean().clamp_min(1e-6))))
 par=pars['full'];ref=lambda ix:prediction(components(E[ix],T[base],Pb,Pb[:0],W,fm,'full')[0]+par['a']*components(E[ix],T[base],Pb,Pb[:0],W,fm,'full')[1],base)==y[ix]
 # Closed reference is stored before any future-class evidence enters scoring.
 test=np.flatnonzero(~z['train']);bt=test[y[test]<NB];rc=ref(cal);rt=ref(bt);closed=macro(rt,y[bt],base)
 atomic(dict(encoder=enc,track=track,seed=seed,fit_ids=iid[actualfit].tolist(),tune_ids=iid[tune].tolist(),cal_ids=iid[cal].tolist() if track=='certified' else [],ridge_lambda=lam,pseudo_lambda=plam,selections=pars,settings_per_fusion=3636,shared_reference_closed_test=closed,features_sha256=hashlib.sha256((D/'features'/ds/enc/'frozen.pt').read_bytes()).hexdigest()),out/'selection.json')
 cpu_features=z['E'];fecam_cache={};fecam_pool=ThreadPoolExecutor(max_workers=1);print('CPU_GPU_OVERLAP FeCAM CPU threads=4, other scoring CUDA',flush=True)
 RP=torch.randn(E.shape[1],10000,device='cuda');feat=lambda ix:F.relu(E[ix]@RP);trainidx=list(actualfit);allrows=[]
 # RanPAC original ridge selection on an 80/20 base-feature split (primal solve).
 rporder=rng.permutation(actualfit);cut=int(.8*len(rporder));a=rporder[:cut];v=rporder[cut:];Ha=feat(a);Hv=feat(v);Y=F.one_hot(torch.as_tensor(y[a],device='cuda'),NC).float();G=Ha.T@Ha;HY=Ha.T@Y;eye=torch.eye(G.shape[0],device='cuda');loss=[]
 for l in 10.**np.arange(-8,9):
  weight=torch.linalg.solve(G+l*eye,HY);loss.append(float(F.mse_loss(Hv@weight,F.one_hot(torch.as_tensor(y[v],device='cuda'),NC).float())))
  torch.cuda.empty_cache()
 rplam=float((10.**np.arange(-8,9))[int(np.argmin(loss))]);del G,HY,Ha,Hv;torch.cuda.empty_cache()
 for session in range(S):
  pause();dest=out/f'session{session:02d}.json'
  if session>0:trainidx.extend(index[v] for v in sessions[str(session+1)])
  if dest.exists() and not ONLY:allrows.extend(json.loads(dest.read_text())['rows']);continue
  n=np.arange(NB,NB+WAY*session);ids=np.r_[base,n];active=test[y[test]<len(ids)];queries=np.r_[tune,cal,active];Q=E[queries];qy=y[queries];N=len(tune);C=len(cal);trainidxnp=np.array(trainidx);P=means(E[trainidxnp],y[trainidxnp],ids);Pn=P[NB:];scores={}
  if ONLY=='FeCAM_transfer':
   scores={'FeCAM_transfer':fecam_cpu(cpu_features[queries],cpu_features[trainidxnp],y[trainidxnp],ids.copy(),fecam_cache).cuda()};hh=coef=None
  else:
   for mode in pars:
    co=components(Q,T[ids],Pb,Pn,W,fm,mode);scores[mode]=fusion(co,pars[mode]['a'],pars[mode]['u'],NB,mode=='dense')
   scores['SimpleCIL_transfer']=Q@P.T
   scores['ZeroShot_CLIP']=Q@T[ids].T
   tp=P.clone()
   if len(n):tp[NB:]=norm(TEEN_ALPHA*P[NB:]+(1-TEEN_ALPHA)*(torch.softmax(TEEN_TAU*(P[NB:]@Pb.T),1)@Pb))
   scores['TEEN_transfer']=Q@tp.T
   fecam_future=fecam_pool.submit(fecam_cpu,cpu_features[queries],cpu_features[trainidxnp],y[trainidxnp],ids.copy(),fecam_cache);scores['FeCAM_transfer']=None
   hh=feat(trainidxnp);YY=F.one_hot(torch.as_tensor(y[trainidxnp],device='cuda'),NC).float();coef=torch.linalg.solve(hh.T@hh+rplam*eye,hh.T@YY);scores['RanPAC_frozen_transfer']=(feat(queries)@coef)[:,:len(ids)]
   scores['HyCal_transfer']=hycal(Q,T[ids],E[trainidxnp],y[trainidxnp],ids)
   if True:
    # No curated CIFAR descriptions exist; BiMC's description branch falls back to the class-name anchors.
    desc=torch.load(dpath,map_location='cuda',weights_only=False).float() if (TEXT=='desc' and dpath.exists()) else Torig;visual=P.clone()
    if len(n):visual[NB:]=norm(.9*P[NB:]+.1*norm(torch.softmax(16*(P[NB:]@Pb.T),1)@Pb))
    fp=norm(.3*(.5*Torig[ids]+.5*desc[ids])+.7*visual);scores['BiMC_transfer']=torch.softmax(Q@fp.T,dim=1)
   scores['FeCAM_transfer']=fecam_future.result().cuda()
  rows=[];ownrefs={} if session==0 else dict(np.load(out/'closed_references.npz'))
  for method,s in scores.items():
   off=pars[method]['offset'] if method in pars else 0.;ss=s[N+C:].clone()
   if session: ss[:,NB:]+=off
   pred=prediction(ss,ids);row=dict(method=method,session=session,unconstrained_macro=macro(pred==y[active],y[active],ids),unconstrained_top1=float(np.mean(pred==y[active])),base=macro(pred[y[active]<NB]==y[active][y[active]<NB],y[active][y[active]<NB],base))
   if session and track=='certified':
    vc=comp(s[:N],ids,y[tune],base,n);cc=comp(s[N:N+C],ids,y[cal],base,n);tc=comp(s[N+C:],ids,y[active],base,n);candidates=np.unique(np.quantile(vc['margin'],np.linspace(0,1,101)));w=risk.weights(y[cal],base);tr=prediction(s[N+C:,:NB],base)==y[active];br=np.isin(y[active],base)
    for kind in ['common','self']:
     selfrc=ownrefs[method+'_cal'];selfrt=ownrefs[method+'_base_test'];refc=rc if kind=='common' else selfrc;refclosed=closed if kind=='common' else macro(selfrt,y[active][br],base);detail=paired.certify(cc,refc,w,candidates,sessions=S-1);dd=detail['offset'];metrics=dict(base=refclosed,novel=0.) if dd is None else evaluate(tc,dd);row[kind]=dict(**detail,**metrics,closed_base=refclosed,met=metrics['base']>=.95*refclosed-1e-10,oracle=frontier(tc,.95*refclosed))
   if session==0:
    ownrefs[method+'_cal']=(prediction(s[N:N+C],base)==y[cal]);ownrefs[method+'_base_test']=(pred==y[active])
   rows.append(row)
  if session==0:
   if ONLY and (out/'closed_references.npz').exists():ownrefs={**dict(np.load(out/'closed_references.npz')),**ownrefs}
   np.savez_compressed(out/'closed_references.npz',**ownrefs)
  if ONLY and dest.exists():rows=[r for r in json.loads(dest.read_text())['rows'] if r['method']!=ONLY]+rows
  atomic(dict(rows=rows,session=session),dest);allrows+=rows;print('FSCIL_READOUT',ds,enc,track,seed,session,'methods',len(rows),flush=True)
  del scores,hh,coef
  torch.cuda.empty_cache()
 fecam_pool.shutdown(wait=True)
 atomic(dict(complete=True,rows=len(allrows),methods=sorted(set(r['method'] for r in allrows)),rp_dimension=10000,rp_lambda=rplam),out/'complete.json')
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CFG),required=True);p.add_argument('--encoder',required=True);p.add_argument('--track',choices=['standard','certified'],required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--only',default='');p.add_argument('--text',choices=['names','desc'],default='names');p.add_argument('--tag',default='');a=p.parse_args();global ONLY,TEXT,TAG;ONLY=a.only;TEXT=a.text;TAG=a.tag;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 if TEXT=='desc' and TAG!='_desc':p.error('Description experiment requires --tag _desc')
 if ONLY and (TEXT!='names' or TAG):p.error('--only replay is restricted to the baseline namespace')
 if ONLY:
  if ONLY!='FeCAM_transfer':p.error('--only supports FeCAM_transfer only')
  from fecam_replay import replay
  replay(a.dataset,a.encoder,a.track,a.seed);return
 run(a.dataset,a.encoder,a.track,a.seed)
if __name__=='__main__':main()
