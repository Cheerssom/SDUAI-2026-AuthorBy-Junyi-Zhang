"""Matched frozen-CLIP CUB FSCIL readouts; official indices, separate certification track."""
from pathlib import Path
import json,argparse,hashlib,time,sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
R=Path(__file__).resolve().parents[2];GRID=[0.,.25,.5,1.,2.,4.]
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def pause():
 if (R/'PAUSE').exists():raise SystemExit('PAUSED')
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
def cdf(v):return .5*(1+torch.erf((v-v.mean(1,keepdim=True))/v.std(1,correction=1,keepdim=True).clamp_min(1e-6)/2**.5))
def hycal(E,T,train,y,classes,reg=.001):
 cs=[];md=[];counts=[]
 for j,k in enumerate(classes):
  x=norm(train[y==k]+T[j]);mu=norm(x.mean(0));A=(x-mu).double();L=torch.linalg.cholesky(A@A.T+reg*(len(x)-1)*torch.eye(len(x),device='cuda'));B=torch.linalg.solve_triangular(L,A,upper=False).float();z=norm(E+T[j]);diff=z-mu;cs.append(z@mu);md.append(-((diff*diff).sum(1)-(diff@B.T).square().sum(1)).clamp_min(0)/reg);counts.append(len(x))
 co=cdf(torch.stack(cs,1));ma=cdf(torch.stack(md,1));w=torch.sigmoid(torch.tensor(counts,device='cuda')-5.)
 return (1-w)*co+w*ma
@torch.inference_mode()
def run(enc,track,seed):
 torch.manual_seed(2026+seed);out=R/'readouts_v2'/enc/track/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return
 z=torch.load(R/'features'/enc/'frozen.pt',map_location='cpu',weights_only=False);E=z['E'].cuda();T=z['T'].cuda();y=np.asarray(z['y']);iid=np.asarray(z['ids']);sessions=z['sessions'];base=np.arange(100);rng=np.random.default_rng(28012026+seed);index={int(v):i for i,v in enumerate(iid)};official=np.array([index[v] for v in sessions['1']]);fit=[];tune=[];cal=[]
 for k in base:
  q=rng.permutation(official[y[official]==k]);tune.extend(q[:5]);cal.extend(q[5:15]);fit.extend(q[15:])
 fit,tune,cal=[np.array(v) for v in [fit,tune,cal]];selectionfit=fit if track=='certified' else np.setdiff1d(official,tune);actualfit=fit if track=='certified' else official
 assert not set(fit)&set(tune) and not set(fit)&set(cal) and not set(tune)&set(cal)
 def fitted(ix,classes,valid):
  best=(-1,None,None)
  for lam in [.01,.1,1.,10.]:
   w=ridge(E[ix],y[ix],classes,lam);acc=macro(prediction(lin(E[valid],w),classes)==y[valid],y[valid],classes)
   if acc>best[0]:best=(acc,w,lam)
  return best[1:]
 Ws,lam=fitted(selectionfit,base,tune);pb=np.arange(50);pn=np.arange(50,100);pfit=selectionfit[np.isin(y[selectionfit],pb)];Wp,plam=fitted(pfit,pb,tune[np.isin(y[tune],pb)]);ps=np.concatenate([rng.choice(selectionfit[y[selectionfit]==k],5,replace=False) for k in pn]);Ppb=means(E[pfit],y[pfit],pb);Ppn=means(E[ps],y[ps],pn);fm=[]
 for v in [E[tune]@T[pb].T,lin(E[tune],Wp),E[tune]@Ppb.T]:fm.append((float(v.mean()),float(v.std(1,correction=0).mean().clamp_min(1e-6))))
 pars={}
 for mode in ['full','none','fixed','dense']:
  c=components(E[tune],T[np.r_[pb,pn]],Ppb,Ppn,Wp,fm,mode);pars[mode]=select(c,np.r_[pb,pn],y[tune],pb,pn,mode=='dense')
 W=ridge(E[actualfit],y[actualfit],base,lam);Pb=means(E[actualfit],y[actualfit],base);fm=[]
 for v in [E[tune]@T[base].T,lin(E[tune],W),E[tune]@Pb.T]:fm.append((float(v.mean()),float(v.std(1,correction=0).mean().clamp_min(1e-6))))
 par=pars['full'];ref=lambda ix:prediction(components(E[ix],T[base],Pb,Pb[:0],W,fm,'full')[0]+par['a']*components(E[ix],T[base],Pb,Pb[:0],W,fm,'full')[1],base)==y[ix]
 # Closed reference is stored before any future-class evidence enters scoring.
 test=np.flatnonzero(~z['train']);bt=test[y[test]<100];rc=ref(cal);rt=ref(bt);closed=macro(rt,y[bt],base)
 atomic(dict(encoder=enc,track=track,seed=seed,fit_ids=iid[actualfit].tolist(),tune_ids=iid[tune].tolist(),cal_ids=iid[cal].tolist() if track=='certified' else [],ridge_lambda=lam,pseudo_lambda=plam,selections=pars,settings_per_fusion=3636,shared_reference_closed_test=closed,features_sha256=hashlib.sha256((R/'features'/enc/'frozen.pt').read_bytes()).hexdigest()),out/'selection.json')
 RP=torch.randn(E.shape[1],10000,device='cuda');H=F.relu(E@RP);trainidx=list(actualfit);allrows=[]
 # RanPAC original ridge selection on an 80/20 base-feature split; dual solve is algebraically equivalent.
 rporder=rng.permutation(actualfit);cut=int(.8*len(rporder));a=rporder[:cut];v=rporder[cut:];Y=F.one_hot(torch.as_tensor(y[a],device='cuda'),200).float();G=H[a]@H[a].T;eye=torch.eye(len(a),device='cuda');loss=[]
 for l in 10.**np.arange(-8,9):
  weight=H[a].T@torch.linalg.solve(G+l*eye,Y);loss.append(float(F.mse_loss(H[v]@weight,F.one_hot(torch.as_tensor(y[v],device='cuda'),200).float())))
 rplam=float((10.**np.arange(-8,9))[int(np.argmin(loss))]);del G,eye
 for session in range(11):
  pause();dest=out/f'session{session:02d}.json'
  if session>0:trainidx.extend(index[v] for v in sessions[str(session+1)])
  if dest.exists():allrows.extend(json.loads(dest.read_text())['rows']);continue
  n=np.arange(100,100+10*session);ids=np.r_[base,n];active=test[y[test]<len(ids)];queries=np.r_[tune,cal,active];Q=E[queries];qy=y[queries];N=len(tune);C=len(cal);trainidxnp=np.array(trainidx);P=means(E[trainidxnp],y[trainidxnp],ids);Pn=P[100:];scores={}
  for mode in pars:
   co=components(Q,T[ids],Pb,Pn,W,fm,mode);scores[mode]=fusion(co,pars[mode]['a'],pars[mode]['u'],100,mode=='dense')
  scores['SimpleCIL_transfer']=Q@P.T
  hh=H[trainidxnp];YY=F.one_hot(torch.as_tensor(y[trainidxnp],device='cuda'),200).float();coef=torch.linalg.solve(hh@hh.T+rplam*torch.eye(len(hh),device='cuda'),YY);scores['RanPAC_frozen_transfer']=(H[queries]@hh.T@coef)[:,:len(ids)]
  scores['HyCal_transfer']=hycal(Q,T[ids],E[trainidxnp],y[trainidxnp],ids)
  if (R/'description_features'/f'{enc}.pt').exists():
   desc=torch.load(R/'description_features'/f'{enc}.pt',map_location='cuda',weights_only=False);visual=P.clone()
   if len(n):visual[100:]=norm(.9*P[100:]+.1*norm(torch.softmax(16*(P[100:]@Pb.T),1)@Pb))
   fp=norm(.3*(.5*T[ids]+.5*desc[ids])+.7*visual);scores['BiMC_transfer']=torch.softmax(Q@fp.T,dim=1)
  rows=[];ownrefs={} if session==0 else dict(np.load(out/'closed_references.npz'))
  for method,s in scores.items():
   off=pars[method]['offset'] if method in pars else 0.;ss=s[N+C:].clone()
   if session: ss[:,100:]+=off
   pred=prediction(ss,ids);row=dict(method=method,session=session,unconstrained_macro=macro(pred==y[active],y[active],ids),unconstrained_top1=float(np.mean(pred==y[active])),base=macro(pred[y[active]<100]==y[active][y[active]<100],y[active][y[active]<100],base))
   if session and track=='certified':
    vc=comp(s[:N],ids,y[tune],base,n);cc=comp(s[N:N+C],ids,y[cal],base,n);tc=comp(s[N+C:],ids,y[active],base,n);candidates=np.unique(np.quantile(vc['margin'],np.linspace(0,1,101)));w=risk.weights(y[cal],base);tr=prediction(s[N+C:,:100],base)==y[active];br=np.isin(y[active],base)
    for kind in ['common','self']:
     selfrc=ownrefs[method+'_cal'];selfrt=ownrefs[method+'_base_test'];refc=rc if kind=='common' else selfrc;refclosed=closed if kind=='common' else macro(selfrt,y[active][br],base);detail=paired.certify(cc,refc,w,candidates,sessions=10);dd=detail['offset'];metrics=dict(base=refclosed,novel=0.) if dd is None else evaluate(tc,dd);row[kind]=dict(**detail,**metrics,closed_base=refclosed,met=metrics['base']>=.95*refclosed-1e-10,oracle=frontier(tc,.95*refclosed))
   if session==0:
    ownrefs[method+'_cal']=(prediction(s[N:N+C],base)==y[cal]);ownrefs[method+'_base_test']=(pred==y[active])
   rows.append(row)
  if session==0:np.savez_compressed(out/'closed_references.npz',**ownrefs)
  atomic(dict(rows=rows,session=session),dest);allrows+=rows;print('CUB_READOUT',enc,track,seed,session,'methods',len(rows),flush=True)
  del scores,hh,coef
 atomic(dict(complete=True,rows=len(allrows),methods=sorted(set(r['method'] for r in allrows)),rp_dimension=10000,rp_lambda=rplam),out/'complete.json')
def main():
 p=argparse.ArgumentParser();p.add_argument('--encoder',required=True);p.add_argument('--track',choices=['standard','certified'],required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.7);run(a.encoder,a.track,a.seed)
if __name__=='__main__':main()
