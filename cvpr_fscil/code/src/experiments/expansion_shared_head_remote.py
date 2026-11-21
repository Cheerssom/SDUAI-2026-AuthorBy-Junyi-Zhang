"""Shared-expert group-gate diagnostic; all numeric work in cloud."""
from pathlib import Path
import argparse,json,hashlib,os,time
import numpy as np
import torch
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
R=Path(__file__).resolve().parents[2];I=R/'inputs';OUT=R/'results';GRID=(0.,.25,.5,1.,2.,4.);METHODS=('full_gate','prototype_gate')
def atomic(x,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pause():
 if (R/'PAUSE').exists():raise SystemExit('PAUSED at checkpoint')
def mean(E,y,ids):return torch.nn.functional.normalize(torch.stack([E[y==c].mean(0) for c in ids]),dim=1)
def support(y,ids):
 rng=np.random.default_rng(7);return np.concatenate([rng.permutation(np.flatnonzero(y==c))[:5] for c in ids])
def linear(E,W):return E@W[:-1]+W[-1]
def components(E,T,Pb,Pn,W,fm):
 nb=len(Pb);eta=min(1.,nb/30);out={}
 for i,(name,v) in enumerate([('text',E@T.T),('learned',linear(E,W)),('prototype',E@torch.cat([Pb,Pn]).T)]):
  ref=v if name=='learned' else v[:,:nb];mu=eta*ref.mean(1,keepdim=True)+(1-eta)*fm[i][0];sd=(eta*ref.std(1,correction=0,keepdim=True).clamp_min(1e-6)+(1-eta)*fm[i][1]).clamp_min(1e-6);out[name]=(v-mu)/sd
 return out
def winners(sc,ids):
 ix=np.argsort(ids);return ids[ix][sc[:,ix].argmax(1).cpu().numpy()]
def compact(c,par,method,shared,b,n,y):
 nb=len(b);bs=c['text'][:,:nb]+par['a']*(c['learned'] if method=='full_gate' else c['prototype'][:,:nb]);ns=c['text'][:,nb:]+par['u']*c['prototype'][:,nb:];bp,npred=shared;bm=np.isin(y,b);nm=np.isin(y,n);w=np.zeros(len(y))
 for group in [b,n]:
  for cl in group:
   ix=y==cl
   if ix.any():w[ix]=1/(len(group)*ix.sum())
 return dict(margin=(bs.max(1).values.double()-ns.max(1).values.double()).cpu().numpy(),base_prediction=bp,novel_prediction=npred,base_correct=bm&(bp==y),novel_correct=nm&(npred==y),base_mask=bm,novel_mask=nm,weights=w,nb=len(b),nn=len(n))
def shared(c,par,b,n):
 nb=len(b);return winners(c['text'][:,:nb]+par['a']*c['learned'],b),winners(c['text'][:,nb:]+par['u']*c['prototype'][:,nb:],n)
def evaluate(c,d):
 ch=risk.choose_new(c,d);ab=float(c['weights']@(c['base_correct']&~ch));an=float(c['weights']@(c['novel_correct']&ch));return dict(old=ab,new=an,overall=(ab*c['nb']+an*c['nn'])/(c['nb']+c['nn']),harmonic=2*ab*an/max(ab+an,1e-12))
def frontier(c,target):
 ix=np.flatnonzero(c['base_correct']);ceiling=float(c['weights'][ix].sum())
 if target>ceiling+1e-10:return dict(feasible=False,ceiling=ceiling)
 order=ix[np.argsort(-c['margin'][ix],kind='stable')];cum=np.cumsum(c['weights'][order]);j=min(int(np.searchsorted(cum,target-1e-12)),len(order)-1);d=float(c['margin'][order[j]])
 if evaluate(c,d)['old']<target-1e-10:d=float(np.nextafter(d,-np.inf))
 assert evaluate(c,d)['old']>=target-1e-10
 return dict(feasible=True,ceiling=ceiling,offset=d,**evaluate(c,d))
def select(c,experts,b,n,y):
 out={}
 for method in METHODS:
  trials=[];best=(-1,None)
  for a in GRID:
   for u in GRID:
    pause();p=dict(a=a,u=u);cc=compact(c,p,method,experts,b,n,y);ds=np.quantile(cc['margin'],np.linspace(0,1,101));ch=(ds[:,None]>cc['margin'][None])|((ds[:,None]==cc['margin'][None])&(cc['novel_prediction'][None]<cc['base_prediction'][None]));ab=(~ch)@(cc['weights']*cc['base_correct']);an=ch@(cc['weights']*cc['novel_correct']);h=np.divide(2*ab*an,ab+an,out=np.zeros_like(ab),where=(ab+an)>0);j=int(h.argmax());trials.append(dict(a=a,u=u,offsets=ds.tolist(),harmonics=h.tolist()))
    if h[j]>best[0]+1e-12:best=(float(h[j]),dict(a=a,u=u,offset=float(ds[j]),harmonic=float(h[j])))
  out[method]=dict(selected=best[1],trials=trials,evaluated_settings=3636)
 return out
def run(ds,en,args):
 d=torch.load(I/ds/en/'frozen.pt',map_location='cpu',weights_only=False)
 for k,v in list(d.items()):
  if torch.is_tensor(v):d[k]=v.to(args.device)
 for part in range(args.parts):
  po=OUT/ds/en/f'p{part}';po.mkdir(parents=True,exist_ok=True)
  if (po/'complete.json').exists():continue
  pause();src=I/ds/en/f'p{part}';s=json.loads((src/'selection.json').read_text());b,n,pb,pn=[np.asarray(s['entry'][k]) for k in ['base','novel','pseudo_base','pseudo_novel']];refpar=s['selections']['full']['selected'];ww=torch.load(src/'weights.pt',map_location=args.device,weights_only=False);W,Wp=ww['W'],ww['Wp'];Ef,Es,T=d['E_fit'],d['E_support'],d['T'];yf,ys=d['fit_y'],d['support_y'];Pb=mean(Ef,yf,b);Ppb=mean(Ef,yf,pb);Ppn=mean(Es[support(ys,pn)],ys[support(ys,pn)],pn);mask=np.isin(d['tune_y'],b);Ev=d['E_tune'][mask];yv=d['tune_y'][mask];pc=components(Ev,T[np.r_[pb,pn]],Ppb,Ppn,Wp,s['pseudo_moments']);experts=shared(pc,refpar,pb,pn);sp=po/'selection.json'
  if sp.exists():sel=json.loads(sp.read_text())
  else:sel=dict(reference_parameters=refpar,selections=select(pc,experts,pb,pn,yv),source_sha256=sha(src/'selection.json'),weights_sha256=sha(src/'weights.pt'));atomic(sel,sp)
  for order in [0,1]:
   ordered=np.random.default_rng(1000+part+100*order).permutation(n)
   for session in range(1,6):
    dest=po/f'o{order}_t{session}.json'
    if dest.exists():continue
    pause();novel=ordered[:len(n)*session//5];si=support(ys,n);Pn=mean(Es[si],ys[si],novel);ids=np.r_[b,novel];systems={};archive=dict(base_classes=b,novel_classes=novel)
    original=np.load(src/f'o{order}_t{session}.npz')
    for pool in ['tune','cal','test']:
     mm=np.isin(d[pool+'_y'],b if pool!='test' else np.r_[b,n]);E=d['E_'+pool][mm];y=d[pool+'_y'][mm];co=components(E,T[ids],Pb,Pn,W,s['full_moments']);ex=shared(co,refpar,b,novel);archive[pool+'_labels']=y
     for method in METHODS:
      par=sel['selections'][method]['selected'];cc=compact(co,par,method,ex,b,novel,y);systems[pool,method]=cc
      if pool!='tune':
       for k in ['margin','base_prediction','novel_prediction']:archive[pool+'_'+method+'_'+k]=cc[k]
     if pool!='tune':assert np.array_equal((ex[0]==y)&np.isin(y,b),original[pool+'_reference_correct']&np.isin(y,b)),(ds,en,part,pool)
    w=risk.weights(archive['cal_labels'],b);refc=systems['cal','full_gate']['base_correct'];reft=systems['test','full_gate']['base_correct'];tc0=systems['test','full_gate'];closed=float(tc0['weights']@reft);archive['cal_reference_correct']=refc;archive['test_reference_correct']=reft;rows=[]
    for method in METHODS:
     cc,tc,vc=[systems[p,method] for p in ['cal','test','tune']];assert np.array_equal(refc,cc['base_correct']) and np.array_equal(reft,tc['base_correct']);candidates=np.unique(np.quantile(vc['margin'],np.linspace(0,1,101)));archive[method+'_candidates']=candidates;detail=paired.certify(cc,refc,w,candidates);second=paired.certify(cc,cc['base_correct'],w,candidates);assert detail==second;off=detail['offset'];par=sel['selections'][method]['selected'];pol=dict(old=closed,new=0.,overall=closed*len(b)/len(ids),harmonic=0.) if off is None else evaluate(tc,off)
     correct=reft if off is None else tc['base_correct']&~risk.choose_new(tc,off);G=float(tc['weights']@(~reft&correct));assert G==0
     rows.append(dict(method=method,fixed_gate=evaluate(tc,par['offset']),capacity95=frontier(tc,.95*closed),policy=dict(**detail,**pol,test_target_met=pol['old']>=.95*closed-1e-10),self_reference_identical_to_common=True,G=G,D=float(tc['weights']@(reft&~correct))))
    for row in rows:row['matched_actual_base']={r['method']:frontier(systems['test',row['method']],r['policy']['old']) for r in rows}
    raw=dest.with_suffix('.npz');np.savez_compressed(raw,**archive);atomic(dict(dataset=ds,encoder=en,part=part,order=order,session=session,reference_closed=closed,rows=rows,source_archive_sha256=sha(src/f'o{order}_t{session}.npz'),archive_sha256=sha(raw)),dest)
   print('SHARED_EXPERT_ORDER',ds,en,part,order,flush=True)
  atomic(dict(complete=True),po/'complete.json');atomic(dict(dataset=ds,encoder=en,completed_partitions=part+1,parts=args.parts),OUT/'status.json')
 atomic(dict(complete=True,parts=args.parts),OUT/ds/en/'complete.json')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='sun');ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--parts',type=int,default=10);ap.add_argument('--device',default='cuda');a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(2026);torch.backends.cuda.matmul.allow_tf32=False
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.2)
 print('SHARED_EXPERT_START',a.dataset,a.device,flush=True)
 for en in ([a.encoder] if a.encoder else ['l14','b32']):run(a.dataset,en,a)
 atomic(dict(complete=True,parts=a.parts),OUT/'complete.json')
if __name__=='__main__':main()
