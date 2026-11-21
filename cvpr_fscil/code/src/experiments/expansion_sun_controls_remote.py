"""CPU-only SUN397 matched dense-cache and existing stacking control."""
from pathlib import Path
import json,hashlib,time,os,argparse
import numpy as np
import torch
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
R=Path(__file__).resolve().parents[2];OUT=R/'results';I=R/'inputs'
ALPHAS=(.1,1.,5.,10.,50.,100.);BETAS=(.1,1.,2.,5.,10.,20.);METHODS=('cache_sum','cache_mean')
def atomic(x,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pause():
 if (R/'PAUSE').exists():raise SystemExit('PAUSED at checkpoint')
def support(y,ids):
 rng=np.random.default_rng(7);return np.concatenate([rng.permutation(np.flatnonzero(y==c))[:5] for c in ids])
def compact(sc,ids,y,b,n):
 nb=len(b);bs=sc[:,:nb];ns=sc[:,nb:];ib=np.argsort(b);inn=np.argsort(n)
 bp=b[ib][bs[:,ib].argmax(1).cpu().numpy()];npred=n[inn][ns[:,inn].argmax(1).cpu().numpy()]
 margin=(bs.max(1).values.double()-ns.max(1).values.double()).cpu().numpy();bm=np.isin(y,b);nm=np.isin(y,n);w=np.zeros(len(y))
 for group in [b,n]:
  for c in group:
   ix=y==c
   if ix.any():w[ix]=1/(len(group)*ix.sum())
 return dict(margin=margin,base_prediction=bp,novel_prediction=npred,base_correct=bm&(bp==y),novel_correct=nm&(npred==y),weights=w,base_mask=bm,nb=len(b),nn=len(n))
def evaluate(c,d):
 new=risk.choose_new(c,d);ab=float(c['weights']@(c['base_correct']&~new));an=float(c['weights']@(c['novel_correct']&new))
 return dict(old=ab,new=an,overall=(ab*c['nb']+an*c['nn'])/(c['nb']+c['nn']),harmonic=2*ab*an/max(ab+an,1e-12))
def frontier(c,target):
 ix=np.flatnonzero(c['base_correct']);ceiling=float(c['weights'][ix].sum())
 if target>ceiling+1e-10:return dict(feasible=False,ceiling=ceiling)
 order=ix[np.argsort(-c['margin'][ix],kind='stable')];cum=np.cumsum(c['weights'][order]);j=min(int(np.searchsorted(cum,target-1e-12)),len(order)-1);d=float(c['margin'][order[j]])
 if evaluate(c,d)['old']<target-1e-10:d=float(np.nextafter(d,-np.inf))
 return dict(feasible=True,ceiling=ceiling,offset=d,**evaluate(c,d))
def cache(E,keys,ky,ids,beta):
 index=torch.tensor(np.searchsorted(np.sort(ids),ky),device=E.device);sortedids=np.sort(ids);perm=np.searchsorted(sortedids,ids)
 out=[]
 with torch.no_grad():
  for at in range(0,len(E),2048):
   aff=(beta*(E[at:at+2048]@keys.T-1)).exp();v=torch.zeros(len(aff),len(ids),device=E.device);v.scatter_add_(1,index[None].expand(len(aff),-1),aff);out.append(v[:,perm])
 return torch.cat(out)
def keys_for(d,b,n):
 m=np.isin(d['fit_y'],b);si=support(d['support_y'],n);return torch.cat([d['E_fit'][m],d['E_support'][si]]),np.r_[d['fit_y'][m],d['support_y'][si]]
def select(d,s):
 b=np.asarray(s['entry']['pseudo_base']);n=np.asarray(s['entry']['pseudo_novel']);ids=np.r_[b,n];mask=np.isin(d['tune_y'],np.r_[s['entry']['base']]);E=d['E_tune'][mask];y=d['tune_y'][mask];keys,ky=keys_for(d,b,n);counts=torch.tensor([(ky==cl).sum() for cl in ids],device=keys.device);text=100*(E@d['T'][ids].T);trials={m:[] for m in METHODS};best={m:(-1,None) for m in METHODS}
 for beta in BETAS:
  pause();v=cache(E,keys,ky,ids,beta)
  for method in METHODS:
   vc=v if method=='cache_sum' else v/counts
   for alpha in ALPHAS:
    c=compact(text+alpha*vc,ids,y,b,n);offsets=np.quantile(c['margin'],np.linspace(0,1,101));ch=(offsets[:,None]>c['margin'][None])|((offsets[:,None]==c['margin'][None])&(c['novel_prediction'][None]<c['base_prediction'][None]));ab=(~ch)@(c['weights']*c['base_correct']);an=ch@(c['weights']*c['novel_correct']);h=np.divide(2*ab*an,ab+an,out=np.zeros_like(ab),where=(ab+an)>0);j=int(h.argmax());p=dict(alpha=alpha,beta=beta,offset=float(offsets[j]));trials[method].append(dict(**p,offsets=offsets.tolist(),harmonics=h.tolist()))
    if h[j]>best[method][0]+1e-12:best[method]=(float(h[j]),dict(**p,harmonic=float(h[j])))
 return dict(selected={m:best[m][1] for m in METHODS},trials=trials,evaluated_settings=3636)
def run(en):
 d=torch.load(I/en/'frozen.pt',map_location='cpu',weights_only=False)
 for k,v in list(d.items()):
  if torch.is_tensor(v):d[k]=v.cuda()
 print('CUDA_FEATURES_READY',en,torch.cuda.get_device_name(),flush=True)
 for part in range(10):
  po=OUT/en/f'p{part}';po.mkdir(parents=True,exist_ok=True)
  if (po/'complete.json').exists():continue
  pause();src=I/en/f'p{part}';s=json.loads((src/'selection.json').read_text());assert s['selections']['dense']['evaluated_settings']==3636
  b=np.asarray(s['entry']['base']);n=np.asarray(s['entry']['novel']);ids=np.r_[b,n];sp=po/'selection.json'
  if sp.exists():sel=json.loads(sp.read_text())
  else:sel=select(d,s);sel['source_selection_sha256']=sha(src/'selection.json');atomic(sel,sp)
  keys,ky=keys_for(d,b,n);counts=torch.tensor([(ky==cl).sum() for cl in ids],device=keys.device);systems={};selectedbetas=sorted(set(v['beta'] for v in sel['selected'].values()))
  for pool in ['tune','cal','test']:
   mask=np.isin(d[pool+'_y'],b if pool!='test' else ids);E=d['E_'+pool][mask];y=d[pool+'_y'][mask];text=100*(E@d['T'][ids].T)
   for beta in selectedbetas:
    pause();v=cache(E,keys,ky,ids,beta)
    for method,p in sel['selected'].items():
     if p['beta']==beta:systems[pool,method]=(text+p['alpha']*(v if method=='cache_sum' else v/counts),y)
  for order in [0,1]:
   ordered=np.random.default_rng(1000+part+100*order).permutation(n)
   for session in range(1,6):
    dest=po/f'o{order}_t{session}.json'
    if dest.exists():continue
    pause();novel=ordered[:len(n)*session//5];active=np.r_[b,novel];cols=np.array([int(np.flatnonzero(ids==cl)[0]) for cl in active]);old=json.loads((src/f'o{order}_t{session}.json').read_text());archive=np.load(src/f'o{order}_t{session}.npz');refc=archive['cal_reference_correct'];reft=archive['test_reference_correct'];rows=[]
    for method,p in sel['selected'].items():
     c={}
     for pool in ['tune','cal','test']:
      scores,y=systems[pool,method];c[pool]=compact(scores[:,cols],active,y,b,novel)
     cc,tc=c['cal'],c['test'];candidates=np.unique(np.quantile(c['tune']['margin'],np.linspace(0,1,101)));w=risk.weights(systems['cal',method][1],b);detail=paired.certify(cc,refc,w,candidates);closed=old['reference_closed'];metric=dict(old=closed,new=0.,overall=closed*len(b)/len(active),harmonic=0.) if detail['offset'] is None else evaluate(tc,detail['offset'])
     matches={r['method']:frontier(tc,r['policy']['old']) for r in old['records'] if r['method'] in ['full','dense']}
     rows.append(dict(method=method,fixed_gate=evaluate(tc,p['offset']),capacity95=frontier(tc,.95*closed),matched_actual_base=matches,policy=dict(**detail,**metric,test_target_met=metric['old']>=.95*closed-1e-10)))
    atomic(dict(encoder=en,part=part,order=order,session=session,reference_closed=old['reference_closed'],source_sha256=sha(src/f'o{order}_t{session}.npz'),records=rows),dest)
  atomic(dict(complete=True),po/'complete.json');print('SUN_CACHE_COMPLETE',en,part,flush=True)
  atomic(dict(stage='SUN exponential cache',encoder=en,completed_partitions=part+1,total_partitions=20),OUT/'status.json')
 atomic(dict(complete=True),OUT/en/'complete.json')
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.42);torch.backends.cuda.matmul.allow_tf32=False
 ap=argparse.ArgumentParser();ap.add_argument('--encoder',choices=['l14','b32']);args=ap.parse_args()
 torch.manual_seed(2026);x=torch.randn(96,96,device='cuda');assert torch.isfinite(x@x.T).all();print('CUDA_WITNESS',torch.cuda.get_device_name(),flush=True)
 for en in ([args.encoder] if args.encoder else ['l14','b32']):run(en)
 atomic(dict(complete=True),OUT/'complete.json')
if __name__=='__main__':main()
