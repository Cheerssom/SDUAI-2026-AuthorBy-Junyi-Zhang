"""Frozen SUN archive replay against deployed predictors independent of every readout.

References: (i) CLIP zero-shot over the base vocabulary, (ii) a standalone multinomial
logistic linear probe on base-fitting images (L2 chosen on base tune accuracy). Neither uses text-prototype fusion, standardization, or any
compared expansion readout. Certification, offsets, and test evaluation follow
expansion_dense_reference_replay.py exactly; only the paired reference changes.
Torch-free: frozen.pt is read with a minimal unpickler.
"""
import io,json,zipfile,hashlib,re,pickle,sys,os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
R=Path(__file__).resolve().parents[2]
O=R/'experiments/expansion_independent_reference_replay'
METHODS=['full','none','fixed','dense','logistic','mlp']
REFS=['common_full','zeroshot','probe']
L2S=(1e-2,3e-3,1e-3,3e-4,1e-4,3e-5)  # warm-started path, chosen on base tune macro accuracy
DT={'FloatStorage':np.float32,'HalfStorage':np.float16,'DoubleStorage':np.float64,'LongStorage':np.int64,'IntStorage':np.int32,'BoolStorage':np.bool_}
def load_pt(raw):
 inner=zipfile.ZipFile(io.BytesIO(raw));root=inner.namelist()[0].split('/')[0]
 class U(pickle.Unpickler):
  def find_class(self,mod,name):
   if mod=='torch._utils' and name=='_rebuild_tensor_v2':
    return lambda st,off,size,stride,*a:np.lib.stride_tricks.as_strided(st[off:],shape=size,strides=[s*st.itemsize for s in stride]).copy()
   if mod=='torch' and name in DT:return DT[name]
   if mod=='collections' and name=='OrderedDict':
    import collections;return collections.OrderedDict
   return super().find_class(mod,name)
  def persistent_load(self,pid):
   _,dtype,key,_,_=pid;return np.frombuffer(inner.read(f'{root}/data/{key}'),dtype=dtype)
 return U(io.BytesIO(inner.read(f'{root}/data.pkl'))).load()
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def compact(z,p,m):
 y=z[p+'_labels'];b=z['base_classes'];n=z['novel_classes'];bm=np.isin(y,b);nm=np.isin(y,n)
 w=np.zeros(len(y))
 for group in [b,n]:
  for c in group:
   ix=y==c
   if ix.any():w[ix]=1/(len(group)*ix.sum())
 return dict(margin=z[p+'_'+m+'_margin'],base_prediction=z[p+'_'+m+'_base_prediction'],novel_prediction=z[p+'_'+m+'_novel_prediction'],base_correct=bm&(z[p+'_'+m+'_base_prediction']==y),novel_correct=nm&(z[p+'_'+m+'_novel_prediction']==y),weights=w,base_mask=bm)
def macro(correct,y,b):
 return float(np.mean([correct[y==c].mean() for c in b]))
def references(d,b,ids):
 """Return {ref: (cal_correct, test_correct, tune_macro, chosen)} on the archive row order."""
 cm=np.isin(d['cal_y'],b);tm=np.isin(d['test_y'],ids);um=np.isin(d['tune_y'],b)
 Ec,yc=d['E_cal'][cm].astype(np.float64),d['cal_y'][cm];Et,yt=d['E_test'][tm].astype(np.float64),d['test_y'][tm];Eu,yu=d['E_tune'][um].astype(np.float64),d['tune_y'][um]
 T=d['T'][b].astype(np.float64);out={}
 pred=lambda E,W:b[(E@W).argmax(1)]
 out['zeroshot']=(pred(Ec,T.T)==yc,pred(Et,T.T)==yt,macro(pred(Eu,T.T)==yu,yu,b),None)
 fm=np.isin(d['fit_y'],b);bs=np.sort(b);mu=d['E_fit'][fm].astype(np.float64).mean(0)
 prep=lambda E:np.c_[20*(E.astype(np.float64)-mu),np.ones(len(E))]
 A=prep(d['E_fit'][fm]);yi=np.searchsorted(bs,d['fit_y'][fm]);n,k=len(A),len(b);Yo=np.eye(k)[yi];U=prep(Eu);W=np.zeros((A.shape[1],k));best=None
 for l2 in L2S:
  def f(v):
   V=v.reshape(-1,k);Z=A@V;Z-=Z.max(1,keepdims=True);P=np.exp(Z);P/=P.sum(1,keepdims=True)
   g=A.T@(P-Yo)/n;g[:-1]+=l2*V[:-1];return -np.log(P[np.arange(n),yi]+1e-300).mean()+.5*l2*(V[:-1]**2).sum(),g.ravel()
  W=minimize(f,W.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=1000)).x.reshape(-1,k)
  acc=macro(bs[(U@W).argmax(1)]==yu,yu,b)
  if best is None or acc>best[0]+1e-12:best=(acc,l2,W.copy())
 acc,l2,W=best;lp=lambda E:bs[(prep(E)@W).argmax(1)]
 out['probe']=(lp(Ec)==yc,lp(Et)==yt,acc,l2)
 return out,yc,yt
def main():
 rows=[];verified=0;refinfo=[]
 for en in os.environ.get('IR_ENCODERS','l14,b32').split(','):
  path=R/f'experiments/expansion_sun_controls_remote/sun_{en}.zip'
  with zipfile.ZipFile(path) as archive:
   d=load_pt(archive.read(f'inputs/{en}/frozen.pt'))
   for k in d:
    if hasattr(d[k],'shape') and d[k].dtype.kind in 'iu':d[k]=np.asarray(d[k])
   for part in [int(v) for v in os.environ.get('IR_PARTS',','.join(map(str,range(10)))).split(',')]:
    cp=O/'checkpoints'/f'{en}_p{part}.json'
    if cp.exists():
     saved=json.loads(cp.read_text());rows.extend(saved['rows']);refinfo.extend(saved['references']);verified+=saved['verified'];print(en,'part',part,'cached',flush=True);continue
    row_start=len(rows);ref_start=len(refinfo);verified_start=verified
    s=json.loads(archive.read(f'inputs/{en}/p{part}/selection.json'));b=np.asarray(s['entry']['base']);ids=np.r_[b,np.asarray(s['entry']['novel'])]
    refs,yc,yt=references(d,b,ids)
    refinfo+=[dict(encoder=en,part=part,reference=r,tune_macro=refs[r][2],lam=refs[r][3]) for r in refs]
    for name in sorted(n for n in archive.namelist() if re.search(rf'/p{part}/o\d_t\d\.npz$',n)):
     o,t=map(int,re.search(r'/o(\d)_t(\d)\.npz$',name).groups())
     raw=archive.read(name);z=dict(np.load(io.BytesIO(raw)));old=json.loads(archive.read(name.replace('.npz','.json')))
     assert np.array_equal(z['cal_labels'],yc) and np.array_equal(z['test_labels'],yt) and np.array_equal(z['base_classes'],b)
     cc={m:compact(z,'cal',m) for m in METHODS};tc={m:compact(z,'test',m) for m in METHODS};w=risk.weights(z['cal_labels'],z['base_classes'])
     for m in METHODS:
      for ref in REFS:
       rc,rt=(z['cal_reference_correct'],z['test_reference_correct']) if ref=='common_full' else refs[ref][:2]
       q=paired.certify(cc[m],rc,w,z[m+'_candidates']);c=tc[m];closed=float(c['weights'][c['base_mask']]@rt[c['base_mask']]);dd=q['offset']
       if dd is None:base,novel=closed,0.
       else:
        ch=risk.choose_new(c,dd);base=float(c['weights']@(c['base_correct']&~ch));novel=float(c['weights']@(c['novel_correct']&ch))
       if ref=='common_full':
        original=next(r['policy'] for r in old['records'] if r['method']==m)
        assert dd==original['offset'] and abs(novel-original['new'])<1e-9;verified+=1
       rows.append(dict(encoder=en,part=part,order=o,session=t,method=m,reference=ref,closed_base=closed,target=.95*closed,base=base,novel=novel,met=base>=.95*closed-1e-10,fallback=q['fallback'],offset=dd))
    atomic(cp,dict(rows=rows[row_start:],references=refinfo[ref_start:],verified=verified-verified_start))
    print(en,'part',part,'done',flush=True)
 if os.environ.get('IR_WORKER')=='1':return
 summary=[];contrasts=[];rng=np.random.default_rng(20261105)
 for en in ['l14','b32']:
  for ref in REFS:
   for m in METHODS:
    a=[r for r in rows if r['encoder']==en and r['reference']==ref and r['method']==m and r['session']==5]
    summary.append(dict(encoder=en,reference=ref,method=m,total=len(a),finite=sum(not r['fallback'] for r in a),met=sum(r['met'] for r in a),**{k:100*float(np.mean([r[k] for r in a])) for k in ['base','novel','closed_base','target']}))
   for m in METHODS[1:]:
    def values(method):
     return np.array([np.mean([r['novel'] for r in rows if r['encoder']==en and r['reference']==ref and r['method']==method and r['session']==5 and r['part']==p]) for p in range(10)])
    delta=100*(values('full')-values(m));boot=delta[rng.integers(0,10,(20000,10))].mean(1)
    contrasts.append(dict(encoder=en,reference=ref,contrast='full - '+m,delta=float(delta.mean()),ci=np.quantile(boot,[.025,.975]).tolist(),wins=int((delta>0).sum())))
 allmet={ref:[sum(r['met'] for r in rows if r['reference']==ref),sum(1 for r in rows if r['reference']==ref)] for ref in REFS}
 atomic(O/'summary.json',dict(complete=True,records=len(rows),reproduced_common=verified,met_all_sessions=allmet,references=refinfo,summary=summary,contrasts=contrasts))
 atomic(O/'rows.json',rows)
 print('COMPLETE',len(rows),verified,flush=True)
if __name__=='__main__':main()
