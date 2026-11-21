"""Frozen SUN archive replay; common Full, common Dense, and self contracts."""
import io,json,zipfile,hashlib,re
from pathlib import Path
import numpy as np
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
R=Path(__file__).resolve().parents[2]
O=R/'experiments/expansion_dense_reference_replay'
METHODS=['full','none','fixed','dense','logistic','mlp']
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def compact(z,p,m):
 y=z[p+'_labels'];b=z['base_classes'];n=z['novel_classes'];bm=np.isin(y,b);nm=np.isin(y,n)
 w=np.zeros(len(y))
 for group in [b,n]:
  for c in group:
   ix=y==c
   if ix.any():w[ix]=1/(len(group)*ix.sum())
 return dict(margin=z[p+'_'+m+'_margin'],base_prediction=z[p+'_'+m+'_base_prediction'],novel_prediction=z[p+'_'+m+'_novel_prediction'],base_correct=bm&(z[p+'_'+m+'_base_prediction']==y),novel_correct=nm&(z[p+'_'+m+'_novel_prediction']==y),weights=w,base_mask=bm)
def main():
 rows=[];verified=0
 for en in ['l14','b32']:
  path=R/f'experiments/expansion_sun_controls_remote/sun_{en}.zip'
  with zipfile.ZipFile(path) as archive:
   names=sorted(n for n in archive.namelist() if re.search(r'/p\d+/o\d_t\d\.npz$',n));assert len(names)==100
   for j,name in enumerate(names):
    p,o,t=map(int,re.search(r'/p(\d+)/o(\d)_t(\d)\.npz$',name).groups());dest=O/en/f'p{p}'/f'o{o}_t{t}.json'
    if dest.exists():
     saved=json.loads(dest.read_text()); rows+=saved['rows'];verified+=saved['verified'];continue
    raw=archive.read(name);z=dict(np.load(io.BytesIO(raw)));old=json.loads(archive.read(name.replace('.npz','.json')))
    cc={m:compact(z,'cal',m) for m in METHODS};tc={m:compact(z,'test',m) for m in METHODS};w=risk.weights(z['cal_labels'],z['base_classes']);out=[];v=0
    for m in METHODS:
     for ref in ['common_full','common_dense','self']:
      rc=z['cal_reference_correct'] if ref=='common_full' else cc['dense' if ref=='common_dense' else m]['base_correct']
      rt=z['test_reference_correct'] if ref=='common_full' else tc['dense' if ref=='common_dense' else m]['base_correct']
      q=paired.certify(cc[m],rc,w,z[m+'_candidates']);c=tc[m];closed=float(c['weights'][c['base_mask']]@rt[c['base_mask']]);d=q['offset']
      if d is None:base,novel=closed,0.
      else:
       ch=risk.choose_new(c,d);base=float(c['weights']@(c['base_correct']&~ch));novel=float(c['weights']@(c['novel_correct']&ch))
      if ref=='common_full':
       original=next(r['policy'] for r in old['records'] if r['method']==m)
       assert d==original['offset'] and abs(novel-original['new'])<1e-9;v+=1
      out.append(dict(encoder=en,part=p,order=o,session=t,method=m,reference=ref,closed_base=closed,target=.95*closed,base=base,novel=novel,met=base>=.95*closed-1e-10,**q))
    atomic(dest,dict(sha256=hashlib.sha256(raw).hexdigest(),verified=v,rows=out));rows+=out;verified+=v
    atomic(O/'status.json',dict(encoder=en,completed=j+1,total=100))
    if (j+1)%10==0:print(en,j+1,'/100',flush=True)
 summary=[];contrasts=[];rng=np.random.default_rng(20261105)
 for en in ['l14','b32']:
  for ref in ['common_full','common_dense','self']:
   for m in METHODS:
    a=[r for r in rows if r['encoder']==en and r['reference']==ref and r['method']==m and r['session']==5]
    summary.append(dict(encoder=en,reference=ref,method=m,total=len(a),finite=sum(not r['fallback'] for r in a),met=sum(r['met'] for r in a),**{k:100*float(np.mean([r[k] for r in a])) for k in ['base','novel','closed_base','target']}))
   for m in METHODS[1:]:
    def values(method):
     return np.array([np.mean([r['novel'] for r in rows if r['encoder']==en and r['reference']==ref and r['method']==method and r['session']==5 and r['part']==p]) for p in range(10)])
    delta=100*(values('full')-values(m));boot=delta[rng.integers(0,10,(20000,10))].mean(1)
    contrasts.append(dict(encoder=en,reference=ref,contrast='full - '+m,delta=float(delta.mean()),ci=np.quantile(boot,[.025,.975]).tolist()))
 atomic(O/'summary.json',dict(complete=True,records=len(rows),reproduced_common=verified,summary=summary,contrasts=contrasts))
 print('COMPLETE',len(rows),verified,flush=True)
if __name__=='__main__':main()
