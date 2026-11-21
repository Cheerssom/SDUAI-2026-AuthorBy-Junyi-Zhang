"""Remote fixed-readout own-reference certification; resumable CPU-only replay."""
from pathlib import Path
import json,os,hashlib,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
R=Path(__file__).resolve().parents[2];I=R/'inputs';O=R/'results';O.mkdir(exist_ok=True)
METHODS=['retained_query','retained_none','retained_fixed','dense_query','dense_none','dense_fixed']
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def compact(z,m,pool):
 y=z[pool+'_labels'];b=z['base_classes'];n=z['novel_classes'];bm=np.isin(y,b);nm=np.isin(y,n);w=np.zeros(len(y))
 for group in [b,n]:
  for c in group:
   ix=y==c
   if ix.any():w[ix]=1/(len(group)*ix.sum())
 bp=z[pool+'_'+m+'_base_prediction'];npred=z[pool+'_'+m+'_novel_prediction']
 return dict(margin=z[pool+'_'+m+'_margin'],base_prediction=bp,novel_prediction=npred,base_correct=bm&(bp==y),novel_correct=nm&(npred==y),weights=w,base_mask=bm)
def evaluate(c,d):
 new=risk.choose_new(c,d)
 return dict(base=float(c['weights']@(c['base_correct']&~new)),novel=float(c['weights']@(c['novel_correct']&new)))
def main():
 jobs=sorted(I.glob('*/*/p*/o*_t*.npz'));done=0
 assert len(jobs)==600
 for f in jobs:
  if (R/'PAUSE').exists():print('PAUSED at checkpoint',flush=True);return
  rel=f.relative_to(I);dest=O/rel.with_suffix('.json')
  if dest.exists():done+=1;continue
  enc,ds,part,name=rel.parts
  with np.load(f) as a:z={k:a[k] for k in a.files}
  rows=[];base=z['base_classes'];cy=z['cal_labels'];ty=z['test_labels'];w=risk.weights(cy,base)
  for m in METHODS:
   cc,tc=compact(z,m,'cal'),compact(z,m,'test')
   # Compare base-only winner to the same fitted readout at the first arrival.
   anchor=f.with_name(name[:2]+'_t1.npz')
   with np.load(anchor) as a:
    for pool in ['cal','test']:assert np.array_equal(z[pool+'_'+m+'_base_prediction'],a[pool+'_'+m+'_base_prediction'])
   for refkind in ['common','self']:
    rc=z['cal_reference_correct'] if refkind=='common' else cc['base_correct']
    rt=z['test_reference_correct'] if refkind=='common' else tc['base_correct']
    closed=float(tc['weights'][tc['base_mask']]@rt[tc['base_mask']]);detail=paired.certify(cc,rc,w,z[m+'_candidates']);d=detail['offset']
    metric=dict(base=closed,novel=0.) if d is None else evaluate(tc,d)
    rows.append(dict(method=m,reference=refkind,closed_base=closed,target=.95*closed,**metric,met=metric['base']>=.95*closed-1e-10,**detail))
  atomic(dict(input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),records=rows),dest);done+=1
  atomic(dict(stage='paired common/self reference',completed=done,total=len(jobs),last=str(rel)),O/'status.json')
  if done%10==0:print('SELF_REFERENCE',done,'/',len(jobs),str(rel),flush=True)
 atomic(dict(complete=True,completed=done,method_sessions=done*12),O/'complete.json');print('COMPLETE',done,flush=True)
if __name__=='__main__':main()
