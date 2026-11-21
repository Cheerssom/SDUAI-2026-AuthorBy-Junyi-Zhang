"""True closed-vocabulary HyCal reference, fixed-time paired certification."""
from pathlib import Path
import json,math,hashlib,os,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
R=Path(__file__).resolve().parents[2];I=R/'inputs';O=R/'results';O.mkdir(exist_ok=True)
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def closed_predictions(a,pool):
 out=[]
 for key in ['cosine','negative_distance']:
  x=torch.from_numpy(a[pool+'_'+key]);sd=x.std(1,correction=1,keepdim=True).clamp_min(1e-6);z=(x-x.mean(1,keepdim=True))/sd;out.append(.5*(1+torch.erf(z/math.sqrt(2))))
 n=torch.as_tensor(a['counts'],dtype=out[0].dtype);w=torch.where(n>=5,torch.sigmoid(n-5),torch.zeros_like(n));s=(1-w[None])*out[0]+w[None]*out[1]
 # Sorted class ids resolve exact score ties canonically.
 return a['base_classes'][s.argmax(1).numpy()]
def compact(a,pool):
 y=a[pool+'_labels'];base=a['base_classes'];novel=a['novel_classes'];bm=np.isin(y,base);nm=np.isin(y,novel);w=np.zeros(len(y))
 for g in [base,novel]:
  for c in g:
   mask=y==c
   if mask.any():w[mask]=1/(len(g)*mask.sum())
 bp=a[pool+'_hycal_base_prediction'];npred=a[pool+'_hycal_novel_prediction']
 return dict(margin=a[pool+'_hycal_margin'],base_prediction=bp,novel_prediction=npred,base_correct=bm&(bp==y),novel_correct=nm&(npred==y),weights=w,base_mask=bm)
def main():
 torch.set_num_threads(2);files=sorted(I.glob('*/*/p*/o*_t*.npz'));assert len(files)==600;refs={};done=0
 for f in files:
  if (R/'PAUSE').exists():print('PAUSED',flush=True);return
  rel=f.relative_to(I);dest=O/rel.with_suffix('.json')
  if dest.exists():done+=1;continue
  folder=f.parent
  if folder not in refs:
   raw=folder/'closed_raw.npz'
   with np.load(raw) as z:a={k:z[k] for k in z.files}
   refs[folder]={pool:closed_predictions(a,pool)==a[pool+'_labels'] for pool in ['cal','test']}
   refs[folder]['input_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
  with np.load(f) as z:a={k:z[k] for k in z.files}
  cc,tc=compact(a,'cal'),compact(a,'test');w=risk.weights(a['cal_labels'],a['base_classes']);rows=[]
  for kind in ['common','self']:
   rc=a['cal_reference_correct'] if kind=='common' else refs[folder]['cal'];rt=a['test_reference_correct'] if kind=='common' else refs[folder]['test']
   closed=float(tc['weights'][tc['base_mask']]@rt[tc['base_mask']]);d=paired.certify(cc,rc,w,a['hycal_candidates']);off=d['offset']
   if off is None:base,novel=closed,0.;correct=rt
   else:
    new=risk.choose_new(tc,off);correct=tc['base_correct']&~new;base=float(tc['weights']@correct);novel=float(tc['weights']@(tc['novel_correct']&new))
   mask=tc['base_mask'];tw=tc['weights'][mask];destroyed=float(tw@(rt[mask]&~correct[mask]));recovered=float(tw@(~rt[mask]&correct[mask]))
   rows.append(dict(reference=kind,closed_base=closed,target=.95*closed,base=base,novel=novel,met=base>=.95*closed-1e-10,destroyed=destroyed,recovered=recovered,**d))
  atomic(dict(input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),closed_input_sha256=refs[folder]['input_sha256'],records=rows),dest);done+=1
  atomic(dict(completed=done,total=600,last=str(rel)),O/'status.json')
  if done%10==0:print('HYCAL_SELF',done,'/',600,rel,flush=True)
 atomic(dict(complete=True,session_files=600,policy_records=1200,closed_predictors=60),O/'complete.json');print('COMPLETE',flush=True)
if __name__=='__main__':main()
