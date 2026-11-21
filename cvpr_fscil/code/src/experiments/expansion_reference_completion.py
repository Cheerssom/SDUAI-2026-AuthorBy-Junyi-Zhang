"""Authorized frozen-prediction reference sensitivity; no fitting or downloads."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
from expansion_hycal_self_replay import closed_predictions
R=Path(__file__).resolve().parents[2];O=R/'experiments/expansion_reference_completion';O.mkdir(exist_ok=True)
def compact(z,pool,m):
 y=z[pool+'_labels'];b=z['base_classes'];n=z['novel_classes'];bm=np.isin(y,b);nm=np.isin(y,n);w=np.zeros(len(y))
 for group in [b,n]:
  for c in group:
   ix=y==c
   if ix.any():w[ix]=1/(len(group)*ix.sum())
 bp=z[pool+'_'+m+'_base_prediction'];pn=z[pool+'_'+m+'_novel_prediction']
 return dict(margin=z[pool+'_'+m+'_margin'],base_prediction=bp,novel_prediction=pn,base_correct=bm&(bp==y),novel_correct=nm&(pn==y),weights=w,base_mask=bm)
def replay(z,pool,m,rc,rt):
 c=compact(z,pool,m);tc=compact(z,'test',m);w=risk.weights(z[pool+'_labels'],z['base_classes']);q=paired.certify(c,rc,w,z[m+'_candidates']);closed=float(tc['weights'][tc['base_mask']]@rt[tc['base_mask']]);off=q['offset']
 if off is None:a,n=closed,0.
 else:
  new=risk.choose_new(tc,off);a=float(tc['weights']@(tc['base_correct']&~new));n=float(tc['weights']@(tc['novel_correct']&new))
 return dict(closed_base=closed,base=a,novel=n,met=a>=.95*closed-1e-10,**q)
def main():
 torch.set_num_threads(2);rows=[];verified=0
 for ds in ['sun','food']:
  for en in ['l14','b32']:
   for p in range(10):
    folder=R/f'experiments/expansion_prospective_validation/external/{en}/{ds}/catalog/p{p}'
    for f in sorted(folder.glob('o*_t*.npz')):
     z=dict(np.load(f));old=json.loads(f.with_suffix('.json').read_text());dest=O/ds/en/f'p{p}'/f.with_suffix('.json').name;rr=[]
     for m in ['full','none','fixed','dense','logistic','mlp']:
      tc=compact(z,'test',m);cc=compact(z,'cal',m)
      for kind in ['common','self']:
       r=replay(z,'cal',m,z['cal_reference_correct'] if kind=='common' else cc['base_correct'],z['test_reference_correct'] if kind=='common' else tc['base_correct'])
       if kind=='common':
        q=next(v['policy'] for v in old['records'] if v['method']==m);assert r['offset']==q['offset'] and abs(r['novel']-q['new'])<1e-9;verified+=1
       rr.append(dict(dataset=ds,encoder=en,part=p,order=int(f.name[1]),session=int(f.name[4]),method=m,reference=kind,**r))
     dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(dict(input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),rows=rr)));rows+=rr
   print('REFERENCES',ds,en,flush=True)
 identity=json.loads((R/'experiments/kbs_sketch_validation/identity_split.json').read_text());budget=json.loads((R/'experiments/kbs_sketch_validation/calibration_budget.json').read_text());sy=np.array([r['class_id'] for r in json.loads((R/'experiments/kbs_calibration_budget/pools/imagenetr.json').read_text())['rows']])[budget['source_indices']];ay=np.array([r['class_id'] for r in identity['records']]);ti=np.array(budget['target_indices']);tests=np.array([i for i,r in enumerate(identity['records']) if r['role']=='test']);nt=len(json.loads((R/'experiments/kbs_independent_risk/splits/imagenetr.json').read_text())['indices']['tune']);nc=len(sy)
 for en in ['l14','b32']:
  folder=R/f'experiments/kbs_sketch_validation/evaluation/{en}';cache={}
  for p in range(10):
   first=folder/f'p{p}/o0_t1.npz';z0=dict(np.load(first));b=z0['base_classes'];reg=json.loads((R/f'experiments/kbs_reviewer_revision/readouts/{en}/imagenetr/p{p}/selection.json').read_text())['hycal']['regularization']
   if reg not in cache:cache[reg]=torch.load(folder/f'hycal_base_{reg:g}.pt',map_location='cpu',weights_only=False)
   raw=cache[reg];mask=np.isin(sy,b);tmask=np.isin(ay[ti],b);a=dict(base_classes=b,counts=raw['counts'][b])
   slices={'source_cal':np.arange(nt,nt+nc)[mask],'target_cal':(nt+nc+ti)[tmask],'test':nt+nc+tests}
   own={}
   for pool,ix in slices.items():
    aa=dict(base_classes=b,counts=raw['counts'][b])
    for key in ['cosine','negative_distance']:aa['cal_'+key]=raw[key][ix][:,b].numpy()
    own[pool]=closed_predictions(aa,'cal')==z0[pool+'_labels']
   for f in sorted((folder/f'p{p}').glob('o0_t*.npz')):
    z=dict(np.load(f));old=json.loads(f.with_suffix('.json').read_text());rr=[]
    for m in ['retained','anchored_tp','hycal']:
     tc=compact(z,'test',m)
     for pool in ['source_cal','target_cal']:
      cc=compact(z,pool,m)
      for kind in ['common','self']:
       rc=z[pool+'_reference_correct'] if kind=='common' else (own[pool] if m=='hycal' else cc['base_correct']);rt=z['test_reference_correct'] if kind=='common' else (own['test'] if m=='hycal' else tc['base_correct']);r=replay(z,pool,m,rc,rt)
       if kind=='common':
        q=next(v for v in old['records'] if v['method']==m and v.get('calibration')==pool and v['policy']=='paired_accuracy');assert r['offset']==q['offset'] and abs(r['novel']-q['new'])<1e-9;verified+=1
       rr.append(dict(dataset='sketch',encoder=en,part=p,order=0,session=int(f.name[4]),method=m,calibration=pool,reference=kind,**r))
    dest=O/'sketch'/en/f'p{p}'/f.with_suffix('.json').name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(dict(input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),rows=rr)));rows+=rr
  print('REFERENCES sketch',en,flush=True)
 summary=[]
 keys=sorted({(r['dataset'],r['encoder'],r['method'],r.get('calibration','cal'),r['reference']) for r in rows})
 for ds,en,m,cal,ref in keys:
  a=[r for r in rows if (r['dataset'],r['encoder'],r['method'],r.get('calibration','cal'),r['reference'],r['session'])==(ds,en,m,cal,ref,5)];summary.append(dict(dataset=ds,encoder=en,method=m,calibration=cal,reference=ref,novel=float(np.mean([r['novel'] for r in a]))*100,base=float(np.mean([r['base'] for r in a]))*100,closed_base=float(np.mean([r['closed_base'] for r in a]))*100,finite=sum(not r['fallback'] for r in a),met=sum(r['met'] for r in a),total=len(a)))
 (O/'summary.json').write_text(json.dumps(dict(complete=True,records=len(rows),reproduced_common_policies=verified,summary=summary),indent=2)+'\n');print('COMPLETE',len(rows),verified)
if __name__=='__main__':main()
