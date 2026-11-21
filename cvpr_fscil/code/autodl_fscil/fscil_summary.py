"""Summaries in the field's format: session-wise top-1, Avg, Last, PD per dataset/encoder/method,
an Average over the three benchmarks, and final-arrival certified novel accuracy (common/self)."""
from pathlib import Path
import json,sys
TAG=sys.argv[1] if len(sys.argv)>1 else ''
import numpy as np
R=Path(__file__).resolve().parent;D=R/'work'
S={'cifar100':9,'mini':9,'cub200':11}
def rows(ds,enc,track,seed):
 p=D/f'readouts{TAG}'/ds/enc/track/f'seed{seed}'
 x=D/'readouts_extra'/ds/enc/track/f'seed{seed}'  # GDA/KLDA (extra_baselines.py), names namespace only
 # four-source AnchorCE (anchor_gda.py) and its ablations; description runs live in readouts_anchorg_desc
 if not TAG: extra=[x]+[D/f'readouts_anchorg{v}'/ds/enc/track/f'seed{seed}' for v in ('','_none','_fixed','_rawg')]
 elif TAG=='_ref4': extra=[D/v/ds/enc/track/f'seed{seed}' for v in ('readouts_extra_ref4','readouts_anchorg_ref4')]  # all certified against the four-source AnchorCE closed head
 else: extra=[D/f'readouts_anchorg{TAG}'/ds/enc/track/f'seed{seed}']
 return [r for q in [p]+[v for v in extra if (v/'complete.json').exists()] for f in sorted(q.glob('session*.json')) for r in json.loads(f.read_text())['rows']]
def done(ds,enc,track):return [s for s in range(3) if (D/f'readouts{TAG}'/ds/enc/track/f'seed{s}'/'complete.json').exists()]
def main():
 out=dict(standard={},certified={},average={})
 encs=sorted({p.name for p in (D/f'readouts{TAG}').glob('*/*') if p.is_dir()})
 for enc in encs:
  for ds in S:
   seeds=done(ds,enc,'standard')
   if seeds:
    rr=[rows(ds,enc,'standard',s) for s in seeds]
    for m in sorted({r['method'] for r in rr[0]}):
     acc=np.array([[next(r['unconstrained_top1'] for r in x if r['method']==m and r['session']==t) for t in range(S[ds])] for x in rr])*100
     out['standard'][f'{ds}/{enc}/{m}']=dict(seeds=seeds,session_top1=acc.mean(0).round(2).tolist(),Avg=float(acc.mean()),Last=float(acc[:,-1].mean()),PD=float((acc[:,0]-acc[:,-1]).mean()))
   seeds=done(ds,enc,'certified')
   if seeds:
    rr=[rows(ds,enc,'certified',s) for s in seeds]
    for m in sorted({r['method'] for r in rr[0]}):
     fin=[next(r for r in x if r['method']==m and r['session']==S[ds]-1) for x in rr]
     out['certified'][f'{ds}/{enc}/{m}']={k:dict(novel=float(np.mean([f[k]['novel'] for f in fin])*100),base=float(np.mean([f[k]['base'] for f in fin])*100),closed=float(np.mean([f[k]['closed_base'] for f in fin])*100),finite=sum(f[k]['offset'] is not None for f in fin),met=sum(bool(f[k]['met']) for f in fin),total=len(fin)) for k in ['common','self']}
  # Average over the three benchmarks (only when all three are complete for that method)
  methods={k.split('/')[2] for k in out['standard'] if k.split('/')[1]==enc}
  for m in sorted(methods):
   keys=[f'{ds}/{enc}/{m}' for ds in S]
   if all(k in out['standard'] for k in keys):
    v=[out['standard'][k] for k in keys]
    out['average'][f'{enc}/{m}']=dict(Avg=float(np.mean([x['Avg'] for x in v])),Last=float(np.mean([x['Last'] for x in v])),PD=float(np.mean([x['PD'] for x in v])))
    if all(k in out['certified'] for k in keys):out['average'][f'{enc}/{m}']['certified_common']=float(np.mean([out['certified'][k]['common']['novel'] for k in keys]))
 (D/f'summary{TAG}.json').write_text(json.dumps(out,indent=2))
 for k,v in out['standard'].items():print('STANDARD %-45s Avg %.2f Last %.2f PD %.2f'%(k,v['Avg'],v['Last'],v['PD']))
 for k,v in out['certified'].items():print('CERTIFIED %-44s common %.2f (%d/%d) self %.2f (%d/%d)'%(k,v['common']['novel'],v['common']['finite'],v['common']['total'],v['self']['novel'],v['self']['finite'],v['self']['total']))
 for k,v in out['average'].items():print('AVERAGE %-46s'%k,{a:round(b,2) for a,b in v.items()})
if __name__=='__main__':main()
