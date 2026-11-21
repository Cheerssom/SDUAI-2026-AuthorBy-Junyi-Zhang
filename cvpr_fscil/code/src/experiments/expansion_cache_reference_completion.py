from pathlib import Path
import sys,json,numpy as np,torch
sys.path.insert(0,str(Path(__file__).parent))
import expansion_sun_controls_remote as c
import kbs_risk_control as risk
import kbs_paired_accuracy as paired
R=Path(__file__).resolve().parents[2];O=R/'experiments/expansion_reference_completion';torch.set_num_threads(4);rows=[]
for en in ['l14','b32']:
 d=torch.load(R/f'experiments/expansion_prospective_validation/external/sun/{en}/frozen.pt',map_location='cpu',weights_only=False)
 for p in range(10):
  path=O/f'cache_{en}_p{p}.json'
  if path.exists():rows+=json.loads(path.read_text());continue
  folder=R/f'experiments/expansion_sun_controls_remote/download/{en}/p{p}';sel=json.loads((folder/'selection.json').read_text());source=R/f'experiments/expansion_prospective_validation/external/{en}/sun/catalog/p{p}';s=json.loads((source/'selection.json').read_text());b=np.array(s['entry']['base']);n=np.array(s['entry']['novel']);ids=np.r_[b,n];keys,ky=c.keys_for(d,b,n);counts=torch.tensor([(ky==cl).sum() for cl in ids]);systems={};ref=np.load(source/'o0_t5.npz');rr=[]
  for pool in ['tune','cal','test']:
   mask=np.isin(d[pool+'_y'],b if pool!='test' else ids);E=d['E_'+pool][mask];y=d[pool+'_y'][mask];text=100*(E@d['T'][ids].T)
   for method in ['cache_mean']:
    par=sel['selected'][method];v=c.cache(E,keys,ky,ids,par['beta']);systems[pool]=c.compact(text+par['alpha']*v/counts,ids,y,b,n)
  cc,tc=systems['cal'],systems['test'];w=risk.weights(d['cal_y'][np.isin(d['cal_y'],b)],b);ds=np.unique(np.quantile(systems['tune']['margin'],np.linspace(0,1,101)))
  for kind in ['common','self']:
   rc=ref['cal_reference_correct'] if kind=='common' else cc['base_correct'];rt=ref['test_reference_correct'] if kind=='common' else tc['base_correct'];closed=float(tc['weights']@rt);q=paired.certify(cc,rc,w,ds);off=q['offset'];metrics=dict(old=closed,new=0.) if off is None else c.evaluate(tc,off);rr.append(dict(dataset='sun',encoder=en,part=p,method='cache_mean',reference=kind,novel=100*metrics['new'],base=100*metrics['old'],closed_base=100*closed,finite=not q['fallback'],met=metrics['old']>=.95*closed-1e-10,policy=q))
  path.write_text(json.dumps(rr));rows+=rr;print('CACHE_REFERENCE',en,p,flush=True)
summary=[]
for en in ['l14','b32']:
 for kind in ['common','self']:
  z=[r for r in rows if r['encoder']==en and r['reference']==kind];summary.append(dict(dataset='sun',encoder=en,method='cache_mean',reference=kind,novel=float(np.mean([r['novel'] for r in z])),base=float(np.mean([r['base'] for r in z])),closed_base=float(np.mean([r['closed_base'] for r in z])),finite=2*sum(r['finite'] for r in z),met=2*sum(r['met'] for r in z),total=20))
(O/'cache_summary.json').write_text(json.dumps(dict(complete=True,scope='Independent CPU FP32 score reconstruction, fixed previously selected cache parameters; final vocabulary is identical across orders',summary=summary),indent=2));print('DONE')
