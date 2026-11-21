"""Summarize CIFAR-100 FSCIL readouts: standard-track session accuracies, AIA and PD; certified-track final-session admission."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'experiments/expansion_cifar_fscil'
def rows(enc,track,seed):
 p=D/'readouts'/enc/track/f'seed{seed}'
 return [r for f in sorted(p.glob('session*.json')) for r in json.loads(f.read_text())['rows']]
def main():
 out=dict(standard={},certified={})
 for enc in ['l14','b32']:
  seeds=[s for s in range(3) if (D/'readouts'/enc/'standard'/f'seed{s}'/'complete.json').exists()]
  if seeds:
   rr=[rows(enc,'standard',s) for s in seeds];methods=sorted({r['method'] for r in rr[0]})
   for m in methods:
    acc=np.array([[next(r['unconstrained_top1'] for r in x if r['method']==m and r['session']==t) for t in range(9)] for x in rr])*100
    mac=np.array([[next(r['unconstrained_macro'] for r in x if r['method']==m and r['session']==t) for t in range(9)] for x in rr])*100
    out['standard'][f'{enc}/{m}']=dict(seeds=seeds,session_top1=acc.mean(0).round(2).tolist(),AIA=float(acc.mean()),last=float(acc[:,-1].mean()),PD=float((acc[:,0]-acc[:,-1]).mean()),AIA_macro=float(mac.mean()))
  seeds=[s for s in range(3) if (D/'readouts'/enc/'certified'/f'seed{s}'/'complete.json').exists()]
  if seeds:
   rr=[rows(enc,'certified',s) for s in seeds];methods=sorted({r['method'] for r in rr[0]})
   for m in methods:
    final=[next(r for r in x if r['method']==m and r['session']==8) for x in rr]
    out['certified'][f'{enc}/{m}']={k:dict(novel=float(np.mean([f[k]['novel'] for f in final])*100),base=float(np.mean([f[k]['base'] for f in final])*100),closed=float(np.mean([f[k]['closed_base'] for f in final])*100),finite=sum(f[k]['offset'] is not None for f in final),met=sum(bool(f[k]['met']) for f in final),total=len(final)) for k in ['common','self']}
 (D/'summary.json').write_text(json.dumps(out,indent=2))
 for k,v in out['standard'].items():print('STANDARD',k,'AIA %.2f last %.2f PD %.2f'%(v['AIA'],v['last'],v['PD']))
 for k,v in out['certified'].items():print('CERTIFIED',k,'common %.2f (%d/%d)'%(v['common']['novel'],v['common']['finite'],v['common']['total']),'self %.2f (%d/%d)'%(v['self']['novel'],v['self']['finite'],v['self']['total']))
if __name__=='__main__':main()
