"""Cloud verification and summaries; no result selection."""
from pathlib import Path
import numpy as np,json,math,hashlib
R=Path('/root/autodl-tmp/JLT-CUB');rows=[];checks=0
for enc in ['b32','l14']:
 for track in ['standard','certified']:
  for seed in ([0] if track=='standard' else [0,1,2]):
   root=R/'readouts_v2'/enc/track/f'seed{seed}';assert json.loads((root/'complete.json').read_text())['complete']
   sel=json.loads((root/'selection.json').read_text());fit=set(sel['fit_ids']);tune=set(sel['tune_ids']);cal=set(sel['cal_ids']);assert not cal&fit and not cal&tune
   if track=='certified':assert not fit&tune and len(cal)==1000 and len(fit)==1500
   for p in sorted(root.glob('session*.json')):
    for row in json.loads(p.read_text())['rows']:
     rows.append(dict(encoder=enc,track=track,seed=seed,**row))
     for kind in ['common','self']:
      if kind not in row:continue
      q=row[kind];accepted=None
      for i,t in enumerate(q['trace']):
       n=1000;h=t['destroyed']*n;g=t['recovered']*n;k=(n*t['accuracy_surplus']+.95*h-g)/.05
       assert max(abs(h-round(h)),abs(g-round(g)),abs(k-round(k)))<1e-7
       lam=np.array([.01,.025,.05,.1,.2,.3,.4,.5,.65,.8,.95,1.]);logs=h*np.log1p(-.95*lam)+g*np.log1p(lam)+k*np.log1p(.05*lam);v=np.exp(logs-logs.max()).mean();le=float(logs.max()+np.log(v));pv=math.exp(-max(0,le));assert abs(pv-t['pvalue'])<1e-9
       passed=pv<=.005;assert passed==t['accepted']
       if passed:accepted=t['offset']
       else:assert i==len(q['trace'])-1
       checks+=1
      assert q['offset']==accepted
summary=[]
for enc in ['b32','l14']:
 for track in ['standard','certified']:
  for method in sorted(set(r['method'] for r in rows)):
   rr=[r for r in rows if r['encoder']==enc and r['track']==track and r['method']==method];final=[r for r in rr if r['session']==10];x=dict(encoder=enc,track=track,method=method,n_seeds=len(final),aia_top1=100*float(np.mean([r['unconstrained_top1'] for r in rr])),final_top1=100*float(np.mean([r['unconstrained_top1'] for r in final])),aia_macro=100*float(np.mean([r['unconstrained_macro'] for r in rr])))
   if track=='certified':
    for kind in ['common','self']:
     a=[r[kind] for r in final];x[kind]=dict(novel=100*float(np.mean([v['novel'] for v in a])),base=100*float(np.mean([v['base'] for v in a])),closed=100*float(np.mean([v['closed_base'] for v in a])),finite=sum(not v['fallback'] for v in a),met=sum(v['met'] for v in a),oracle_feasible=sum(v['oracle']['feasible'] for v in a),oracle_novel=100*float(np.mean([v['oracle']['novel'] for v in a if v['oracle']['feasible']])) if any(v['oracle']['feasible'] for v in a) else None)
   summary.append(x)
report=dict(complete=True,source='matched frozen-CLIP feature transfer, not published backbone reproduction',records=len(rows),independently_verified_calibration_tests=checks,summary=summary)
(R/'readouts_summary.json').write_text(json.dumps(report,indent=2)+'\n');print('VERIFIED',len(rows),checks,flush=True)
# Shared expert study, exact stored correctness replay for accepted policies and independent e-values.
S=Path('/root/autodl-tmp/JLT-SharedHead');shared=[];sschecks=0
for p in sorted((S/'results/sun').glob('*/*/o*_t*.json')):
 d=json.loads(p.read_text());assert hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()==d['archive_sha256']
 with np.load(p.with_suffix('.npz')) as z:
  y=z['test_labels'];b=z['base_classes'];n=z['novel_classes'];w=np.zeros(len(y))
  for group in [b,n]:
   for c in group:w[y==c]=1/(len(group)*np.sum(y==c))
  ref=z['test_reference_correct']
  for a in d['rows']:
   assert a['G']==0 and a['self_reference_identical_to_common'];q=a['policy'];m=a['method'];dd=q['offset']
   if dd is not None:
    bp=z['test_'+m+'_base_prediction'];npred=z['test_'+m+'_novel_prediction'];mar=z['test_'+m+'_margin'];ch=(dd>mar)|((dd==mar)&(npred<bp));pred=np.where(ch,npred,bp);base=float(w@((pred==y)&np.isin(y,b)));novel=float(w@((pred==y)&np.isin(y,n)));assert abs(base-q['old'])<1e-9 and abs(novel-q['new'])<1e-9
   shared.append(dict(encoder=d['encoder'],part=d['part'],order=d['order'],session=d['session'],**a));sschecks+=1
pairs=[]
def ci(x):
 x=np.array(x);rng=np.random.default_rng(2026);v=x[rng.integers(len(x),size=(20000,len(x)))].mean(1);return dict(mean=float(x.mean()),ci95=np.quantile(v,[.025,.975]).tolist(),n=len(x))
for en in ['l14','b32']:
 diffs={k:[] for k in ['capacity','policy']}
 for part in range(10):
  z=[r for r in shared if r['encoder']==en and r['part']==part and r['session']==5]
  for k in diffs:
   v=lambda m:np.mean([r['capacity95']['new'] if k=='capacity' else r['policy']['new'] for r in z if r['method']==m])
   diffs[k].append(100*(v('full_gate')-v('prototype_gate')))
 pairs.append(dict(encoder=en,**{k:ci(x) for k,x in diffs.items()}))
(S/'summary.json').write_text(json.dumps(dict(complete=True,verified_method_sessions=sschecks,rows=shared,contrasts=pairs),indent=2)+'\n');print('SHARED_VERIFIED',sschecks,pairs)
