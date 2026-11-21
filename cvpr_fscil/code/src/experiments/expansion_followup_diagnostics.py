"""Fixed-model coverage and descriptive new-domain diagnostics."""
from pathlib import Path
import sys,json,time,math,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import expansion_prospective_validation as p
import kbs_calibration_budget as budget
from kbs_runtime import atomic_json as save_json,check_pause,digest_file
def plain(v):
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v
def atomic_json(v,f):save_json(plain(v),f)
ROOT=p.ROOT;OUT=ROOT/'experiments/expansion_followup_diagnostics'
def loadz(f):
    with np.load(f) as z:return {k:z[k] for k in z.files}
def cm(z,m,pool):return budget.compact(z,m,z[pool+'_labels'],z['base_classes'],z['novel_classes'],pool+'_')
def coverage():
 for enc in ['l14','b32']:
  for B in [5,10,15,30]:
   for part in range(10):
    out=OUT/'coverage'/enc/f'B{B}'/f'p{part}.json'
    if out.exists():continue
    records=[];hashes={}
    for order in [0,1]:
     f=p.OUT/'cardinality'/enc/'imagenetr'/f'B{B}'/f'p{part}'/f'o{order}_t5.npz'
     z=loadz(f);hashes[str(f.relative_to(ROOT))]=digest_file(f)
     y=z['cal_labels'];b=z['base_classes'];yc=z['test_labels'];ref=z['test_reference_correct'];closed=np.mean([ref[yc==cl].mean() for cl in b])
     c=cm(z,'C30','cal');tc=cm(z,'C30','test');capacity=p.frontier(tc,.95*closed)
     for seed in range(10):
      rng=np.random.default_rng(41000+seed+100*part);ix={cl:rng.permutation(np.flatnonzero(y==cl)) for cl in b}
      for cap in sorted({5,10,20,40,150//B}):
       check_pause()
       if cap==150//B and cap not in [5,10,20,40] and min(map(len,ix.values()))<cap:continue
       ids=np.sort(np.concatenate([v[:cap] for v in ix.values()]))
       sub=p.gates.subcompact(c,np.isin(np.arange(len(y)),ids),y,b,z['novel_classes']);w=p.risk.weights(y[ids],b)
       result=p.fairness.evaluate_policy(sub,z['cal_reference_correct'][ids],w,z['C30_candidates'],tc,ref,closed,'paired_accuracy')
       # Direct-product witness for the first tested candidate.
       tr=result['trace'][0];correct=sub['base_correct']&~p.risk.choose_new(sub,tr['offset']);R=z['cal_reference_correct'][ids]
       logs=np.array([np.log1p(l*(w/w.max())*(correct.astype(float)-.95*R)).sum() for l in p.paired.LAMBDAS]);le=float(np.logaddexp.reduce(logs)-math.log(len(logs)))
       assert abs(le-tr['log_evalue'])<1e-9
       records.append(dict(encoder=enc,B=B,part=part,order=order,subset=seed,cap=cap,total=len(ids),equal_total150=len(ids)==150,capacity=capacity,policy=result))
    atomic_json(dict(inputs=hashes,records=records),out);print('COVERAGE',enc,B,part,flush=True)
def domain():
 for enc in ['l14','b32']:
  for ds in ['food','sun']:
   for part in range(10):
    out=OUT/'domain'/enc/ds/f'p{part}.json'
    if out.exists():continue
    rows=[];hashes={}
    for order in [0,1]:
     f=p.OUT/'external'/enc/ds/'catalog'/f'p{part}'/f'o{order}_t5.npz';z=loadz(f);hashes[str(f.relative_to(ROOT))]=digest_file(f)
     jp=json.loads(f.with_suffix('.json').read_text());pol={r['method']:r['policy'] for r in jp['records']};F=cm(z,'full','test');D=cm(z,'dense','test');target=pol['dense']['old'];fo=p.frontier(F,target);do=p.frontier(D,target)
     assert fo['feasible'] and do['feasible'];fd=fo['offset'];dd=do['offset']
     nf=p.risk.choose_new(F,fd);nd=p.risk.choose_new(D,dd)
     cf=np.where(nf,F['novel_correct'],F['base_correct']);cd=np.where(nd,D['novel_correct'],D['base_correct'])
     agree=F['base_prediction']==D['base_prediction']
     for group in ['base','novel']:
      mask=F[group+'_mask'];w=F['weights'][mask];assert np.isclose(w.sum(),1)
      for name,g in [('all',np.ones(len(mask),bool)),('base_winners_agree',agree),('base_winners_disagree',~agree)]:
       m=mask&g;ww=F['weights'][m];mass=float(ww.sum());den=mass or 1
       rows.append(dict(encoder=enc,dataset=ds,part=part,order=order,group=group,stratum=name,mass=mass,full_correct=float(ww@cf[m]/den),dense_correct=float(ww@cd[m]/den),correction=float(ww@(cf[m]&~cd[m])/den),destruction=float(ww@(~cf[m]&cd[m])/den),full_auc=p.auc(F),dense_auc=p.auc(D),target=float(target)))
    atomic_json(dict(inputs=hashes,records=rows),out);print('DOMAIN',enc,ds,part,flush=True)
def main():
 OUT.mkdir(exist_ok=True,parents=True);proto=ROOT/'experiments/expansion_followup_diagnostics_protocol.md';m=dict(sources={str(f.relative_to(ROOT)):digest_file(f) for f in [Path(__file__),proto]})
 mf=OUT/'manifest.json'
 if mf.exists():assert json.loads(mf.read_text())==m
 else:atomic_json(m,mf)
 coverage();domain();atomic_json(dict(complete=True),OUT/'prediction_diagnostics_complete.json')
if __name__=='__main__':main()
