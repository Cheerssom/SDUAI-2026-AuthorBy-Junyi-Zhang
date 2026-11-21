"""Verify and summarize common/self HyCal replay results with paired intervals."""
from pathlib import Path
import json,hashlib,zipfile
import numpy as np
R=Path(__file__).resolve().parents[2];H=R/'experiments/expansion_hycal_self_remote';D=H/'download'
def stats(x):
 x=np.asarray(x);rng=np.random.default_rng(2026);v=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
 return dict(mean=float(x.mean()),ci95=np.quantile(v,[.025,.975]).tolist(),values=x.tolist(),n=len(x))
def main():
 with zipfile.ZipFile(H/'results_bundle.zip') as z:assert z.testzip() is None;z.extractall(D)
 rows=[];checked=0
 for p in sorted((D/'results').glob('*/*/p*/o*_t*.json')):
  rel=p.relative_to(D/'results');en,ds,part,name=rel.parts;v=json.loads(p.read_text());oldpath=R/'experiments/kbs_reviewer_revision/readouts'/rel;old=json.loads(oldpath.read_text())['records']
  for a in v['records']:
   if a['reference']=='common':
    b=next(r for r in old if r['method']=='hycal' and r['policy']=='paired_accuracy')
    assert a['offset']==b['offset'] and abs(a['base']-b['old'])<1e-9 and abs(a['novel']-b['new'])<1e-9
    checked+=1
   rows.append(dict(encoder=en,dataset=ds,part=int(part[1:]),order=int(name[1]),session=int(name[4]),**a))
 assert checked==600
 out=[]
 for en in ['l14','b32']:
  for ds in ['cifar','pets','imagenetr']:
   for ref in ['common','self']:
    q=[r for r in rows if r['encoder']==en and r['dataset']==ds and r['reference']==ref and r['session']==5];assert len(q)==20
    vals={k:[100*np.mean([r[k] for r in q if r['part']==p]) for p in range(10)] for k in ['novel','base','closed_base','destroyed','recovered']}
    v=dict(encoder=en,dataset=ds,reference=ref,finite=sum(not r['fallback'] for r in q),met=sum(r['met'] for r in q),**{k:stats(x) for k,x in vals.items()});out.append(v)
    print(en,ds,ref,'novel',round(v['novel']['mean'],2),'finite',v['finite'],'met',v['met'],'closed',round(v['closed_base']['mean'],2),'G',round(v['recovered']['mean'],3))
 report=dict(complete=True,reconstructed_common_policies=checked,records=len(rows),summary=out)
 (H/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
