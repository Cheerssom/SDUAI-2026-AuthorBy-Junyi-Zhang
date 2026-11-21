"""Explicit fixed-cohort T-norm witness and matched frozen-archive comparison."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_reviewer_revision as previous
import kbs_full_fusion_retuning as tuning
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_tnorm_control';OUT.mkdir(exist_ok=True)
def tn(q,cohort):
    mean=cohort.mean(dim=1,keepdim=True)
    centered=cohort-mean
    sd=torch.sqrt(torch.mean(centered*centered,dim=1,keepdim=True)).clamp_min(1e-6)
    return (q-mean)/sd

def stat(x):
    x=np.asarray(x);rng=np.random.default_rng(2026);v=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(v,[.025,.975]).tolist(),n=len(x),values=x.tolist())

def main():
 torch.set_num_threads(2);rows=[];witness=[];hashes={}
 for enc in ['l14','b32']:
  for ds in ['pets','cifar','imagenetr']:
   vals={m:dict(capacity=[],novel=[],finite=[],met=[],base=[]) for m in ['primary','query']}
   for part in range(10):
    ordervals={m:[] for m in vals}
    sp=ROOT/f'experiments/expansion_cardinality_rule_v2/{enc}/{ds}/p{part}/selection.json';sel=json.loads(sp.read_text())
    qp=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p{part}/selection.json';qs=json.loads(qp.read_text())
    assert sel['settings']==qs['selections']['retained_query']['evaluated_settings']==3636
    for order in [0,1]:
     fp=sp.parent/f'o{order}_t5.json';data=json.loads(fp.read_text());hashes[str(fp.relative_to(ROOT))]=hashlib.sha256(fp.read_bytes()).hexdigest()
     for m in vals:
      rec=[r for r in data['records'] if r['method']==m]
      cap=next(r for r in rec if r['policy']=='matched_test' and r.get('retention')==.95)
      pol=next(r for r in rec if r['policy']=='paired_accuracy')
      assert cap['feasible']
      ordervals[m].append([100*cap['new'],100*pol['new'],int(not pol['fallback']),int(pol['test_target_met']),100*pol['old']])
    for m in vals:
     for k,v in zip(vals[m],np.mean(ordervals[m],axis=0)):vals[m][k].append(float(v))
   rows.append(dict(encoder=enc,dataset=ds,full={k:stat(v) for k,v in vals['primary'].items()},tnorm={k:stat(v) for k,v in vals['query'].items()},gain={k:stat(np.array(vals['primary'][k])-vals['query'][k]) for k in vals['primary']}))
   # Independent T-norm implementation on real cached queries, not a renamed function.
   d,_=previous.load_data(ds,enc);entry=json.loads((ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json').read_text())[0]
   orig=json.loads((ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p0/selection.json').read_text());qs=json.loads((ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p0/selection.json').read_text())
   base=np.array(entry['base']);novel=np.array(entry['novel']);nb=len(base);ids=np.r_[base,novel];si=np.array(orig['support_indices'])
   W=core.fit_ridge(d['E_fit'],d['fit_y'],base,orig['ridge_lambda']);Pb=core.means(d['E_fit'],d['fit_y'],base);Pn=core.means(d['E_support'][si],d['support_y'][si],novel)
   ix=np.linspace(0,len(d['E_test'])-1,128,dtype=int);E=d['E_test'][ix];T=d['T'][ids]
   qt=E@T.T;ql=core.linear(E,W);qp=E@torch.cat([Pb,Pn]).T
   z=[tn(qt,qt[:,:nb]),tn(ql,ql),tn(qp,qp[:,:nb])];par=qs['selections']['retained_query']['selected'];scores=z[0].clone();scores[:,:nb]+=par['a']*z[1];scores[:,nb:]+=par['u']*z[2][:,nb:]
   ap=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p0/o0_t5.npz'
   with np.load(ap) as a:
    margin=(scores[:,:nb].max(1).values.double()-scores[:,nb:].max(1).values.double()).numpy();err=float(np.max(np.abs(margin-a['test_retained_query_margin'][ix])))
    assert err<2e-4,(enc,ds,err)
    assert np.array_equal(base[scores[:,:nb].argmax(1)],a['test_retained_query_base_prediction'][ix])
    assert np.array_equal(novel[scores[:,nb:].argmax(1)],a['test_retained_query_novel_prediction'][ix])
   witness.append(dict(encoder=enc,dataset=ds,queries=128,maximum_margin_error=err,within_group_predictions_identical=True));print(enc,ds,'capacity gain',rows[-1]['gain']['capacity']['mean'],'policy gain',rows[-1]['gain']['novel']['mean'],flush=True)
   del d
 report=dict(complete=True,method='Traditional fixed base-cohort T-norm, population std and floor 1e-6; identical to Full-query eta=1; independently tuned with 3636 settings',scope='Frozen matched archive reuse, not new model training. Both orders averaged within each of ten partitions, final fifth arrival; same original common Ridge reference and cap40 independent calibration.',rows=rows,witness=witness,source_hashes=hashes)
 (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
