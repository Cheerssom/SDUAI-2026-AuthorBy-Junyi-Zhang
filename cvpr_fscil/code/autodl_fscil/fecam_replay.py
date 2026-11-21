"""CPU-only FeCAM covariance-transfer replay from immutable archived selections."""
from pathlib import Path
import json,hashlib,shutil,os,time
import numpy as np
import torch
import fscil_readouts as f

METHOD='FeCAM_transfer'
VERSION='regularized_all_class_covariance_v2_cpu_replay'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def others(rows):return [r for r in rows if r['method']!=METHOD]

@torch.inference_mode()
def replay(ds,enc,track,seed):
 start=time.monotonic();torch.set_num_threads(int(os.environ.get('FECAM_CPU_THREADS','4')))
 nb,way,sessions,ntune,ncal,_=f.CFG[ds];f.NB=nb
 key=Path(ds)/enc/track/f'seed{seed}';out=f.D/'readouts'/key
 assert (out/'complete.json').exists(),f'Missing original completed task: {key}'
 backup=f.D/'fecam_v1_backups'/key
 if not backup.exists():
  tmp=backup.with_name(backup.name+'.pending');tmp.parent.mkdir(parents=True,exist_ok=True)
  if tmp.exists():raise RuntimeError(f'Incomplete backup requires review: {tmp}')
  shutil.copytree(out,tmp);tmp.rename(backup)
 immutable={n:digest(backup/n) for n in ['selection.json','complete.json']}
 selection=json.loads((backup/'selection.json').read_text())
 feature=f.D/'features'/ds/enc/'frozen.pt';feature_hash=digest(feature)
 assert feature_hash==selection['features_sha256']
 marker=out/'fecam_patch2.json'
 if marker.exists():
  old=json.loads(marker.read_text());assert old['version']==VERSION and old['feature_sha256']==feature_hash
  for n,h in immutable.items():assert digest(out/n)==h
  for si in range(sessions):
   name=f'session{si:02d}.json';a=json.loads((backup/name).read_text());b=json.loads((out/name).read_text())
   assert canon(others(a['rows']))==canon(others(b['rows']))
  prior=dict(np.load(backup/'closed_references.npz'));current=dict(np.load(out/'closed_references.npz'))
  assert set(prior)==set(current)
  assert all(np.array_equal(prior[k],current[k]) for k in prior if not k.startswith(METHOD+'_'))
  print('FECAM_ALREADY_VERIFIED',str(key),flush=True);return
 stage=f.D/'fecam_replay2_staging'/key;stage.mkdir(parents=True,exist_ok=True)
 z=torch.load(feature,map_location='cpu',weights_only=False);E=z['E'];y=np.asarray(z['y']);iid=np.asarray(z['ids']);base=np.arange(nb)
 assert E.device.type=='cpu' and not torch.cuda.is_initialized()
 index={int(v):i for i,v in enumerate(iid)}
 fit=np.array([index[int(v)] for v in selection['fit_ids']]);tune=np.array([index[int(v)] for v in selection['tune_ids']])
 # Standard-track archives omit calibration IDs, but the original per-class RNG split is fixed.
 rng=np.random.default_rng(28012026+seed);official=np.array([index[v] for v in z['sessions']['1']]);tune_check=[];cal=[];fit_check=[]
 for k in base:
  q=rng.permutation(official[y[official]==k]);tune_check.extend(q[:ntune]);cal.extend(q[ntune:ntune+ncal]);fit_check.extend(q[ntune+ncal:])
 cal=np.array(cal);assert np.array_equal(tune,np.array(tune_check))
 assert np.array_equal(fit,np.array(fit_check) if track=='certified' else official)
 if track=='certified':assert np.array_equal(iid[cal],np.asarray(selection['cal_ids']))
 original_refs=dict(np.load(backup/'closed_references.npz'));refs=dict(np.load(stage/'closed_references.npz')) if (stage/'closed_references.npz').exists() else dict(original_refs);refc=original_refs['full_cal'];test=np.flatnonzero(~z['train']);bt=test[y[test]<nb]
 closed=selection['shared_reference_closed_test'];assert abs(f.macro(refs['full_base_test'],y[bt],base)-closed)<1e-12
 assert len(refc)==len(cal)
 trainidx=list(fit);cache={};all_new=[]
 for si in range(sessions):
  if (f.D/'PAUSE').exists():raise RuntimeError('PAUSED before FeCAM session')
  if si:trainidx.extend(index[v] for v in z['sessions'][str(si+1)])
  dest=stage/f'session{si:02d}.json';original=json.loads((backup/dest.name).read_text())
  if dest.exists():
   saved=json.loads(dest.read_text());assert canon(others(saved['rows']))==canon(others(original['rows']));all_new.extend(saved['rows']);continue
  novel=np.arange(nb,nb+way*si);ids=np.r_[base,novel];active=test[y[test]<len(ids)];queries=np.r_[tune,cal,active];train=np.asarray(trainidx)
  score=f.fecam_cpu(E[queries],E[train],y[train],ids,cache)
  assert score.device.type=='cpu' and bool(torch.isfinite(score).all())
  n=len(tune);c=len(cal);pred=f.prediction(score[n+c:],ids);ya=y[active];br=ya<nb
  row=dict(method=METHOD,session=si,unconstrained_macro=f.macro(pred==ya,ya,ids),unconstrained_top1=float(np.mean(pred==ya)),base=f.macro(pred[br]==ya[br],ya[br],base))
  if si==0:
   # This revision changes the base predictor as well: refresh only FeCAM's own reference.
   refs[METHOD+'_cal']=(f.prediction(score[n:n+c],base)==y[cal])
   refs[METHOD+'_base_test']=(pred==ya)
   np.savez_compressed(stage/'closed_references.npz',**refs)
  elif track=='certified':
   vc=f.comp(score[:n],ids,y[tune],base,novel);cc=f.comp(score[n:n+c],ids,y[cal],base,novel);tc=f.comp(score[n+c:],ids,ya,base,novel)
   candidates=np.unique(np.quantile(vc['margin'],np.linspace(0,1,101)));weights=f.risk.weights(y[cal],base)
   for kind in ['common','self']:
    reference=refc if kind=='common' else refs[METHOD+'_cal']
    floor=closed if kind=='common' else f.macro(refs[METHOD+'_base_test'],y[bt],base)
    detail=f.paired.certify(cc,reference,weights,candidates,sessions=sessions-1);offset=detail['offset']
    metrics=dict(base=floor,novel=0.) if offset is None else f.evaluate(tc,offset)
    row[kind]=dict(**detail,**metrics,closed_base=floor,met=metrics['base']>=.95*floor-1e-10,oracle=f.frontier(tc,.95*floor))
  patched=dict(original);patched['rows']=[row if old['method']==METHOD else old for old in original['rows']]
  assert len(patched['rows'])==11 and canon(others(patched['rows']))==canon(others(original['rows']))
  f.atomic(patched,dest);all_new.extend(patched['rows'])
  print('FECAM_CPU_SESSION',str(key),si,'/',sessions-1,round(time.monotonic()-start,1),flush=True)
 assert (stage/'closed_references.npz').exists()
 assert set(refs)==set(original_refs)
 assert all(np.array_equal(refs[k],original_refs[k]) for k in refs if not k.startswith(METHOD+'_'))
 # Commit staged rows only after a full task succeeds; original archive remains available.
 for si in range(sessions):
  dest=stage/f'session{si:02d}.json';f.atomic(json.loads(dest.read_text()),out/dest.name)
 ref_tmp=out/'closed_references.replay2.tmp.npz';np.savez_compressed(ref_tmp,**refs);ref_tmp.replace(out/'closed_references.npz')
 for n,h in immutable.items():assert digest(out/n)==h
 assert not torch.cuda.is_initialized()
 f.atomic(dict(version=VERSION,feature_sha256=feature_hash,other_methods_unchanged=True,other_reference_arrays_unchanged=True,feCAM_reference_updated=True,immutable_files_sha256=immutable,sessions=sessions,cuda_initialized=False,seconds=time.monotonic()-start),marker)
 print('FECAM_CPU_COMPLETE',str(key),round(time.monotonic()-start,1),flush=True)
