from pathlib import Path
import subprocess,os,time,json,sys,csv,requests
root=Path(__file__).resolve().parent;work=root/'work';work.mkdir(exist_ok=True);data=Path('/root/autodl-tmp/data')
os.environ['HF_HOME']='/root/autodl-tmp/hf-cache';os.environ['OMP_NUM_THREADS']='8';os.environ['PYTHONUNBUFFERED']='1'
encs=['b16','l14','b32'];datasets=['cifar100','mini','cub200'];active={};failed={};start=time.time()
assignment=json.loads((root/'node_assignment.json').read_text());encs=assignment.get('encoder_priority',encs);max_active=assignment.get('max_active',2);external_features={tuple(v) for v in assignment.get('external_feature_pairs',[])};allowed={tuple(v) for v in assignment['pairs']};feature_allowed=allowed|{tuple(v) for v in assignment.get('feature_only_pairs',[])};expected_readouts=len(allowed)*6;expected_features=len(feature_allowed)
retry_path=work/'oom_retry_state.json';retry_state=json.loads(retry_path.read_text()) if retry_path.exists() else {'attempts':{},'max_active':max_active};max_active=min(max_active,retry_state['max_active'])
roots={'cifar100':data/'cifar-100-python','mini':data/'miniimagenet','cub200':data/'cub'};idx={'cifar100':'cifar100','mini':'mini_imagenet','cub200':'cub200'}
def ready(ds):
 if ds=='cifar100':return (roots[ds]/'train').exists()
 if ds=='cub200':return (work/'cub_parallel.log').exists() and 'CUB_READY' in (work/'cub_parallel.log').read_text()
 if not (data/'MINI_READY').exists():return False
 if not (roots[ds]/'split/train.csv').exists():return False
 if not (roots[ds]/'class_names.txt').exists():
  r=requests.get('https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json',timeout=30);r.raise_for_status();names={v[0]:v[1].replace('_',' ') for v in r.json().values()}
  with (roots[ds]/'split/train.csv').open() as f:wnids=sorted({r[1] for r in list(csv.reader(f))[1:]})
  assert len(wnids)==100 and all(w in names for w in wnids)
  (roots[ds]/'class_names.txt').write_text('\n'.join(w+' '+names[w] for w in wnids))
 return True
class AdoptedProcess:
 def __init__(self,pid,key):self.pid=pid;self.key=key
 def poll(self):
  if Path(f'/proc/{self.pid}').exists():return None
  parts=self.key.split('_')
  if parts[0]=='features':done=work/'features'/parts[1]/parts[2]/'verification.json'
  else:done=work/'readouts'/parts[1]/parts[2]/parts[3]/f'seed{parts[4]}'/'complete.json'
  return 0 if done.exists() else 1
if (work/'queue_status.json').exists():
 previous=json.loads((work/'queue_status.json').read_text())
 for key,pid in previous.get('active',{}).items():
  if Path(f'/proc/{pid}').exists():active[key]=(AdoptedProcess(pid,key),(work/('task_'+key+'.log')).open('a'));print('ADOPT',key,pid,flush=True)
def launch(key,args):
 f=(work/('task_'+key+'.log')).open('a');p=subprocess.Popen([sys.executable,'-u']+args,cwd=root,stdout=f,stderr=subprocess.STDOUT);active[key]=(p,f);print('START',key,p.pid,flush=True)
while True:
 for k,(p,f) in list(active.items()):
  rc=p.poll()
  if rc is not None:
   f.close();del active[k]
   if rc:
    oom='CUDA out of memory' in (work/('task_'+k+'.log')).read_text(errors='replace')[-10000:]
    count=retry_state['attempts'].get(k,0)
    if oom and count<2:
     max_active=1;retry_state['max_active']=1;retry_state['attempts'][k]=count+1;retry_path.write_text(json.dumps(retry_state,indent=2));print('OOM_BACKOFF_CHECKPOINT_RETRY',k,count+1,'max_active=1',flush=True)
    else:failed[k]=rc
   print('EXIT',k,rc,flush=True)
 if (work/'PAUSE').exists() and not active:print('PAUSED',flush=True);break
 if not (work/'PAUSE').exists():
  for en in encs:
   for ds in datasets:
    if (ds,en) not in feature_allowed or (ds,en) in external_features:continue
    key=f'features_{ds}_{en}';cache=work/'features'/ds/en/'verification.json'
    if cache.exists() or key in active or key in failed or not ready(ds):continue
    if len(active)<max_active and sum(k.startswith('features_') for k in active)<2:launch(key,['fscil_features.py','--dataset',ds,'--encoder',en,'--root',str(roots[ds]),'--index_dir',str(data/'index_list'/idx[ds])])
   for ds in datasets:
    if (ds,en) not in allowed:continue
    if not (work/'features'/ds/en/'verification.json').exists():continue
    for tr in ['standard','certified']:
     for seed in range(3):
      key=f'readout_{ds}_{en}_{tr}_{seed}'
      if (work/'readouts'/ds/en/tr/f'seed{seed}'/'complete.json').exists() or key in active or key in failed:continue
      if len(active)<max_active:launch(key,['fscil_readouts.py','--dataset',ds,'--encoder',en,'--track',tr,'--seed',str(seed)])
 done=list((work/'readouts').glob('*/*/*/seed*/complete.json'));features=list((work/'features').glob('*/*/verification.json'))
 status=dict(active={k:p.pid for k,(p,f) in active.items()},failed=failed,readouts_completed=len(done),features_completed=len(features),expected_readouts=expected_readouts,expected_features=expected_features,node=assignment['node'],pairs=assignment['pairs'],elapsed=time.time()-start)
 tmp=work/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(work/'queue_status.json')
 if len(done)==expected_readouts and not active:
  (work/'complete_node.json').write_text(json.dumps(status,indent=2));print('NODE_COMPLETE',flush=True);break
 if failed and not active and all(ready(ds) for ds,en in allowed):print('ERRORS_REQUIRE_FIX',failed,flush=True);break
 time.sleep(5)
