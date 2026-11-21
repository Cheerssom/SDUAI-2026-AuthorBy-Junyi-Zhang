from pathlib import Path
import subprocess,os,time,json,sys,csv,requests
root=Path(__file__).resolve().parent;work=root/'work';work.mkdir(exist_ok=True);data=Path('/root/autodl-tmp/data')
os.environ['HF_HOME']='/root/autodl-tmp/hf-cache';os.environ['OMP_NUM_THREADS']='8';os.environ['PYTHONUNBUFFERED']='1'
encs=['b16','l14','b32'];datasets=['cifar100','mini','cub200'];active={};failed={};start=time.time()
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
   if rc:failed[k]=rc
   print('EXIT',k,rc,flush=True)
 if (work/'PAUSE').exists() and not active:print('PAUSED',flush=True);break
 if not (work/'PAUSE').exists():
  for en in encs:
   for ds in datasets:
    key=f'features_{ds}_{en}';cache=work/'features'/ds/en/'verification.json'
    if cache.exists() or key in active or key in failed or not ready(ds):continue
    if len(active)<3 and sum(k.startswith('features_') for k in active)<2:launch(key,['fscil_features.py','--dataset',ds,'--encoder',en,'--root',str(roots[ds]),'--index_dir',str(data/'index_list'/idx[ds])])
   for ds in datasets:
    if not (work/'features'/ds/en/'verification.json').exists():continue
    for tr in ['standard','certified']:
     for seed in range(3):
      key=f'readout_{ds}_{en}_{tr}_{seed}'
      if (work/'readouts'/ds/en/tr/f'seed{seed}'/'complete.json').exists() or key in active or key in failed:continue
      if len(active)<3:launch(key,['fscil_readouts.py','--dataset',ds,'--encoder',en,'--track',tr,'--seed',str(seed)])
 done=list((work/'readouts').glob('*/*/*/seed*/complete.json'));features=list((work/'features').glob('*/*/verification.json'))
 status=dict(active={k:p.pid for k,(p,f) in active.items()},failed=failed,readouts_completed=len(done),features_completed=len(features),expected_readouts=54,expected_features=9,elapsed=time.time()-start)
 tmp=work/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(work/'queue_status.json')
 if len(done)==54 and not active:
  with (work/'summary.txt').open('w') as f:subprocess.run([sys.executable,'fscil_summary.py'],cwd=root,stdout=f,check=True)
  (work/'complete_all.json').write_text(json.dumps(status,indent=2));print('ALL_COMPLETE',flush=True);break
 if failed and not active and all(ready(ds) for ds in datasets):print('ERRORS_REQUIRE_FIX',failed,flush=True);break
 time.sleep(5)
