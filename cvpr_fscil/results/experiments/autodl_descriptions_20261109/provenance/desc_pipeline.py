"""Resume the fixed official-description experiment as original feature caches arrive."""
from pathlib import Path
import argparse,os,json,time,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parent;WORK=ROOT/'work'
MODELS={'b16':'base-patch16','b32':'base-patch32','l14':'large-patch14'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--max-active',type=int,default=2);a=p.parse_args()
 cap=a.max_active;assert 1<=cap<=4
 os.environ.update(HF_HOME='/root/autodl-tmp/hf-cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 encoders=[e for e in ['b32','b16','l14'] if e in os.environ.get('ENCODERS','b16 l14 b32').split()]
 pairs=[(d,e) for e in encoders for d in ['cifar100','mini','cub200']]
 active={};failed={};retries={};start=time.time()
 class Adopted:
  def __init__(self,pid,key):self.pid=pid;self.key=key
  def poll(self):
   proc=Path(f'/proc/{self.pid}')
   if proc.exists() and (proc/'stat').read_text().split()[2]!='Z':return None
   parts=self.key.split('_')
   if parts[0]=='encode':marker=WORK/'description_features'/parts[1]/(parts[2]+'.json')
   else:marker=WORK/'readouts_desc'/parts[1]/parts[2]/parts[3]/f'seed{parts[4]}'/'complete.json'
   return 0 if marker.exists() else 1
 previous=WORK/'desc_queue_status.json'
 if previous.exists():
  prior=json.loads(previous.read_text());failed=prior.get('failed',{});retries=prior.get('retries',{})
  for key,pid in prior.get('active',{}).items():
   if Path(f'/proc/{pid}').exists():active[key]=(Adopted(pid,key),(WORK/f'desc_{key}.log').open('a'));print('ADOPT',key,pid,flush=True)
 def memory_available():
  try:return int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])>=7000
  except Exception:return len(active)<2
 def launch(k,args):
  log=(WORK/f'desc_{k}.log').open('a');process=subprocess.Popen([sys.executable,'-u']+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);active[k]=(process,log);print('START',k,process.pid,flush=True)
 def encoded(ds,en):
  folder=WORK/'description_features'/ds;marker=folder/f'{en}.json';cache=folder/f'{en}.pt'
  if not marker.exists():return False
  value=json.loads(marker.read_text());assert value['complete'] and hashlib.sha256(cache.read_bytes()).hexdigest()==value['cache_sha256']
  assert hashlib.sha256((ROOT/'descriptions'/f'{ds}.json').read_bytes()).hexdigest()==value['descriptions_sha256']
  return True
 while True:
  for k,(process,log) in list(active.items()):
   rc=process.poll()
   if rc is None:continue
   log.close();del active[k];print('EXIT',k,rc,flush=True)
   if rc:
    tail=(WORK/f'desc_{k}.log').read_text(errors='replace')[-12000:]
    if 'CUDA out of memory' in tail and retries.get(k,0)<2:
     cap=1;retries[k]=retries.get(k,0)+1;print('OOM_BACKOFF',k,flush=True)
    else:failed[k]=rc
  for ds,en in pairs:
   if len(active)>=cap or (WORK/'PAUSE').exists():break
   if not (WORK/'features'/ds/en/'verification.json').exists():continue
   key=f'encode_{ds}_{en}'
   if not encoded(ds,en):
    model=Path('/root/autodl-tmp/hf-cache/hub')/('models--openai--clip-vit-'+MODELS[en])
    if model.exists() and key not in active and key not in failed and not any(k.startswith('encode_') for k in active):
     launch(key,['encode_descriptions.py','--dataset',ds,'--encoder',en])
    continue
   for track in ['standard','certified']:
    for seed in range(3):
     k=f'readout_{ds}_{en}_{track}_{seed}'
     if len(active)>=cap or not memory_available():break
     if (WORK/'readouts_desc'/ds/en/track/f'seed{seed}'/'complete.json').exists() or k in active or k in failed:continue
     launch(k,['fscil_readouts.py','--dataset',ds,'--encoder',en,'--track',track,'--seed',str(seed),'--text','desc','--tag','_desc'])
  done=list((WORK/'readouts_desc').glob('*/*/*/seed*/complete.json'))
  status=dict(expected=len(pairs)*6,complete=len(done),encoded=sum(encoded(d,e) for d,e in pairs),active={k:v[0].pid for k,v in active.items()},failed=failed,retries=retries,max_active=cap,waiting_features=[f'{d}/{e}' for d,e in pairs if not (WORK/'features'/d/e/'verification.json').exists()],elapsed=time.time()-start)
  tmp=WORK/'desc_queue_status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(WORK/'desc_queue_status.json')
  if len(done)==len(pairs)*6 and not active:
   with (WORK/'summary_desc.txt').open('w') as f:subprocess.run([sys.executable,'fscil_summary.py','_desc'],cwd=ROOT,stdout=f,check=True)
   for marker in done:
    task=marker.parent;ds=task.parts[-4];sessions=11 if ds=='cub200' else 9;files=sorted(task.glob('session*.json'));assert len(files)==sessions
    for i,file in enumerate(files):
     value=json.loads(file.read_text());assert value['session']==i and len(value['rows'])==11
    selection=json.loads((task/'selection.json').read_text());assert selection['settings_per_fusion']==3636
   (WORK/'desc_complete.json').write_text(json.dumps(status,indent=2));print('DESCRIPTION_EXPERIMENT_COMPLETE',len(done),flush=True);break
  if failed and not active:print('FAILED_REQUIRES_REVIEW',failed,flush=True)
  time.sleep(5)
if __name__=='__main__':main()
