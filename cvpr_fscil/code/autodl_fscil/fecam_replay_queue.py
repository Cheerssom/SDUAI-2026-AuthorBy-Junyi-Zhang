"""Bounded CPU-only replays of existing completed tasks; no new experiment cells."""
from pathlib import Path
import argparse,os,json,time,subprocess,sys,concurrent.futures
ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 assert 1<=a.workers<=4
 encoders=set(os.environ.get('ENCODERS','b16 l14 b32').split())
 tasks=[]
 for path in sorted((WORK/'readouts').glob('*/*/*/seed*/complete.json')):
  ds,enc,track,seed=path.parent.relative_to(WORK/'readouts').parts
  if enc in encoders:tasks.append((ds,enc,track,int(seed[4:])))
 assert len(tasks)==len(set(tasks)) and tasks
 start=time.monotonic();finished=[];failed={};active={}
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 def worker(task):
  ds,enc,track,seed=task;name=f'{ds}_{enc}_{track}_{seed}';log=WORK/f'fecam_replay2_{name}.log'
  command=[sys.executable,'-u',str(ROOT/'fscil_readouts.py'),'--dataset',ds,'--encoder',enc,'--track',track,'--seed',str(seed),'--only','FeCAM_transfer']
  active[name]=True
  with log.open('a') as f:result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  active.pop(name,None)
  if result.returncode:raise RuntimeError(f'{name}: exit {result.returncode}; see {log}')
  return name
 def status():
  s=dict(expected=len(tasks),complete=len(finished),active=list(active),failed=failed,elapsed=time.monotonic()-start)
  q=WORK/'fecam_replay2_status.tmp';q.write_text(json.dumps(s,indent=2));q.replace(WORK/'fecam_replay2_status.json')
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(worker,t):t for t in tasks};pending=set(futures)
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in done:
    try:finished.append(future.result())
    except Exception as e:failed[str(futures[future])]=str(e)
   status()
   if done:print('FECAM_QUEUE',len(finished),'/',len(tasks),'failed',len(failed),flush=True)
 if failed:raise RuntimeError(json.dumps(failed))
 # Verify immutable files and the other ten methods once more across every updated task.
 import fecam_replay as replay
 for ds,enc,track,seed in tasks:
  key=Path(ds)/enc/track/f'seed{seed}';out=WORK/'readouts'/key;backup=WORK/'fecam_v1_backups'/key
  marker=json.loads((out/'fecam_patch2.json').read_text());assert marker['other_methods_unchanged'] and not marker['cuda_initialized']
  for name,digest in marker['immutable_files_sha256'].items():assert replay.digest(out/name)==digest
  import numpy as np
  old_refs=dict(np.load(backup/'closed_references.npz'));new_refs=dict(np.load(out/'closed_references.npz'));assert set(old_refs)==set(new_refs)
  assert all(np.array_equal(old_refs[k],new_refs[k]) for k in old_refs if not k.startswith('FeCAM_transfer_'))
  for session in out.glob('session*.json'):
   old=json.loads((backup/session.name).read_text());new=json.loads(session.read_text())
   assert replay.canon(replay.others(old['rows']))==replay.canon(replay.others(new['rows']))
 with (WORK/'summary.txt').open('w') as f:subprocess.run([sys.executable,str(ROOT/'fscil_summary.py')],cwd=ROOT,stdout=f,check=True)
 subprocess.run([sys.executable,str(ROOT/'audit_completed_cpu.py')],cwd=ROOT,check=True)
 status();(WORK/'fecam_replay2_complete.json').write_text(json.dumps(dict(complete=True,tasks=len(tasks),other_methods_unchanged=True,immutable_selection_complete_and_other_reference_arrays_unchanged=True,cuda_initialized=False,seconds=time.monotonic()-start),indent=2))
 print('FECAM_ALL_VERIFIED',len(tasks),flush=True)

if __name__=='__main__':main()
