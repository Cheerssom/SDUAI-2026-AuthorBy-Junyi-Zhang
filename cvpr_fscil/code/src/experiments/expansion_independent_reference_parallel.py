import os,sys,subprocess,concurrent.futures,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
script=root/'src/experiments/expansion_independent_reference_replay.py'
out=root/'experiments/expansion_independent_reference_replay';out.mkdir(parents=True,exist_ok=True)
def run(task):
 en,part=task;env=os.environ.copy();env.update(IR_ENCODERS=en,IR_PARTS=str(part),IR_WORKER='1',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONUTF8='1')
 with (out/f'worker_{en}_p{part}.log').open('w',encoding='utf8') as f:
  r=subprocess.run([sys.executable,'-u',str(script)],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError(f'{en} p{part}: exit {r.returncode}')
 return en,part
if __name__=='__main__':
 tasks=[(en,p) for p in range(10) for en in ['l14','b32']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  fs=[pool.submit(run,t) for t in tasks]
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):print('PROGRESS',i,'/20',*f.result(),flush=True)
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONUTF8='1')
 r=subprocess.run([sys.executable,'-u',str(script)],cwd=root,env=env);sys.exit(r.returncode)
