"""Serial, resumable local execution of the four author-requested blocks."""
from pathlib import Path
import subprocess,sys,time,json,os
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_prospective_validation';PY=sys.executable
def paused():return (ROOT/'experiments/kbs_strengthening/PAUSE').exists()
def wait_for(path,label):
    if not path.exists():
        (OUT/'queue_status.json').write_text(json.dumps(dict(stage='wait_'+label.replace(' ','_').lower(),state='waiting_for_data',dependency=str(path),started=time.strftime('%Y-%m-%d %H:%M:%S')),indent=2))
    while not path.exists():
        if paused():raise SystemExit('Paused before '+label)
        print('WAIT',label,flush=True);time.sleep(30)
def run(name,script,args,complete):
    if complete.exists():print('REUSE',name,flush=True);return
    if paused():raise SystemExit('Paused before '+name)
    status=dict(stage=name,state='running',started=time.strftime('%Y-%m-%d %H:%M:%S'),command=[PY,script,*args]);(OUT/'queue_status.json').write_text(json.dumps(status,indent=2))
    print('START',name,flush=True)
    with (ROOT/f'experiments/expansion_prospective_{name}.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([PY,str(ROOT/'src/experiments'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode or not complete.exists():
        status.update(state='failed_or_paused',exit_code=result.returncode);(OUT/'queue_status.json').write_text(json.dumps(status,indent=2));raise SystemExit('Stage not complete: '+name)
    print('FINISH',name,flush=True)
def main():
    os.environ['KBS_GPU_FRACTION']='.50';os.environ['KBS_CPU_THREADS']='2';os.environ['PYTHONUTF8']='1';OUT.mkdir(parents=True,exist_ok=True)
    wait_for(OUT/'mechanism/complete.json','mechanism')
    run('cardinality','expansion_prospective_validation.py',['--block','cardinality'],OUT/'cardinality/complete.json')
    if '--wait-existing-shift' in sys.argv:wait_for(OUT/'shift/complete.json','existing shift process')
    run('shift','expansion_progressive_shift.py',[],OUT/'shift/complete.json')
    wait_for(Path('E:/expansion_external_data/food-101.tar.gz'),'Food download')
    run('food_features','expansion_external_features.py',['--dataset','food'],OUT/'external/food/b32/verification.json')
    for enc in ['l14','b32']:
        run('food_'+enc,'expansion_external_equal_budget.py',['--block','external','--dataset','food','--encoder',enc],OUT/f'external/{enc}/food/catalog/p9/complete.json')
    wait_for(Path('E:/expansion_external_data/sun397/download_complete.json'),'SUN download')
    run('sun_features','expansion_external_features.py',['--dataset','sun'],OUT/'external/sun/b32/verification.json')
    for enc in ['l14','b32']:
        run('sun_'+enc,'expansion_external_equal_budget.py',['--block','external','--dataset','sun','--encoder',enc],OUT/f'external/{enc}/sun/catalog/p9/complete.json')
    run('verification','verify_expansion_prospective.py',[],OUT/'verification.json')
    run('summary','summarize_expansion_prospective.py',[],OUT/'summary_complete.json')
    (OUT/'complete_all.json').write_text(json.dumps(dict(complete=True,blocks=['external','mechanism','cardinality','shift'])))
    (OUT/'queue_status.json').write_text(json.dumps(dict(stage='all',state='complete')));print('ALL FOUR BLOCKS COMPLETE',flush=True)
if __name__=='__main__':main()
