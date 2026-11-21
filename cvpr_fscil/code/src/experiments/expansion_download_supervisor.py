"""Watch owned data transfers and retry missing chunks after network failures."""
from pathlib import Path
import sys,time,subprocess,concurrent.futures,json
import psutil
ROOT=Path(__file__).resolve().parents[2];DATA=Path('E:/expansion_external_data');OUT=ROOT/'experiments/expansion_prospective_validation'
def active(script):
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            if process.pid==psutil.Process().pid:continue
            argv=process.info['cmdline'] or []
            if any(Path(a).name==script for a in argv):return True
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return False
def task(name,script,args,complete):
    failures=0
    while not complete.exists():
        if active(script):time.sleep(30);continue
        if failures>=6:raise RuntimeError(f'{name}:six unsuccessful resumptions;inspect download logs')
        print('RESUME TRANSFER',name,flush=True)
        with (ROOT/f'experiments/expansion_{name}_supervised.log').open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'src/experiments'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if not complete.exists():
            failures+=1;print('TRANSFER RETRY',name,result.returncode,failures,flush=True);time.sleep(30)
    print('TRANSFER READY',name,flush=True)
def main():
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        futures=[ex.submit(task,'food','expansion_range_download.py',['--url','https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz','--path',str(DATA/'food-101.tar.gz'),'--bytes','4996278331'],DATA/'food-101.tar.gz'),ex.submit(task,'sun','expansion_sun_download.py',[],DATA/'sun397/download_complete.json')]
        for f in futures:f.result()
    (OUT/'downloads_complete.json').write_text(json.dumps(dict(complete=True)))
if __name__=='__main__':main()
