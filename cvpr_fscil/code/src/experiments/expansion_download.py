"""Resumable public data downloads; no authentication or environment changes."""
from pathlib import Path
import argparse,time,requests,hashlib,json,concurrent.futures

def fetch(url,path,expected=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (expected is None or path.stat().st_size==expected):return path
    tmp=path.with_name(path.name+'.partial')
    for attempt in range(12):
        try:
            start=tmp.stat().st_size if tmp.exists() else 0
            r=requests.get(url,headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(20,90))
            r.raise_for_status()
            if start and r.status_code!=206:start=0
            total=start+int(r.headers.get('content-length','0'));last=time.monotonic()
            with tmp.open('ab' if start else 'wb') as f:
                for block in r.iter_content(2**20):
                    if block:f.write(block)
                    if time.monotonic()-last>30:
                        print('DOWNLOAD',path.name,round(f.tell()/2**20),round(total/2**20),'MiB',flush=True);last=time.monotonic()
            r.close()
            if expected:assert tmp.stat().st_size==expected,(path,tmp.stat().st_size,expected)
            tmp.replace(path);print('DOWNLOADED',path,flush=True);return path
        except Exception as e:
            print('DOWNLOAD RETRY',attempt,type(e).__name__,str(e)[:160],flush=True);time.sleep(min(10,attempt+1))
    raise RuntimeError(f'Download did not complete: {url}')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['food','sun'],required=True);a=ap.parse_args()
    root=Path('E:/expansion_external_data');root.mkdir(exist_ok=True)
    if a.dataset=='food':fetch('https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz',root/'food-101.tar.gz',4996278331)
