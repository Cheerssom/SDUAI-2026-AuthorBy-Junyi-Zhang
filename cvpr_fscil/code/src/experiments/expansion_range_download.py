"""Bounded parallel range transfer with persistent exact-size chunks."""
from pathlib import Path
import argparse,requests,time,hashlib,concurrent.futures,json

def download(url,path,total,workers=12,chunk=16*2**20):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==total:return
    folder=path.with_name(path.name+'.ranges');folder.mkdir(exist_ok=True)
    def one(i):
        start=i*chunk;end=min(total,start+chunk)-1;p=folder/f'{i:05d}.bin';expected=end-start+1
        if p.exists() and p.stat().st_size==expected:return
        tmp=p.with_suffix('.partial')
        for attempt in range(12):
            try:
                offset=tmp.stat().st_size if tmp.exists() else 0
                if offset==expected:tmp.replace(p);return
                r=requests.get(url,headers={'Range':f'bytes={start+offset}-{end}'},stream=True,timeout=(30,90));r.raise_for_status()
                assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start+offset}-'),r.headers
                with tmp.open('ab') as f:
                    for block in r.iter_content(2**20):
                        if block:f.write(block)
                r.close();assert tmp.stat().st_size==expected;tmp.replace(p);return
            except Exception as e:
                print('RANGE RETRY',i,attempt,str(e)[:100],flush=True);time.sleep(min(10,attempt+1))
        raise RuntimeError(f'Failed range {i}')
    n=(total+chunk-1)//chunk;last=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for future in concurrent.futures.as_completed([ex.submit(one,i) for i in range(n)]):
            future.result()
            if time.monotonic()-last>20:
                size=sum(p.stat().st_size for p in folder.glob('*.bin'));print('RANGES',path.name,round(size/2**20),'/',round(total/2**20),'MiB',flush=True);last=time.monotonic()
    tmp=path.with_name(path.name+'.assembling');sha=hashlib.sha256()
    with tmp.open('wb') as out:
        for i in range(n):
            with (folder/f'{i:05d}.bin').open('rb') as f:
                while block:=f.read(2**20):out.write(block);sha.update(block)
    assert tmp.stat().st_size==total;tmp.replace(path)
    path.with_suffix(path.suffix+'.sha256').write_text(sha.hexdigest());print('DOWNLOADED',path,sha.hexdigest(),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--url',required=True);a.add_argument('--path',required=True);a.add_argument('--bytes',type=int,required=True);a.add_argument('--workers',type=int,default=12);v=a.parse_args();download(v.url,v.path,v.bytes,v.workers)
