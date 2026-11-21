from pathlib import Path
import requests,hashlib,time,os,json
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path('/root/autodl-tmp/JLT-CUB');D=R/'data/parts';D.mkdir(exist_ok=True)
N=1150585339;CHUNK=16*1024*1024;URL='https://hf-mirror.com/datasets/XiN0919/FGVC/resolve/main/CUB_200_2011.tgz';jobs=[]
def fetch(i):
 start=i*CHUNK;end=min(N,start+CHUNK)-1;p=D/f'{i:04d}.part'
 if p.exists() and p.stat().st_size==end-start+1:return i
 for attempt in range(6):
  try:
   r=requests.get(URL,headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache'},timeout=(20,60),stream=True);r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),(r.status_code,r.headers.get('Content-Range'))
   q=p.with_suffix('.tmp')
   with q.open('wb') as f:
    for x in r.iter_content(1024*1024):f.write(x)
   assert q.stat().st_size==end-start+1;q.replace(p);print('CUB_CHUNK',i,end-start+1,flush=True);return i
  except Exception as e:print('RETRY',i,repr(e),flush=True);time.sleep(min(20,attempt+1))
 raise RuntimeError(f'chunk {i} failed')
with ThreadPoolExecutor(8) as ex:
 for future in as_completed([ex.submit(fetch,i) for i in range((N+CHUNK-1)//CHUNK)]):future.result()
p=R/'data/CUB_parallel.tgz'
with p.open('wb') as out:
 for i in range((N+CHUNK-1)//CHUNK):
  with (D/f'{i:04d}.part').open('rb') as source:
   while x:=source.read(2**20):out.write(x)
with p.open('rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
assert md5=='97eceeb196236b17998738112f37df78',md5
print('CUB_DATA_VERIFIED',p.stat().st_size,md5,flush=True)
