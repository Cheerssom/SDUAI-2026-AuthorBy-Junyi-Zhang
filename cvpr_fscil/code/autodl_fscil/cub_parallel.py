from pathlib import Path
import requests,concurrent.futures,hashlib,tarfile,time
root=Path('/root/autodl-tmp/data');parts=root/'cub_parts';parts.mkdir(exist_ok=True);size=1150585339;chunk=8*1024*1024
url='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
def get(i):
 a=i*chunk;b=min(size-1,a+chunk-1);p=parts/str(i)
 if p.exists() and p.stat().st_size==b-a+1:return
 for attempt in range(12):
  try:
   r=requests.get(url+f'?part={i}&retry={attempt}&identity=1',headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity'},timeout=(15,30));r.raise_for_status();assert r.status_code==206,(r.status_code,i);assert len(r.content)==b-a+1,(i,attempt,len(r.content),r.headers.get("Content-Range"),r.headers.get("Content-Encoding"));p.write_bytes(r.content);return
  except Exception as e:
   if attempt==11:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for j,_ in enumerate(pool.map(get,range((size+chunk-1)//chunk)),1):
  if j%8==0:print('CUB_CHUNKS',j,'/',(size+chunk-1)//chunk,flush=True)
p=root/'CUB_verified.tgz'
with p.open('wb') as f:
 for i in range((size+chunk-1)//chunk):f.write((parts/str(i)).read_bytes())
assert hashlib.md5(p.read_bytes()).hexdigest()=='97eceeb196236b17998738112f37df78'
out=root/'cub';out.mkdir(exist_ok=True)
with tarfile.open(p) as t:
 for m in t.getmembers():assert (out/m.name).resolve().is_relative_to(out.resolve()) and not m.issym() and not m.islnk()
 t.extractall(out)
print('CUB_READY',flush=True)
