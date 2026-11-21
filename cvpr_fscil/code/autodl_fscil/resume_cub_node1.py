from pathlib import Path
import requests,concurrent.futures,time,json
NODE='node1'
indices=[127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137]
parts=Path('/root/autodl-tmp/data/cub_parts');parts.mkdir(exist_ok=True)
size=1150585339;chunk=8388608;unit=1048576
url='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
def get(i):
 a=i*chunk;end=min(size,a+chunk);length=end-a;p=parts/str(i);temp=p.with_suffix('.part');failures=0
 if p.exists():assert p.stat().st_size==length;return
 if not temp.exists():temp.touch()
 assert temp.stat().st_size<=length
 while temp.stat().st_size<length:
  before=temp.stat().st_size;start=a+before;stop=min(end-1,start+unit-1)
  try:
   with requests.get(url+f'?resume_node={NODE}&part={i}&offset={before}&try={failures}',headers={'Range':f'bytes={start}-{stop}','Accept-Encoding':'identity'},stream=True,timeout=(10,12)) as response:
    response.raise_for_status();assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {start}-{stop}/{size}',(response.status_code,response.headers.get('Content-Range'))
    with temp.open('ab') as f:
     for content in response.iter_content(65536):f.write(content)
  except Exception as e:print('RETRY_RESUME',i,temp.stat().st_size,repr(e)[:140],flush=True)
  now=temp.stat().st_size;assert now<=length,(i,now,length)
  if now>before:
   failures=0;print('BYTE_PROGRESS',i,now,'/',length,flush=True)
  else:
   failures+=1
   if failures>=12:raise RuntimeError(f'No byte progress for chunk {i} at {now}')
   time.sleep(2)
 temp.replace(p);print('CHUNK_COMPLETE',i,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(get,indices))
print('ASSIGNED_DOWNLOAD_COMPLETE',NODE,flush=True)
