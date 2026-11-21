import requests,tarfile,hashlib,concurrent.futures,time
from pathlib import Path
root=Path('/root/autodl-tmp/data')
def get(ds,url):
 dest=root/('CUB_200_2011.tgz' if ds=='cub' else 'miniimagenet.tar')
 print('DOWNLOAD_START',ds,flush=True)
 with requests.get(url,stream=True,timeout=(15,45)) as r:
  r.raise_for_status();print('HEADERS',ds,r.status_code,r.headers.get('Content-Length'),r.headers.get('Content-Type'),flush=True)
  assert 'text/html' not in r.headers.get('Content-Type','')
  with dest.with_suffix('.download').open('wb') as f:
   n=0
   for b in r.iter_content(4*1024*1024):
    f.write(b);n+=len(b)
    if n%(64*1024*1024)==0:print('BYTES',ds,n,flush=True)
 dest.with_suffix('.download').replace(dest)
 if ds=='cub':assert hashlib.md5(dest.read_bytes()).hexdigest()=='97eceeb196236b17998738112f37df78'
 out=root/'cub' if ds=='cub' else root;out.mkdir(exist_ok=True)
 with tarfile.open(dest) as t:
  for m in t.getmembers():assert (out/m.name).resolve().is_relative_to(out.resolve()) and not m.issym() and not m.islnk()
  t.extractall(out)
 print('DATA_READY',ds,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:
 fs=[p.submit(get,'cub','https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'),p.submit(get,'mini','https://drive.usercontent.google.com/download?id=1_x4o0iFetEv-T3PeIxdSbBPUG4hFfT8U&export=download&confirm=t&uuid=098bb15b-bfc5-4142-8ff6-1036497b1c4c')]
 for f in concurrent.futures.as_completed(fs):
  try:f.result()
  except Exception as e:print('DATA_ERROR',repr(e),flush=True)
