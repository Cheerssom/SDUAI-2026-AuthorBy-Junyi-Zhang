from pathlib import Path
import concurrent.futures,requests,tarfile,zipfile,io,hashlib,json,sys
import gdown
root=Path('/root/autodl-tmp/data');root.mkdir(exist_ok=True)
def download(url,dest):
 if dest.exists():return
 temp=dest.with_suffix(dest.suffix+'.part')
 with requests.get(url,stream=True,timeout=60) as r:
  r.raise_for_status()
  with temp.open('wb') as f:
   for chunk in r.iter_content(1024*1024):f.write(chunk)
 temp.replace(dest)
def extract(path,dest):
 dest.mkdir(exist_ok=True,parents=True)
 with tarfile.open(path) as t:
  for m in t.getmembers():
   q=(dest/m.name).resolve()
   if not q.is_relative_to(dest.resolve()) or m.issym() or m.islnk():raise ValueError('Unsafe tar path')
  t.extractall(dest)
def dataset(ds):
 print('DATA_START',ds,flush=True)
 if ds=='cifar100':
  path=root/'cifar.tar.gz';download('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',path)
  assert hashlib.md5(path.read_bytes()).hexdigest()=='eb9058c3a382ffc7106e4002c42a8d85'
  if not (root/'cifar-100-python/train').exists():extract(path,root)
 else:
  fid,fn,dest=('1EtKb1Bxd9ZtK-WUFk5dIhoOdhLeohA1q','CUB_200_2011.tgz',root/'cub') if ds=='cub200' else ('1_x4o0iFetEv-T3PeIxdSbBPUG4hFfT8U','miniimagenet.tar',root)
  path=root/fn
  if not path.exists():
   temp=str(path)+'.part';r=gdown.download(id=fid,output=temp,quiet=False,resume=True)
   if not r:raise RuntimeError('Google Drive download failed '+ds)
   Path(temp).replace(path)
  if ds=='cub200':assert hashlib.md5(path.read_bytes()).hexdigest()=='97eceeb196236b17998738112f37df78'
  marker=root/'cub/CUB_200_2011/images.txt' if ds=='cub200' else root/'miniimagenet/split/train.csv'
  if not marker.exists():extract(path,dest)
 print('DATA_COMPLETE',ds,flush=True)
print('FETCH_CEC',flush=True)
r=requests.get('https://codeload.github.com/icoz69/CEC-CVPR2021/zip/refs/heads/main',timeout=60);r.raise_for_status();z=zipfile.ZipFile(io.BytesIO(r.content));manifest={}
for info in z.infolist():
 if '/data/index_list/' in info.filename and not info.is_dir():
  rel=info.filename.split('/data/index_list/',1)[1];p=root/'index_list'/rel;p.parent.mkdir(parents=True,exist_ok=True);data=z.read(info);p.write_bytes(data);manifest[rel]=hashlib.sha256(data).hexdigest()
(root/'index_manifest.json').write_text(json.dumps(manifest,indent=2));print('CEC_READY',len(manifest),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(dataset,['cifar100','mini','cub200']))
r=requests.get('https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json',timeout=30);r.raise_for_status();mapping={a[0]:a[1].replace('_',' ') for a in r.json().values()}
import csv
with (root/'miniimagenet/split/train.csv').open() as f:wnids=sorted({row[1] for row in list(csv.reader(f))[1:]})
assert len(wnids)==100 and all(w in mapping for w in wnids)
(root/'miniimagenet/class_names.txt').write_text('\n'.join(w+' '+mapping[w] for w in wnids))
print('ALL_DATA_READY',flush=True)
