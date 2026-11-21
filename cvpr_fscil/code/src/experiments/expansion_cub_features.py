"""Pinned CLIP extraction of official CUB images; no test-driven task selection."""
from pathlib import Path
import argparse,json,hashlib,tarfile,time,os,gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT=Path('/root/autodl-tmp/JLT-CUB')
REVISIONS={'b32':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268','l14':'32bd64288804d66eefd0ccbe215aa642df71cc41'}
TEMPLATES=['a photo of a {}.','a bad photo of a {}.','a cropped photo of a {}.','a close-up photo of a {}.','a bright photo of a {}.','a good photo of a {}.','a photo of the {}.','a photo of a {}.']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def save(v,p):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');torch.save(v,q);q.replace(p)
def inventory():
 receipt=ROOT/'data/extracted.json';base=ROOT/'data/CUB_200_2011'
 if not receipt.exists():
  archives=[p for p in (ROOT/'data').glob('*.tgz') if p.stat().st_size==1150585339]
  assert archives,'No complete official archive'
  archive=archives[0]
  with archive.open('rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
  assert md5=='97eceeb196236b17998738112f37df78',md5
  with tarfile.open(archive) as tar:
   for member in tar:
    if not member.isfile():continue
    p=ROOT/'data'/member.name;assert p.resolve().is_relative_to((ROOT/'data').resolve()) and not member.issym() and not member.islnk()
    if p.exists() and p.stat().st_size==member.size:continue
    p.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,p.open('wb') as dst:
     while buf:=src.read(2**20):dst.write(buf)
  atomic(dict(md5=md5,sha256=sha(archive)),receipt)
 manifest=ROOT/'data/inventory.json'
 if manifest.exists():return json.loads(manifest.read_text())
 read=lambda file:{int(l.split(' ',1)[0]):l.split(' ',1)[1] for l in (base/file).read_text().splitlines()}
 paths=read('images.txt');labels=read('image_class_labels.txt');split=read('train_test_split.txt');names=read('classes.txt')
 rows=[dict(id=i,path=paths[i],label=int(labels[i])-1,train=bool(int(split[i])),sha256=sha(base/'images'/paths[i])) for i in sorted(paths)]
 assert len(rows)==11788 and len(names)==200
 lists={};allselected=set()
 for session in range(1,12):
  p=ROOT/f'index_list/session_{session}.txt';rr=p.read_text().splitlines();images={r['path']:r for r in rows};ids=[]
  for entry in rr:
   k=entry.split('CUB_200_2011/images/',1)[1];row=images[k];assert row['train'];ids.append(row['id'])
  expected=list(range(100)) if session==1 else list(range(100+10*(session-2),110+10*(session-2)))
  assert sorted(set(images[v.split('CUB_200_2011/images/',1)[1]]['label'] for v in rr))==expected
  assert len(ids)==(3000 if session==1 else 50) and not allselected.intersection(ids)
  lists[str(session)]=ids;allselected.update(ids)
 spec=dict(rows=rows,names=[names[i].split('.',1)[1].replace('_',' ').replace('-',' ').lower() for i in range(1,201)],sessions=lists,index_manifest_sha256=sha(ROOT/'official_index_manifest.json'),source_md5='97eceeb196236b17998738112f37df78')
 atomic(spec,manifest);return spec
@torch.inference_mode()
def extract(enc,provisional=False):
 from transformers import CLIPModel,CLIPTokenizer,CLIPConfig
 from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.42)
 if provisional:
  archive=ROOT/'data/CUB_hf.tgz'
  try:
   with tarfile.open(archive,'r|gz') as tar:
    for m in tar:
     if not m.isfile() or not m.name.lower().endswith('.jpg'):continue
     target=ROOT/'data'/m.name
     assert target.resolve().is_relative_to((ROOT/'data').resolve()) and not m.issym() and not m.islnk()
     if target.exists() and target.stat().st_size==m.size:continue
     raw=tar.extractfile(m).read()
     if len(raw)!=m.size:break
     target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  except (EOFError,tarfile.ReadError,OSError) as e:print('PREFIX_END',str(e),flush=True)
  images=sorted((ROOT/'data/CUB_200_2011/images').rglob('*.jpg'))
  spec=dict(rows=[dict(path=p.relative_to(ROOT/'data/CUB_200_2011/images').as_posix(),id=i,sha256=sha(p)) for i,p in enumerate(images)])
  print('PROVISIONAL_IMAGES',len(images),'UNVERIFIED_UNTIL_FULL_MD5',flush=True)
 else:spec=inventory()
 out=ROOT/('provisional_features' if provisional else 'features')/enc;out.mkdir(parents=True,exist_ok=True)
 if (out/'verification.json').exists():print('ALREADY COMPLETE',enc,flush=True);return
 modelpath=ROOT/'models'/enc;hashpath=ROOT/f'model_{enc}_hashes.json'
 hashes=json.loads(hashpath.read_text())
 for path,h in hashes.items():assert sha(ROOT/path)==h,path
 if enc=='b32':
  model=CLIPModel(CLIPConfig.from_pretrained(modelpath,local_files_only=True));state=torch.load(modelpath/'pytorch_model.bin',map_location='cpu',weights_only=True);missing,unexpected=model.load_state_dict(state,strict=False);assert not missing and all(k.endswith('position_ids') for k in unexpected),(missing,unexpected);del state
 else:model=CLIPModel.from_pretrained(modelpath,local_files_only=True,use_safetensors=True)
 model=model.to('cuda').eval();tokenizer=CLIPTokenizer.from_pretrained(modelpath,local_files_only=True)
 transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
 def image(r):
  with Image.open(ROOT/'data/CUB_200_2011/images'/r['path']) as im:return transform(im.convert('RGB'))
 cached={}
 if not provisional:
  for p in (ROOT/'provisional_features'/enc).glob('batch_*.pt'):
   z=torch.load(p,map_location='cpu',weights_only=False)
   for i,path in enumerate(z.get('paths',[])):cached[path]=(z['features'][i],z['hashes'][i])
 N=len(spec['rows']);batchsize=128 if enc=='b32' else 64;blocks=[];done=0;start=time.monotonic()
 for p in sorted(out.glob('batch_*.pt')):
  z=torch.load(p,map_location='cpu',weights_only=False);assert z['start']==done;blocks.append(z['features']);done+=len(z['features'])
 with ThreadPoolExecutor(8) as executor:
  while done<N:
   if (ROOT/'PAUSE').exists():print('PAUSED',enc,done,flush=True);return
   subset=spec['rows'][done:done+batchsize]
   if not provisional and all(r['path'] in cached and cached[r['path']][1]==r['sha256'] for r in subset):
    v=torch.stack([cached[r['path']][0] for r in subset]);save(dict(start=done,features=v,ids=[r['id'] for r in subset],paths=[r['path'] for r in subset],hashes=[r['sha256'] for r in subset]),out/f'batch_{done:06d}.pt');blocks.append(v);done+=len(v);continue
   x=torch.stack(list(executor.map(image,subset))).pin_memory().to('cuda',non_blocking=True)
   try:
    with torch.autocast('cuda',dtype=torch.float16):v=model.visual_projection(model.vision_model(pixel_values=x).pooler_output)
   except torch.cuda.OutOfMemoryError:
    del x;gc.collect();torch.cuda.empty_cache();batchsize=max(1,batchsize//2);continue
   v=F.normalize(v.float(),dim=1).cpu();save(dict(start=done,features=v,ids=[r['id'] for r in subset],paths=[r['path'] for r in subset],hashes=[r['sha256'] for r in subset]),out/f'batch_{done:06d}.pt');blocks.append(v);done+=len(v)
   atomic(dict(encoder=enc,stage='image features',completed=done,total=N,seconds=time.monotonic()-start,batchsize=batchsize),out/'status.json')
   if done%512==0 or done==N:print('CUB_FEATURES',enc,done,'/',N,'seconds',round(time.monotonic()-start),flush=True)
 if provisional:
  atomic(dict(provisional=True,accepted=False,completed=N,seconds=time.monotonic()-start),out/'provisional_complete.json');return
 features=torch.cat(blocks);texts=[]
 for template in TEMPLATES:
  chunks=[]
  for i in range(0,200,64):
   tokens=tokenizer([template.format(n) for n in spec['names'][i:i+64]],padding=True,return_tensors='pt').to('cuda');chunks.append(F.normalize(model.text_projection(model.text_model(**tokens).pooler_output).float(),dim=1).cpu())
  texts.append(torch.cat(chunks))
 T=F.normalize(torch.stack(texts).mean(0),dim=1)
 x=torch.stack([image(r) for r in spec['rows'][:8]]).to('cuda');w=F.normalize(model.visual_projection(model.vision_model(pixel_values=x).pooler_output).float(),dim=1).cpu();cos=F.cosine_similarity(w,features[:8]);assert cos.min()>.999
 save(dict(E=features,T=T,y=np.array([r['label'] for r in spec['rows']]),ids=np.array([r['id'] for r in spec['rows']]),train=np.array([r['train'] for r in spec['rows']]),sessions=spec['sessions'],names=spec['names'],templates=TEMPLATES,model_revision=REVISIONS[enc],data_manifest_sha256=sha(ROOT/'data/inventory.json')),out/'frozen.pt')
 atomic(dict(complete=True,rows=N,min_fp32_witness_cosine=float(cos.min()),cache_sha256=sha(out/'frozen.pt'),model_revision=REVISIONS[enc],seconds=time.monotonic()-start),out/'verification.json');print('CUB COMPLETE',enc,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--encoder',choices=['b32','l14'],required=True);p.add_argument('--provisional',action='store_true');a=p.parse_args();extract(a.encoder,a.provisional)
