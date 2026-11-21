"""Frozen CLIP features for the standard FSCIL benchmarks with the public CEC index lists.

cifar100: 60 base + 8x5-way 5-shot; mini: 60 base + 8x5-way 5-shot; cub200: 100 base + 10x10-way 5-shot.
Session 1 uses every training image of the base classes; later sessions use the listed CEC images.
Output: work/features/<dataset>/<encoder>/frozen.pt in the format read by fscil_readouts.py.
"""
from pathlib import Path
import argparse,json,hashlib,pickle,csv,os,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
R=Path(__file__).resolve().parent;D=R/'work'
MODELS={'b16':'openai/clip-vit-base-patch16','b32':'openai/clip-vit-base-patch32','l14':'openai/clip-vit-large-patch14'}
TEMPLATES=['a photo of a {}.','a bad photo of a {}.','a cropped photo of a {}.','a close-up photo of a {}.','a bright photo of a {}.','a good photo of a {}.','a photo of the {}.','a photo of a {}.']
SPEC={'cifar100':(60,5,9),'mini':(60,5,9),'cub200':(100,10,11)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]

def load_cifar(root,idx):
 def ld(split):
  with open(root/split,'rb') as f:z=pickle.load(f,encoding='latin1')
  return np.asarray(z['data'],np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1),np.asarray(z['fine_labels'])
 xtr,ytr=ld('train');xte,yte=ld('test')
 with open(root/'meta','rb') as f:names=[n.replace('_',' ') for n in pickle.load(f,encoding='latin1')['fine_label_names']]
 imgs=[Image.fromarray(a) for a in np.concatenate([xtr,xte])];y=np.r_[ytr,yte];train=np.r_[np.ones(len(ytr),bool),np.zeros(len(yte),bool)]
 sess={'1':np.flatnonzero(ytr<60).tolist()}
 for s in range(2,10):sess[str(s)]=[int(v) for v in lines(idx/f'session_{s}.txt')]
 return imgs,y,train,sess,names

def load_mini(root,idx):
 # CEC layout: <root>/images/*.jpg, <root>/split/{train,test}.csv with columns filename,label(wnid)
 rows={}
 for split in ['train','test']:
  with open(root/'split'/f'{split}.csv') as f:rows[split]=[r for r in csv.reader(f)][1:]
 wnids=[];[wnids.append(w) for _,w in rows['train'] if w not in wnids]
 assert len(wnids)==100,len(wnids)
 paths=[];y=[];train=[]
 for split in ['train','test']:
  for fn,w in rows[split]:paths.append(root/'images'/fn);y.append(wnids.index(w));train.append(split=='train')
 y=np.asarray(y);train=np.asarray(train);pos={p.name:i for i,p in enumerate(paths) if train[i]}
 sess={'1':np.flatnonzero(train&(y<60)).tolist()}
 for s in range(2,10):sess[str(s)]=[pos[Path(l).name] for l in lines(idx/f'session_{s}.txt')]
 names=wnids
 namefile=root/'class_names.txt'
 if namefile.exists():m=dict(l.split(None,1) for l in lines(namefile));names=[m.get(w,w).split(',')[0] for w in wnids]
 return paths,y,train,sess,names

def load_cub(root,idx):
 base=root/'CUB_200_2011';rd=lambda f:{int(l.split(' ',1)[0]):l.split(' ',1)[1] for l in lines(base/f)}
 p=rd('images.txt');lab=rd('image_class_labels.txt');sp=rd('train_test_split.txt');cl=rd('classes.txt')
 ids=sorted(p);paths=[base/'images'/p[i] for i in ids];y=np.asarray([int(lab[i])-1 for i in ids]);train=np.asarray([sp[i]=='1' for i in ids])
 pos={p[i]:k for k,i in enumerate(ids)}
 sess={'1':np.flatnonzero(train&(y<100)).tolist()}
 for s in range(2,12):sess[str(s)]=[pos[l.split('CUB_200_2011/images/',1)[1]] for l in lines(idx/f'session_{s}.txt')]
 names=[cl[i].split('.',1)[1].replace('_',' ').lower() for i in range(1,201)]
 return paths,y,train,sess,names

@torch.inference_mode()
def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=list(SPEC),required=True);a.add_argument('--encoder',choices=list(MODELS),required=True);a.add_argument('--root',required=True);a.add_argument('--index_dir',required=True);a=a.parse_args()
 out=D/'features'/a.dataset/a.encoder;out.mkdir(parents=True,exist_ok=True)
 if (out/'frozen.pt').exists():
  verification=out/'verification.json'
  if not verification.exists():raise RuntimeError(f'Incomplete feature cache: {out}')
  verified=json.loads(verification.read_text())
  if not verified.get('complete') or verified.get('cache_sha256')!=sha(out/'frozen.pt'):raise RuntimeError(f'Feature checksum mismatch: {out}')
  print('EXISTS_VERIFIED',out,flush=True);return
 root,idx=Path(a.root),Path(a.index_dir)
 items,y,train,sess,names={'cifar100':load_cifar,'mini':load_mini,'cub200':load_cub}[a.dataset](root,idx)
 nb,way,S=SPEC[a.dataset]
 if (D/'PAUSE').exists():raise SystemExit('PAUSED')
 if a.dataset=='mini' and any(name.startswith('n') and len(name)==9 and name[1:].isdigit() for name in names):raise ValueError('miniImageNet requires complete readable class_names.txt; refusing wnid text anchors')
 assert len(set(sess['1']))==len(sess['1']) and set(y[sess['1']])==set(range(nb))
 assert np.all(train[sess['1']])
 for s in range(2,S+1):
  cls=sorted(set(y[sess[str(s)]].tolist()));exp=list(range(nb+way*(s-2),nb+way*(s-1)))
  assert cls==exp and len(sess[str(s)])==way*5,(s,cls[:3],len(sess[str(s)]))
  assert all(train[sess[str(s)]])
  assert len(set(sess[str(s)]))==len(sess[str(s)])
  assert np.all(np.unique(y[sess[str(s)]],return_counts=True)[1]==5),(s,'expected exactly 5 images per novel class')
 from transformers import CLIPModel,CLIPTokenizer,CLIPImageProcessor
 model=CLIPModel.from_pretrained(MODELS[a.encoder],local_files_only=True).cuda().eval();tok=CLIPTokenizer.from_pretrained(MODELS[a.encoder],local_files_only=True);proc=CLIPImageProcessor.from_pretrained(MODELS[a.encoder],local_files_only=True)
 feats=[];B=256
 torch.set_num_threads(4)
 class ImageDataset(torch.utils.data.Dataset):
  def __len__(self):return len(items)
  def __getitem__(self,i):
   it=items[i]
   if isinstance(it,Image.Image):im=it.convert('RGB')
   else:
    with Image.open(it) as original:im=original.convert('RGB')
   return proc(images=im,return_tensors='pt')['pixel_values'][0]
 workers=int(os.environ.get('FSCIL_IMAGE_WORKERS','6'))
 loader=torch.utils.data.DataLoader(ImageDataset(),batch_size=B,num_workers=workers,pin_memory=True,persistent_workers=workers>0,shuffle=False)
 started=time.time()
 for batch,x in enumerate(loader):
  i=batch*B
  if batch==0:
   ims=[(it if isinstance(it,Image.Image) else Image.open(it)).convert('RGB') for it in items[:8]]
   expected=proc(images=ims,return_tensors='pt')['pixel_values']
   assert torch.equal(x[:8],expected),'Parallel preprocessing mismatch'
   print('PREPROCESS_EXACT_MATCH',a.dataset,a.encoder,flush=True)
  x=x.cuda(non_blocking=True)
  with torch.autocast('cuda',dtype=torch.float16):v=model.get_image_features(pixel_values=x)
  feats.append(F.normalize(v.float(),dim=1).cpu())
  if i%(B*40)==0:print('FEATURES',a.dataset,a.encoder,min(i+B,len(items)),'/',len(items),'elapsed',round(time.time()-started,1),flush=True)
 texts=[]
 for t in TEMPLATES:
  b=tok([t.format(n) for n in names],padding=True,return_tensors='pt').to('cuda');texts.append(F.normalize(model.get_text_features(**b).float(),dim=1).cpu())
 T=F.normalize(torch.stack(texts).mean(0),dim=1)
 torch.save(dict(E=torch.cat(feats),T=T,y=y,ids=np.arange(len(y)),train=train,sessions=sess,names=names,model=MODELS[a.encoder]),out/'frozen.pt.tmp')
 (out/'frozen.pt.tmp').replace(out/'frozen.pt')
 json.dump(dict(complete=True,rows=len(y),cache_sha256=sha(out/'frozen.pt')),open(out/'verification.json','w'),indent=2)
 print('FEATURES_COMPLETE',a.dataset,a.encoder,flush=True)
if __name__=='__main__':main()
