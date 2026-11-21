"""Frozen CLIP features for the standard CIFAR-100 FSCIL split (60 base + 8x5-way 5-shot).

Session lists are the public CEC index files (data/index_list/cifar100/session_{1..9}.txt).
Session 1 uses every training image of classes 0-59; sessions 2-9 use the 25 listed
training indices. Output format matches the CUB cache used by expansion_cub_readouts.py.
"""
from pathlib import Path
import argparse,json,hashlib,pickle
import numpy as np
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[2];D=R/'experiments/expansion_cifar_fscil'
MODELS={'b32':'openai/clip-vit-base-patch32','l14':'openai/clip-vit-large-patch14'}
TEMPLATES=['a photo of a {}.','a bad photo of a {}.','a cropped photo of a {}.','a close-up photo of a {}.','a bright photo of a {}.','a good photo of a {}.','a photo of the {}.','a photo of a {}.']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(root,split):
 with open(root/split,'rb') as f:z=pickle.load(f,encoding='latin1')
 return np.asarray(z['data'],np.uint8).reshape(-1,3,32,32),np.asarray(z['fine_labels'])
def sessions(index_dir,ytrain):
 out={'1':np.flatnonzero(ytrain<60).tolist()};assert len(out['1'])==30000
 for s in range(2,10):
  ids=[int(l.strip()) for l in (index_dir/f'session_{s}.txt').read_text().splitlines() if l.strip()]
  expected=list(range(60+5*(s-2),65+5*(s-2)))
  assert len(ids)==25 and sorted(set(ytrain[ids].tolist()))==expected,(s,len(ids),sorted(set(ytrain[ids].tolist())))
  out[str(s)]=ids
 return out
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--encoder',choices=list(MODELS),required=True)
 p.add_argument('--cifar_root',default=r'G:\科研新\data\cifar-100-python');p.add_argument('--index_dir',required=True,help='CEC data/index_list/cifar100')
 a=p.parse_args();root=Path(a.cifar_root);index_dir=Path(a.index_dir);out=D/'features'/a.encoder;out.mkdir(parents=True,exist_ok=True)
 from transformers import CLIPModel,CLIPTokenizer
 xtr,ytr=load(root,'train');xte,yte=load(root,'test');sess=sessions(index_dir,ytr)
 with open(root/'meta','rb') as f:names=[n.replace('_',' ') for n in pickle.load(f,encoding='latin1')['fine_label_names']]
 model=CLIPModel.from_pretrained(MODELS[a.encoder]).cuda().eval();tok=CLIPTokenizer.from_pretrained(MODELS[a.encoder])
 mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None];std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
 X=np.concatenate([xtr,xte]);feats=[]
 for i in range(0,len(X),256):
  x=torch.from_numpy(X[i:i+256]).cuda().float()/255;x=F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False)
  with torch.autocast('cuda',dtype=torch.float16):v=model.visual_projection(model.vision_model(pixel_values=(x-mean)/std).pooler_output)
  feats.append(F.normalize(v.float(),dim=1).cpu())
  if i%10240==0:print('CIFAR_FEATURES',a.encoder,i,'/',len(X),flush=True)
 texts=[]
 for t in TEMPLATES:
  b=tok([t.format(n) for n in names],padding=True,return_tensors='pt').to('cuda');texts.append(F.normalize(model.text_projection(model.text_model(**b).pooler_output).float(),dim=1).cpu())
 T=F.normalize(torch.stack(texts).mean(0),dim=1)
 torch.save(dict(E=torch.cat(feats),T=T,y=np.r_[ytr,yte],ids=np.arange(len(X)),train=np.r_[np.ones(len(ytr),bool),np.zeros(len(yte),bool)],sessions=sess,names=names,templates=TEMPLATES,model=MODELS[a.encoder]),out/'frozen.pt')
 json.dump(dict(complete=True,rows=len(X),cache_sha256=sha(out/'frozen.pt'),index_sha256={s:sha(index_dir/f'session_{s}.txt') for s in range(2,10)}),open(out/'verification.json','w'),indent=2)
 print('CIFAR_FEATURES_COMPLETE',a.encoder,flush=True)
if __name__=='__main__':main()
