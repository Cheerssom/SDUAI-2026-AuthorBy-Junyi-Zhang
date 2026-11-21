from pathlib import Path
import json,torch
import torch.nn.functional as F
from transformers import CLIPModel,CLIPTokenizer,CLIPConfig
R=Path('/root/autodl-tmp/JLT-CUB');torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.4)
def main():
 d=json.loads((R/'author_sources/bimc/description/modified_cub200_output.json').read_text());d={k.lower().replace('_',' ').replace('-',' '):v for k,v in d.items()}
 for enc in ['b32','l14']:
  out=R/'description_features'/f'{enc}.pt';out.parent.mkdir(exist_ok=True)
  if out.exists():continue
  z=torch.load(R/'features'/enc/'frozen.pt',map_location='cpu',weights_only=False);path=R/'models'/enc
  if enc=='b32':
   model=CLIPModel(CLIPConfig.from_pretrained(path));sd=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True);m,u=model.load_state_dict(sd,strict=False);assert not m and all(k.endswith('position_ids') for k in u)
  else:model=CLIPModel.from_pretrained(path,local_files_only=True,use_safetensors=True)
  model=model.cuda().eval();tok=CLIPTokenizer.from_pretrained(path,local_files_only=True);res=[]
  with torch.inference_mode():
   for i,name in enumerate(z['names']):
    texts=d[name.lower().replace('-',' ')];t=tok(texts,padding=True,truncation=True,return_tensors='pt').to('cuda');v=model.text_projection(model.text_model(**t).pooler_output);res.append(F.normalize(F.normalize(v.float(),dim=1).mean(0),dim=0).cpu())
    if i%50==0:print('DESCRIPTIONS',enc,i,flush=True)
  torch.save(torch.stack(res),out);del model;torch.cuda.empty_cache();print('DESCRIPTIONS COMPLETE',enc,flush=True)
if __name__=='__main__':main()
