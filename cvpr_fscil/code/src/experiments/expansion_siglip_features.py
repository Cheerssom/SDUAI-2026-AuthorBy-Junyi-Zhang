"""Native SigLIP preprocessing and batch-resumable original ImageNet-R pools."""
from pathlib import Path
import json,sys,time,gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import CFG
from experiments.fcs_pets_data import TEMPLATES
from experiments.kbs_runtime import configure,check_pause,atomic_json,atomic_torch,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_cross_encoder'

@torch.inference_mode()
def main():
    device=configure();check_pause();modeldir=OUT/'model';featuredir=OUT/'features';featuredir.mkdir(exist_ok=True)
    mm=modeldir/'manifest.json';assert mm.exists(),'Pinned model download incomplete.'
    modelmanifest=json.loads(mm.read_text())
    for name,sha in modelmanifest['file_sha256'].items():assert digest_file(modeldir/name)==sha
    src=ROOT/'experiments/kbs_imagenetr/data_manifest.json';calp=ROOT/'experiments/kbs_calibration_budget/pools/imagenetr.json'
    source=json.loads(src.read_text());cal=json.loads(calp.read_text())
    pools={k:list(v) for k,v in source['pools'].items()}
    pools['extra']=[dict(path=r['identity'],**{'class':r['class_id'],'sha256':r['sha256']}) for r in cal['rows'] if 'cached_index' not in r]
    config=dict(model_manifest_sha256=digest_file(mm),data_manifest_sha256=digest_file(src),
                calibration_pool_sha256=digest_file(calp),source_sha256=digest_file(Path(__file__)),
                protocol_sha256=digest_file(ROOT/'experiments/expansion_cross_encoder_protocol.md'),
                model_revision=modelmanifest['revision'],templates=TEMPLATES,
                preprocessing='native SiglipImageProcessor RGB resize224 rescale1/255 mean/std .5; native SiglipTokenizer padding64')
    mp=featuredir/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==config
    else:atomic_json(config,mp)
    if (featuredir/'verification.json').exists():
        assert digest_file(featuredir/'frozen.pt')==json.loads((featuredir/'verification.json').read_text())['cache_sha256'];return
    from transformers import SiglipModel,SiglipTokenizer,SiglipImageProcessor
    model=SiglipModel.from_pretrained(str(modeldir),local_files_only=True,use_safetensors=True).to(device).eval()
    tokenizer=SiglipTokenizer.from_pretrained(str(modeldir),local_files_only=True)
    processor=SiglipImageProcessor.from_pretrained(str(modeldir),local_files_only=True)
    def encode(rows):
        images=[]
        for row in rows:
            path=CFG.data_root/'imagenet-r'/row['path'];assert digest_file(path)==row['sha256']
            with Image.open(path) as im:images.append(im.convert('RGB'))
        batch=processor(images=images,return_tensors='pt').to(device)
        with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):result=model.get_image_features(**batch)
        return F.normalize(result.float(),dim=1).cpu()
    data={};started=time.perf_counter();batchsize=16
    for name,rows in pools.items():
        folder=featuredir/name;folder.mkdir(exist_ok=True);blocks=[];done=0
        for p in sorted(folder.glob('batch_*.pt')):
            r=torch.load(p,map_location='cpu',weights_only=False)
            assert r['start']==done and r['manifest_sha256']==digest_file(mp)
            blocks.append(r['features']);done+=len(r['features'])
        while done<len(rows):
            check_pause()
            try:block=encode(rows[done:done+batchsize])
            except torch.cuda.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache()
                if batchsize==1:raise
                batchsize=max(1,batchsize//2);continue
            atomic_torch(dict(start=done,features=block,manifest_sha256=digest_file(mp)),folder/f'batch_{done:06d}.pt')
            blocks.append(block);done+=len(block)
            if done%320==0 or done==len(rows):print('SIGLIP FEATURES',name,done,'/',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
        data['E_'+name]=torch.cat(blocks);data[name+'_y']=np.array([r['class'] for r in rows])
    texts=[]
    for template in TEMPLATES:
        check_pause();batch=tokenizer([template.format(n) for n in source['class_names']],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(device)
        chunks=[]
        for a in range(0,len(source['class_names']),32):
            with torch.autocast(device_type=device,enabled=device=='cuda',dtype=torch.float16):v=model.get_text_features(**{k:x[a:a+32] for k,x in batch.items()})
            chunks.append(F.normalize(v.float(),dim=1).cpu())
        texts.append(torch.cat(chunks))
    data['T100']=F.normalize(torch.stack(texts).mean(0),dim=1)
    data['meta']={k:v for k,v in source.items() if k!='pools'};data['feature_manifest_sha256']=digest_file(mp)
    witness=encode(pools['test'][:8]);cos=F.cosine_similarity(witness,data['E_test'][:8]);assert float(cos.min())>.9999
    # Normalized embeddings must reproduce the native model's affine pair logits.
    native_scale=float(model.logit_scale.exp());native_bias=float(model.logit_bias)
    atomic_torch(data,featuredir/'frozen.pt')
    atomic_json(dict(complete=True,cache_sha256=digest_file(featuredir/'frozen.pt'),minimum_reembedding_cosine=float(cos.min()),
                     image_counts={k:len(v) for k,v in pools.items()},feature_shapes={k:list(v.shape) for k,v in data.items() if torch.is_tensor(v)},
                     native_scale=native_scale,native_bias=native_bias,elapsed_seconds=time.perf_counter()-started,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0),featuredir/'verification.json')
    print('SIGLIP FEATURE CACHE COMPLETE',flush=True)
if __name__=='__main__':main()
