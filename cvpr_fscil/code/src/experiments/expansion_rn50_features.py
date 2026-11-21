"""Native CLIP RN50 encoding of the fixed ImageNet-R identities."""
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
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_rn50'

@torch.inference_mode()
def main():
    device=configure();check_pause();import clip,clip.clip,clip.model,clip.simple_tokenizer
    modeldir=OUT/'model';featuredir=OUT/'features';featuredir.mkdir(exist_ok=True)
    mm=modeldir/'manifest.json';modelmanifest=json.loads(mm.read_text())
    weight=modeldir/'RN50.pt';assert digest_file(weight)==modelmanifest['sha256']
    src=ROOT/'experiments/kbs_imagenetr/data_manifest.json';calp=ROOT/'experiments/kbs_calibration_budget/pools/imagenetr.json'
    source=json.loads(src.read_text());cal=json.loads(calp.read_text())
    pools={k:list(v) for k,v in source['pools'].items()}
    pools['extra']=[dict(path=r['identity'],**{'class':r['class_id'],'sha256':r['sha256']}) for r in cal['rows'] if 'cached_index' not in r]
    modules=[clip.clip,clip.model,clip.simple_tokenizer]
    config=dict(model_manifest_sha256=digest_file(mm),data_manifest_sha256=digest_file(src),
                calibration_pool_sha256=digest_file(calp),source_sha256=digest_file(Path(__file__)),
                protocol_sha256=digest_file(ROOT/'experiments/expansion_rn50_protocol.md'),
                clip_source_sha256={Path(m.__file__).name:digest_file(Path(m.__file__)) for m in modules},
                templates=TEMPLATES,preprocessing='native CLIP Resize224 BICUBIC/CenterCrop224/RGB/ToTensor/CLIP mean/std; native context77 tokenizer')
    mp=featuredir/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==config
    else:atomic_json(config,mp)
    if (featuredir/'verification.json').exists():
        assert digest_file(featuredir/'frozen.pt')==json.loads((featuredir/'verification.json').read_text())['cache_sha256'];return
    model,preprocess=clip.load(str(weight),device=device,jit=False);model.eval()
    assert type(model.visual).__name__=='ModifiedResNet' and int(model.visual.input_resolution)==224
    def pixels(rows):
        images=[]
        for row in rows:
            path=CFG.data_root/'imagenet-r'/row['path'];assert digest_file(path)==row['sha256']
            with Image.open(path) as im:images.append(preprocess(im.convert('RGB')))
        return torch.stack(images).to(device)
    def encode(rows):return F.normalize(model.encode_image(pixels(rows)).float(),dim=1).cpu()
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
            if done%320==0 or done==len(rows):print('RN50 FEATURES',name,done,'/',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
        data['E_'+name]=torch.cat(blocks);data[name+'_y']=np.array([r['class'] for r in rows])
    texts=[]
    for template in TEMPLATES:
        check_pause();batch=clip.tokenize([template.format(n) for n in source['class_names']],context_length=77,truncate=False).to(device)
        chunks=[]
        for a in range(0,len(batch),32):chunks.append(F.normalize(model.encode_text(batch[a:a+32]).float(),dim=1).cpu())
        texts.append(torch.cat(chunks))
    data['T100']=F.normalize(torch.stack(texts).mean(0),dim=1)
    data['meta']={k:v for k,v in source.items() if k!='pools'};data['feature_manifest_sha256']=digest_file(mp)
    witness=encode(pools['test'][:8]);cos=F.cosine_similarity(witness,data['E_test'][:8]);assert float(cos.min())>.9999
    imgs=pixels(pools['test'][:8]);tokens=clip.tokenize([TEMPLATES[0].format(n) for n in source['class_names'][:8]]).to(device)
    raw_i=model.encode_image(imgs);raw_t=model.encode_text(tokens)
    ni=raw_i/raw_i.norm(dim=-1,keepdim=True);nt=raw_t/raw_t.norm(dim=-1,keepdim=True)
    reconstructed=model.logit_scale.exp()*ni@nt.T;native,_=model(imgs,tokens)
    logit_error=float((native-reconstructed).abs().max());assert logit_error<1e-5
    assert data['E_fit'].shape[1]==1024 and torch.isfinite(data['T100']).all()
    atomic_torch(data,featuredir/'frozen.pt');cache_sha=digest_file(featuredir/'frozen.pt')
    atomic_json(dict(complete=True,cache_sha256=cache_sha,minimum_reembedding_cosine=float(cos.min()),
                     image_counts={k:len(v) for k,v in pools.items()},feature_shapes={k:list(v.shape) for k,v in data.items() if torch.is_tensor(v)},
                     elapsed_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0),featuredir/'verification.json')
    atomic_json(dict(complete=True,cache_sha256=cache_sha,native_logit_max_absolute_error=logit_error,
                     minimum_reembedding_cosine=float(cos.min()),visual_backbone=type(model.visual).__name__,
                     text_context=77,native_tokenizer=True,preprocessing=repr(preprocess),
                     common_cosine_interface='Native CLIP global scale omitted consistently from every readout.'),featuredir/'native_witness.json')
    print('RN50 FEATURE CACHE COMPLETE',flush=True)
if __name__=='__main__':main()
