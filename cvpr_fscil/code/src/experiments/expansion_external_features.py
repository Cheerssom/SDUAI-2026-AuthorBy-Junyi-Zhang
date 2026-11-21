"""Identity-audited new-domain pools and local-weight CLIP feature shards."""
from pathlib import Path
import argparse,json,sys,time,hashlib,gc,tarfile,io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fcs_pets_data import TEMPLATES
from kbs_calibration_pool import REVISIONS
from kbs_runtime import configure,check_pause,atomic_json,atomic_torch,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_prospective_validation/external';DATA=Path('E:/expansion_external_data')

def food_inventory():
    root=DATA/'food-101';archive=DATA/'food-101.tar.gz';receipt=DATA/'food_extracted.json'
    if not receipt.exists():
        assert archive.exists() and archive.stat().st_size==4996278331
        with archive.open('rb') as source:assert hashlib.file_digest(source,'md5').hexdigest()=='85eeb15f3717b99a5da872d97d918f87','Food official archive checksum mismatch'
        with tarfile.open(archive,'r:gz') as tar:
            for i,m in enumerate(tar):
                if not m.isfile():continue
                target=DATA/m.name
                assert target.resolve().is_relative_to(DATA.resolve()) and not m.issym() and not m.islnk()
                if target.exists() and target.stat().st_size==m.size:continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as src,target.open('wb') as dst:
                    while block:=src.read(2**20):dst.write(block)
                if i%10000==0:print('FOOD EXTRACT',i,flush=True)
        atomic_json(dict(complete=True,archive_sha256=digest_file(archive)),receipt)
    names=(root/'meta/classes.txt').read_text().splitlines();lookup={s:i for i,s in enumerate(names)};rows=[]
    for role in ['train','test']:
        for item in (root/f'meta/{role}.txt').read_text().splitlines():
            p=root/'images'/f'{item}.jpg';rows.append(dict(path=str(p),class_id=lookup[item.split('/')[0]],origin=role,identity='food/'+item,sha256=digest_file(p)))
    return names,rows

def sun_inventory():
    import pyarrow.parquet as pq
    root=DATA/'sun397';receipt=root/'inventory.json'
    if receipt.exists():return json.loads(receipt.read_text())['names'],json.loads(receipt.read_text())['rows']
    manifest=json.loads((root/'download_manifest.json').read_text());names=None;rows=[]
    # Every source shard must be fully present and verified before split assignment.
    for f in manifest['files']:
        p=root/Path(f['path']).name
        assert p.exists() and p.stat().st_size==f['size'],str(p)
        assert digest_file(p)==f['lfs']['oid']
    for f in manifest['files']:
        path=root/Path(f['path']).name;parquet=pq.ParquetFile(path)
        if names is None:
            metadata=json.loads(parquet.schema_arrow.metadata[b'huggingface']);names=metadata['info']['features']['label']['names']
        shard=root/'inventory_shards'/f'{path.stem}.json'
        if shard.exists():rows+=json.loads(shard.read_text());continue
        found=[];index=0
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                im=row['image'];raw=im['bytes'];sha=hashlib.sha256(raw).hexdigest()
                # Store selected-domain images once; never decode/recompress source bytes.
                p=root/'images'/f'{sha}.jpg'
                if not p.exists():
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
                found.append(dict(path=str(p),class_id=int(row['label']),origin=path.name,identity=f'sun/{path.stem}/{index}',sha256=sha));index+=1
        atomic_json(found,shard);rows+=found;print('SUN INVENTORY',path.name,len(rows),flush=True)
    assert len(names)==397 and len(rows)>100000
    atomic_json(dict(names=names,rows=rows,source_revision=manifest['revision']),receipt);return names,rows

def pools(ds):
    dest=OUT/ds/'data_manifest.json'
    if dest.exists():return json.loads(dest.read_text())
    names,rows=food_inventory() if ds=='food' else sun_inventory();seen=set();unique=[];duplicates=[]
    # Food training listed before test,ensuring byte identity never crosses roles.
    for row in rows:
        if row['sha256'] in seen:duplicates.append(row);continue
        seen.add(row['sha256']);unique.append(row)
    K=len(names);pieces={k:[] for k in ['fit','tune','cal','support','test']}
    for c in range(K):
        rng=np.random.default_rng(28102026+c)
        if ds=='food':
            training=[r for r in unique if r['class_id']==c and r['origin']=='train'];test=[r for r in unique if r['class_id']==c and r['origin']=='test'];training=[training[i] for i in rng.permutation(len(training))]
            cal,tun,sup,fit=training[:40],training[40:60],training[60:65],training[65:]
        else:
            images=[r for r in unique if r['class_id']==c];images=[images[i] for i in rng.permutation(len(images))];nt=max(20,int(len(images)*.2))
            if len(images)-nt-55<2:nt=20
            test=images[:nt];rest=images[nt:];cal,tun,sup,fit=rest[:40],rest[40:50],rest[50:55],rest[55:]
        count=min(len(fit),max(2,round(200*100**(-c/(K-1)))))
        assert len(cal)==40 and len(sup)==5 and len(tun)>0 and count>=2 and len(test)>=20,(ds,c)
        for name,rr in [('fit',fit[:count]),('cal',cal),('tune',tun),('support',sup),('test',test)]:pieces[name]+=rr
    assert sum(map(len,pieces.values()))==len({r['sha256'] for v in pieces.values() for r in v})
    result=dict(dataset=ds,names=[n.replace('_',' ').removeprefix('/').split('/',1)[-1].replace('/',' ') for n in names],original_names=names,pools=pieces,duplicates_removed=duplicates,protocol_sha256=digest_file(ROOT/'experiments/expansion_prospective_validation_protocol.md'),templates=TEMPLATES)
    atomic_json(result,dest);print('NEW POOLS',ds,{k:len(v) for k,v in pieces.items()},flush=True);return result

@torch.inference_mode()
def extract(ds,enc,device):
    from transformers import CLIPModel,CLIPTokenizer,CLIPImageProcessor
    spec=pools(ds);folder=OUT/ds/enc;folder.mkdir(parents=True,exist_ok=True);target=folder/'frozen.pt'
    if (folder/'verification.json').exists():return
    modelname='large-patch14' if enc=='l14' else 'base-patch32';modelpath=Path.home()/f'.cache/huggingface/hub/models--openai--clip-vit-{modelname}/snapshots'/REVISIONS[enc]
    model=CLIPModel.from_pretrained(str(modelpath),local_files_only=True,use_safetensors=enc=='l14').to(device).eval();tokenizer=CLIPTokenizer.from_pretrained(str(modelpath),local_files_only=True)
    from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
    transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
    def image(row):
        with Image.open(row['path']) as im:return transform(im.convert('RGB'))
    result={};start=time.monotonic();batchsize=24;maxerr=0.
    with ThreadPoolExecutor(4) as ex:
        for role,rows in spec['pools'].items():
            blocks=[];done=0;shards=folder/role;shards.mkdir(exist_ok=True)
            for path in sorted(shards.glob('batch_*.pt')):
                z=torch.load(path,map_location='cpu',weights_only=False);assert z['start']==done;blocks.append(z['features']);done+=len(z['features'])
            while done<len(rows):
                check_pause();subset=rows[done:done+batchsize];xx=torch.stack(list(ex.map(image,subset))).to(device)
                try:
                    with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):z=model.visual_projection(model.vision_model(pixel_values=xx).pooler_output)
                except torch.cuda.OutOfMemoryError:
                    del xx;gc.collect();torch.cuda.empty_cache();batchsize=max(1,batchsize//2)
                    if batchsize==1:raise
                    continue
                z=F.normalize(z.float(),dim=1).cpu();atomic_torch(dict(start=done,features=z,identity_hash=core_hash([r['sha256'] for r in subset])),shards/f'batch_{done:06d}.pt');blocks.append(z);done+=len(z)
                if done%480==0 or done==len(rows):print('EXTERNAL FEATURES',enc,ds,role,done,'/',len(rows),'elapsed',round(time.monotonic()-start),flush=True)
            result['E_'+role]=torch.cat(blocks);result[('test_y' if role=='test' else role+'_y')]=np.array([r['class_id'] for r in rows])
    texts=[]
    for template in TEMPLATES:
        chunks=[]
        for i in range(0,len(spec['names']),64):
            tokens=tokenizer([template.format(n) for n in spec['names'][i:i+64]],padding=True,return_tensors='pt').to(device)
            chunks.append(F.normalize(model.text_projection(model.text_model(**tokens).pooler_output).float(),dim=1).cpu())
        texts.append(torch.cat(chunks))
    result['T']=F.normalize(torch.stack(texts).mean(0),dim=1);result['data_manifest_sha256']=digest_file(OUT/ds/'data_manifest.json');result['model_revision']=REVISIONS[enc]
    # Float32 model witness on eight independently reloaded fitting images.
    xx=torch.stack([image(r) for r in spec['pools']['fit'][:8]]).to(device);z=F.normalize(model.visual_projection(model.vision_model(pixel_values=xx).pooler_output).float(),dim=1).cpu();cos=F.cosine_similarity(z,result['E_fit'][:8]);assert cos.min()>.999
    atomic_torch(result,target);atomic_json(dict(complete=True,cache_sha256=digest_file(target),min_float32_witness_cosine=float(cos.min()),source_sha256=digest_file(Path(__file__)),rows={r:len(v) for r,v in spec['pools'].items()},seconds=time.monotonic()-start),folder/'verification.json')
    del model,result;gc.collect();torch.cuda.empty_cache()

def core_hash(values):return hashlib.sha256(json.dumps(values).encode()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['food','sun'],required=True);ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    if a.prepare_only:pools(a.dataset);return
    device=configure();check_pause()
    for enc in ([a.encoder] if a.encoder else ['l14','b32']):extract(a.dataset,enc,device)
if __name__=='__main__':main()
