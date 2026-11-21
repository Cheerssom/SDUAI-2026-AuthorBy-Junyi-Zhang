"""Progressive calibration/deployment mismatch with immutable source models."""
from pathlib import Path
import argparse,json,sys,gc,time,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageFilter
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import fcs_retention_value as rv
import kbs_reviewer_revision as previous
import kbs_calibration_fairness as fairness
import kbs_risk_control as risk
from kbs_calibration_pool import REVISIONS
from kbs_runtime import configure,check_pause,atomic_json,atomic_torch,atomic_npz,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_prospective_validation/shift'
KINDS={'clean':('clean',0),'noise04':('noise',.04),'noise08':('noise',.08),'noise12':('noise',.12),'noise20':('noise',.20),'blur075':('blur',.75),'blur125':('blur',1.25),'blur2':('blur',2.),'blur3':('blur',3.)}

def pixel(x,kind,identity):
    group,value=KINDS[kind]
    if group=='clean':return x.copy()
    if group=='noise':
        seed=int(hashlib.sha256(('23102026/'+identity).encode()).hexdigest()[:16],16)
        return np.clip(np.rint(x.astype(float)+np.random.default_rng(seed).normal(0,value*255,x.shape)),0,255).astype(np.uint8)
    return np.asarray(Image.fromarray(x.transpose(1,2,0)).filter(ImageFilter.GaussianBlur(value))).transpose(2,0,1).copy()

@torch.inference_mode()
def extract(device,enc):
    from lt_dataset import _load_split
    from transformers import CLIPModel
    spec=json.loads((ROOT/'experiments/kbs_reviewer_revision/shift_features/manifest.json').read_text())
    tr,yr,_=_load_split('train');te,yt,_=_load_split('test');ci=spec['calibration_raw_ids'];ti=spec['test_raw_ids']
    pool=[('cal',int(i)) for i in ci]+[('test',int(i)) for i in ti];labels=np.r_[yr[ci],yt[ti]]
    OUT.mkdir(parents=True,exist_ok=True);mp=OUT/'feature_manifest.json'
    cfg=dict(protocol_sha256=digest_file(ROOT/'experiments/expansion_prospective_validation_protocol.md'),source_sha256=digest_file(Path(__file__)),original_manifest_sha256=digest_file(ROOT/'experiments/kbs_reviewer_revision/shift_features/manifest.json'),severities=KINDS,calibration_raw_ids=ci,test_raw_ids=ti)
    # JSON canonicalizes tuple severities.
    cfg=json.loads(json.dumps(cfg))
    if mp.exists():assert json.loads(mp.read_text())==cfg
    else:atomic_json(cfg,mp)
    modelname='large-patch14' if enc=='l14' else 'base-patch32';path=Path.home()/f'.cache/huggingface/hub/models--openai--clip-vit-{modelname}/snapshots'/REVISIONS[enc]
    model=CLIPModel.from_pretrained(str(path),local_files_only=True,use_safetensors=enc=='l14').to(device).eval()
    mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None];std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
    def encode(rows,kind):
        im=np.stack([pixel((tr if role=='cal' else te)[i],kind,f'{role}/{i}') for role,i in rows]);x=torch.from_numpy(im).to(device).float()/255
        x=F.interpolate(x,(224,224),mode='bicubic',align_corners=False)
        with torch.autocast(device,enabled=device=='cuda',dtype=torch.float16):z=model.visual_projection(model.vision_model(pixel_values=(x-mean)/std).pooler_output)
        return F.normalize(z.float(),dim=1).cpu()
    for kind in KINDS:
        folder=OUT/'features'/enc/kind;folder.mkdir(parents=True,exist_ok=True);target=folder/'features.pt'
        if (folder/'verification.json').exists():continue
        if kind in ['noise08','blur125']:
            old=ROOT/f'experiments/kbs_reviewer_revision/shift_features/{enc}/{kind}/features.pt';z=torch.load(old,map_location='cpu',weights_only=False)
            assert np.array_equal(z['labels'],labels) and z['calibration_count']==len(ci)
            ids=[0,len(ci),len(pool)-1];got=encode([pool[i] for i in ids],kind);cos=F.cosine_similarity(got,z['features'][ids]);assert cos.min()>.999
            atomic_json(dict(complete=True,reused=str(old.relative_to(ROOT)),sha256=digest_file(old),min_cosine=float(cos.min())),folder/'verification.json');continue
        blocks=[];done=0
        for p in sorted(folder.glob('batch_*.pt')):
            z=torch.load(p,map_location='cpu',weights_only=False);assert z['start']==done;blocks.append(z['features']);done+=len(z['features'])
        batch=16
        while done<len(pool):
            check_pause()
            try:z=encode(pool[done:done+batch],kind)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();gc.collect();batch=max(1,batch//2)
                if batch==1:raise
                continue
            atomic_torch(dict(start=done,features=z),folder/f'batch_{done:06d}.pt');blocks.append(z);done+=len(z)
            if done%640==0 or done==len(pool):print('PROGRESSIVE FEATURES',enc,kind,done,'/',len(pool),flush=True)
        atomic_torch(dict(features=torch.cat(blocks),labels=labels,calibration_count=len(ci),test_ids=np.array(ti)),target)
        atomic_json(dict(complete=True,sha256=digest_file(target),images=len(pool)),folder/'verification.json')
    del model;gc.collect();torch.cuda.empty_cache()

def features(enc,kind):
    base=OUT/'features'/enc/kind;v=json.loads((base/'verification.json').read_text())
    p=ROOT/v['reused'] if 'reused' in v else base/'features.pt';assert digest_file(p)==v['sha256'];return torch.load(p,map_location='cpu',weights_only=False)

@torch.no_grad()
def evaluate(device,enc):
    d,_=previous.load_data('cifar',enc);Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']];yf,ys=d['fit_y'],d['support_y']
    entries=json.loads((ROOT/f'experiments/kbs_revision/{enc}/cifar/partitions.json').read_text())
    for entry in entries:
        part=entry['seed'];po=OUT/'results'/enc/f'p{part}';po.mkdir(parents=True,exist_ok=True)
        if (po/'complete.json').exists():continue
        b,n=[np.array(entry[k]) for k in ['base','novel']];s=json.loads((ROOT/f'experiments/kbs_independent_risk/{enc}/cifar/p{part}/selection.json').read_text())
        W=core.fit_ridge(Ef,yf,b,s['ridge_lambda']);Pb=core.means(Ef,yf,b);Pn=core.means(Es[np.array(s['support_indices'])],ys[np.array(s['support_indices'])],n);ids=np.r_[b,n]
        cp=ROOT/f'experiments/kbs_independent_risk/{enc}/cifar/p{part}/o0_t5.npz'
        with np.load(cp) as zz:candidates={m:zz[m+'_candidates'] for m in ['retained','anchored_tp']}
        systems={};refs={};labels={}
        for kind in KINDS:
            data=features(enc,kind);nc=data['calibration_count']
            for pool,E,y in [('cal',data['features'][:nc],data['labels'][:nc]),('test',data['features'][nc:],data['labels'][nc:])]:
                if pool=='cal':mask=np.isin(y,b);E=E[mask];y=y[mask]
                E=E.to(device);arr=rv.arrays(E,T[ids],Pb,Pn,W);vals=arr['text'][:,:len(b)]+s['parameters']['retained'][0]*arr['learned']
                refs[kind,pool]=b[vals.argmax(1).cpu().numpy()]==y;labels[kind,pool]=y
                for m in candidates:systems[kind,pool,m]=rv.compact(rv.score(arr,m,s['parameters'][m]),ids,y,b,n)
        for kind in KINDS:
            jp=po/f'{kind}.json'
            if jp.exists():continue
            check_pause();yt=labels[kind,'test'];tref=refs[kind,'test'];closed=float(np.mean([tref[yt==c].mean() for c in b]));archive=dict(base_classes=b,novel_classes=n,test_labels=yt,test_reference_correct=tref);rows=[]
            for m in candidates:
                tc=systems[kind,'test',m]
                archive.update({'test_'+m+'_'+k:tc[k] for k in ['margin','base_prediction','novel_prediction']});archive[m+'_candidates']=candidates[m]
                for calibration,ck in [('source','clean'),('target',kind)]:
                    cc=systems[ck,'cal',m];ref=refs[ck,'cal'];yc=labels[ck,'cal'];wc=risk.weights(yc,b)
                    detail=fairness.evaluate_policy(cc,ref,wc,candidates[m],tc,tref,closed,'paired_accuracy')
                    archive.update({calibration+'_cal_'+m+'_'+k:cc[k] for k in ['margin','base_prediction','novel_prediction']});archive[calibration+'_cal_reference_correct']=ref;archive[calibration+'_cal_labels']=yc
                    rows.append(dict(method=m,calibration=calibration,reference_closed=closed,retention=detail['old']/max(closed,1e-12),violation_pp=max(0.,-detail['target_gap_pp']),**detail))
            rows.append(dict(method='reference',calibration='fallback',reference_closed=closed,old=closed,new=0.,retention=1.,violation_pp=0.,fallback=True,test_target_met=True,target_gap_pp=5*closed))
            raw=jp.with_suffix('.npz');atomic_npz(raw,**archive);atomic_json(dict(encoder=enc,part=part,kind=kind,group=KINDS[kind][0],severity=KINDS[kind][1],archive_sha256=digest_file(raw),records=rows),jp)
        atomic_json(dict(complete=True),po/'complete.json');print('SHIFT EVALUATED',enc,part,flush=True)
    del d,Ef,Es,T;gc.collect();torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','features','evaluate'],default='all');ap.add_argument('--encoder',choices=['l14','b32']);a=ap.parse_args();device=configure();check_pause()
    for enc in ([a.encoder] if a.encoder else ['l14','b32']):
        if a.stage in ['all','features']:extract(device,enc)
        if a.stage in ['all','evaluate']:evaluate(device,enc)
    if not a.encoder and a.stage=='all':atomic_json(dict(complete=True),OUT/'complete.json')
if __name__=='__main__':main()
