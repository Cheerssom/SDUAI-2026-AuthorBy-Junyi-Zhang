"""Build immutable HyCal own-closed-reference replay payload from raw branch caches."""
from pathlib import Path
import sys,json,hashlib,zipfile,io
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kbs_reviewer_revision as previous
R=Path(__file__).resolve().parents[2];O=R/'experiments/expansion_hycal_self_remote';O.mkdir(parents=True,exist_ok=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);manifest={}
 with zipfile.ZipFile(O/'payload.zip','w',zipfile.ZIP_STORED) as bundle:
  for name in ['expansion_hycal_self_replay.py','kbs_paired_accuracy.py','kbs_risk_control.py']:
   p=R/'src/experiments'/name;bundle.write(p,'src/experiments/'+name);manifest['src/experiments/'+name]=digest(p)
  bundle.write(O/'protocol.md','protocol.md')
  for enc in ['l14','b32']:
   for ds in ['cifar','pets','imagenetr']:
    d,_=previous.load_data(ds,enc);nt=len(d['tune_y']);nc=len(d['cal_y']);yt=d['test_y'];yc=d['cal_y'];cache={}
    folder=R/f'experiments/kbs_reviewer_revision/readouts/{enc}/{ds}'
    entries=json.loads((R/f'experiments/kbs_revision/{enc}/{ds}/partitions.json').read_text())
    for ent in entries:
     p=ent['seed'];po=folder/f'p{p}';sel=json.loads((po/'selection.json').read_text());reg=sel['hycal']['regularization'];base=np.array(ent['base'])
     if reg not in cache:cache[reg]=torch.load(folder/f'hycal_base_{reg:g}.pt',map_location='cpu',weights_only=False)
     a=cache[reg];mask=np.isin(yc,base);arrays={'base_classes':base,'counts':a['counts'][base]}
     for pool,sl,labels in [('cal',slice(nt,nt+nc),yc),('test',slice(nt+nc,nt+nc+len(yt)),yt)]:
      for name,key in [('cosine','cosine'),('negative_distance','negative_distance')]:
       v=a[key][sl][:,base].numpy()
       if pool=='cal':v=v[mask]
       arrays[pool+'_'+name]=v
      arrays[pool+'_labels']=labels[mask] if pool=='cal' else labels
     bio=io.BytesIO();np.savez_compressed(bio,**arrays);raw=bio.getvalue();name=f'inputs/{enc}/{ds}/p{p}/closed_raw.npz';bundle.writestr(name,raw);manifest[name]=hashlib.sha256(raw).hexdigest()
     for f in po.glob('o*_t*.npz'):
      with np.load(f) as z:
       data={k:z[k] for k in ['base_classes','novel_classes','cal_labels','test_labels','cal_reference_correct','test_reference_correct','hycal_candidates','cal_hycal_margin','cal_hycal_base_prediction','cal_hycal_novel_prediction','test_hycal_margin','test_hycal_base_prediction','test_hycal_novel_prediction']}
      assert np.array_equal(data['cal_labels'],arrays['cal_labels']) and np.array_equal(data['test_labels'],yt)
      bio=io.BytesIO();np.savez_compressed(bio,**data);raw=bio.getvalue();name=f'inputs/{enc}/{ds}/p{p}/{f.name}';bundle.writestr(name,raw);manifest[name]=hashlib.sha256(raw).hexdigest()
     manifest[f'source/{enc}/{ds}/p{p}/selection.json']=digest(po/'selection.json')
    print('PACKED',enc,ds,flush=True);del d,cache
  bundle.writestr('input_hashes.json',json.dumps(manifest,indent=2))
 print('PAYLOAD_MB',round((O/'payload.zip').stat().st_size/1e6,2),flush=True)
if __name__=='__main__':main()
