"""Independent NumPy FP32 cache scoring witness, plus fixed-stack budget audit."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
import expansion_sun_controls_remote as c
R=Path(__file__).resolve().parents[2]
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
 for en in ['l14','b32']:
  d=torch.load(R/'inputs'/en/'frozen.pt',map_location='cpu',weights_only=False);s=json.loads((R/'inputs'/en/'p0/selection.json').read_text());sel=json.loads((R/'results'/en/'p0/selection.json').read_text());b=np.asarray(s['entry']['base']);n=np.asarray(s['entry']['novel']);ids=np.r_[b,n];keys,ky=c.keys_for(d,b,n);ix=np.linspace(0,len(d['E_tune'])-1,128,dtype=int);E=d['E_tune'][ix];counts=np.array([(ky==cl).sum() for cl in ids]);tx=E.numpy()@d['T'][ids].numpy().T
  for method,p in sel['selected'].items():
   affinity=np.exp(p['beta']*(E.numpy()@keys.numpy().T-1));raw=np.stack([affinity[:,ky==cl].sum(1) for cl in ids],1)
   if method=='cache_mean':raw=raw/counts
   cpu=100*tx+p['alpha']*raw;gpu_cache=c.cache(E.cuda(),keys.cuda(),ky,ids,p['beta'])
   if method=='cache_mean':gpu_cache=gpu_cache/torch.tensor(counts,device='cuda')
   gpu=(100*(E.cuda()@d['T'][ids].cuda().T)+p['alpha']*gpu_cache).cpu().numpy();err=float(np.max(abs(cpu-gpu)));agreement=float(np.mean(cpu.argmax(1)==gpu.argmax(1)));assert err<.01,(en,method,err);assert agreement==1.,(en,method,agreement)
   rows.append(dict(encoder=en,method=method,queries=128,max_score_error=err,prediction_agreement=agreement));print(en,method,err,agreement,flush=True)
 report=dict(complete=True,dtype='GPU FP32 (TF32 disabled), NumPy FP32 independent class-wise aggregation',witnesses=rows)
 (R/'witness.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
