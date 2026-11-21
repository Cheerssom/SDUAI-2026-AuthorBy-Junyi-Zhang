"""Timing-only fits and complete cached-feature readout costs; immutable accuracy models."""
from pathlib import Path
import sys,json,time,gc
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import expansion_prospective_validation as p
import expansion_external_equal_budget as eq
from kbs_runtime import atomic_json,check_pause,digest_file
OUT=p.ROOT/'experiments/expansion_followup_diagnostics/router'
def timed(fn,n=30):
 for _ in range(5):fn()
 vals=[]
 for _ in range(n):
  t=time.perf_counter();fn();vals.append(1000*(time.perf_counter()-t))
 return dict(median_ms=float(np.median(vals)),p10_ms=float(np.quantile(vals,.1)),p90_ms=float(np.quantile(vals,.9)),repetitions=n)
def main():
 torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
 for enc in ['l14','b32']:
  dest=OUT/f'{enc}.json'
  if dest.exists():continue
  check_pause();folder=p.OUT/'external'/enc/'sun/catalog/p0';selection=json.loads((folder/'selection.json').read_text());d,cache=p.load_data('sun',enc)
  entry=selection['entry'];b,n,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']];w=torch.load(folder/'weights.pt',map_location='cpu',weights_only=False)
  Ef,Es,T=d['E_fit'],d['E_support'],d['T'];yf,ys=d['fit_y'],d['support_y'];Pb=p.core.means(Ef,yf,b);Ppb=p.core.means(Ef,yf,pb)
  ids=p.core.support_rows(ys,n,5,7);Pn=p.core.means(Es[ids],ys[ids],n)
  par=selection['selections']['full']['selected'];E=d['E_test'][:512];y=d['test_y'][:512]
  def full():
   r=p.raw(E,T[np.r_[b,n]],Pb,Pn,w['W']);comp=p.comps(r,selection['full_moments'],'full');sc=p.score(comp,'full',par)
   # Inference only: no label-dependent metric construction in timed code.
   c=(sc[:,:len(b)].max(1),sc[:,len(b):].max(1));return r,comp,c
  r,comp,c=full();X=p.gate_features(r,comp,par,c)
  result=dict(encoder=enc,dataset='sun',partition=0,device='CPU',torch_threads=2,batch=512,excludes_image_encoder=True,full=timed(full),rows=[],inputs={str((folder/'selection.json').relative_to(p.ROOT)):digest_file(folder/'selection.json')})
  m=np.isin(d['tune_y'],b);Ev=d['E_tune'][m];yv=d['tune_y'][m];si=p.core.support_rows(ys,pn,5,7);Ppn=p.core.means(Es[si],ys[si],pn)
  pr=p.raw(Ev,T[np.r_[pb,pn]],Ppb,Ppn,w['Wp']);pc=p.comps(pr,selection['pseudo_moments'],'full');vc=p.rv.compact(p.score(pc,'full',par),np.r_[pb,pn],yv,pb,pn);VX=p.gate_features(pr,pc,par,vc)
  for kind,state in selection['gates'].items():
   def complete():
    r,comp,c=full();xx=p.gate_features(r,comp,par,c);return p.gates.apply_gate(state,xx)
   gate=state['model'];count=(len(gate['coef'])+1 if kind=='logistic' else sum(np.asarray(v).size for v in gate['state'].values()))
   row=dict(kind=kind,trainable_router_parameters=int(count),standardization_constants=24,complete_readout=timed(complete),gate_forward=timed(lambda:p.gates.apply_gate(state,X)))
   xx=(VX-np.array(state['mean']))/np.array(state['scale']);labels=np.isin(yv,pn).astype(int);fit=[]
   for seed in [72000,72001,72002]:
    t=time.perf_counter();p.gates.fit_model(kind,xx,labels,p.gates.macro_weights(yv),state['selected']['reg'],seed);fit.append(time.perf_counter()-t)
   row['selected_setting_fit_seconds']=fit
   t=time.perf_counter();trial=eq.select_gate(kind,VX,vc,yv,pb,pn,72000);row['full_selection_seconds']=time.perf_counter()-t;assert trial['selection_settings']==3636
   result['rows'].append(row);print('COST',enc,kind,row['complete_readout'],row['full_selection_seconds'],flush=True)
  result['script_sha256']=digest_file(Path(__file__));atomic_json(result,dest);del d;gc.collect()
 atomic_json(dict(complete=True),OUT/'complete.json')
if __name__=='__main__':main()
