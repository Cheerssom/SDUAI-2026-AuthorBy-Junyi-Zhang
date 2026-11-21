"""Minimal GPU runner reusing frozen cardinality fits for prespecified C10/C50."""
from pathlib import Path
import argparse,ast,types,json,os,hashlib,shutil,sys,gc,time,math
import numpy as np
import torch
import torch.nn.functional as F
R=Path(__file__).resolve().parents[2];OUT=R/'results';ROOT=R
sys.path.insert(0,str(Path(__file__).parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
from kbs_runtime import atomic_json,atomic_npz,atomic_torch,digest_file

def check_pause():
 if (R/'PAUSE').exists():print('PAUSED',flush=True);raise SystemExit(0)
def functions(file,names,env):
 tree=ast.parse((R/'frozen_sources'/file).read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
 assert len(nodes)==len(names),(file,names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),env)
 return types.SimpleNamespace(**{k:env[k] for k in names})
def predict(scores,ids):return np.asarray(ids)[scores.argmax(1).cpu().numpy()]
env=dict(np=np,torch=torch,F=F,predict=predict,check_pause=check_pause,json=json)
core=functions('fcs_expansion.py',['norm','means','support_rows','linear'],env)
env.update(core=core)
rv=functions('fcs_retention_value.py',['compact','evaluate','threshold'],env)
env.update(rv=rv,GRID=(0.,.25,.5,1.,2.,4.))
tune=functions('kbs_full_fusion_retuning.py',['scores','tuning_scores','select','pack'],env)
env.update(tune=tune,paired=paired,risk=risk)
fairness=functions('kbs_calibration_fairness.py',['evaluate_policy'],env)
def roc_auc_score(y,score,sample_weight):
 y=np.asarray(y,bool);s=np.asarray(score);w=np.asarray(sample_weight);ix=np.argsort(s,kind='stable');y=y[ix];s=s[ix];w=w[ix]
 _,starts=np.unique(s,return_index=True);pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*~y,starts)
 return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(pos.sum()*neg.sum()))
env.update(fairness=fairness,roc_auc_score=roc_auc_score,ROOT=ROOT,OUT=OUT,Path=Path,gc=gc,time=time,math=math,atomic_torch=atomic_torch,atomic_npz=atomic_npz,atomic_json=atomic_json,digest_file=digest_file)
# Reuse the locked mechanisms and exact task evaluator, substituting only requested C values.
tree=ast.parse((R/'frozen_sources/expansion_prospective_validation.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['raw','comps','score','makecompact','auc','frontier','run_task']]
text=ast.unparse(ast.Module(body=nodes,type_ignores=[]));text=text.replace("['C15', 'C20', 'C30', 'C40', 'C60']","['C10', 'C50']")
exec(compile(text,'frozen-cardinality-evaluator','exec'),env)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--encoder',choices=['l14','b32']);ap.add_argument('--parts',type=int,default=10);a=ap.parse_args();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35);torch.manual_seed(28102026)
 jobs=0
 for enc in ([a.encoder] if a.encoder else ['l14','b32']):
  cache=R/'inputs'/enc/'frozen.pt';d=torch.load(cache,map_location='cpu',weights_only=False);d={k:(v.numpy() if k.endswith('_y') else v) for k,v in d.items()}
  for src in sorted((R/'inputs'/enc/'tasks').glob('B*/p*/selection.json')):
   original=json.loads(src.read_text());entry=original['entry'];part=entry['seed'];B=len(entry['base'])
   if part>=a.parts:continue
   check_pause();dest=OUT/'cardinality'/enc/'imagenetr'/f'B{B}'/f'p{part}';dest.mkdir(parents=True,exist_ok=True)
   if (dest/'complete.json').exists():jobs+=1;continue
   if not (dest/'selection.json').exists():
    b,n,pb,pn=[np.array(entry[k]) for k in ['base','novel','pseudo_base','pseudo_novel']];weights=torch.load(src.parent/'weights.pt',map_location='cuda',weights_only=False);Wp=weights['Wp'];Ef=d['E_fit'].cuda();Es=d['E_support'].cuda();T=d['T'].cuda();yf=d['fit_y'];ys=d['support_y'];mask=np.isin(d['tune_y'],b);Ev=d['E_tune'][mask].cuda();yv=d['tune_y'][mask]
    Ppb=core.means(Ef,yf,pb);si=core.support_rows(ys,pn,5,7);Ppn=core.means(Es[si],ys[si],pn)
    with torch.no_grad():
     raw=env['raw'](Ev,T[np.r_[pb,pn]],Ppb,Ppn,Wp);selections={'C30':original['selections']['C30']}
     for C in [10,50]:
      pc=env['comps'](raw,original['pseudo_moments'],'full',C,11+part*100);selections[f'C{C}']=tune.select(pc,'retained',np.r_[pb,pn],yv,pb,pn);print('SELECT',enc,B,part,C,flush=True)
    shutil.copy2(src.parent/'weights.pt',dest/'weights.pt');selection=dict(original,selections=selections,extended_constants=[10,50],main_constant=30,parent_selection_sha256=digest_file(src),weights_sha256=digest_file(dest/'weights.pt'));atomic_json(selection,dest/'selection.json')
    del weights,Wp,Ef,Es,T,Ev,raw,pc;gc.collect();torch.cuda.empty_cache()
   env['run_task']('imagenetr',enc,'cardinality',entry,d,cache,'cuda');jobs+=1;atomic_json(dict(completed_tasks=jobs,last=[enc,B,part]),OUT/'status.json')
  del d;gc.collect();torch.cuda.empty_cache()
 atomic_json(dict(complete=True,completed_tasks=jobs,constants=[10,50],main_constant=30),OUT/'complete.json')
if __name__=='__main__':main()
