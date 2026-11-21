"""Fixed-model reference-subset interventions on base-tuning queries only."""
from pathlib import Path
import json,sys,gc
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fcs_expansion as core
import kbs_reviewer_revision as previous
from kbs_runtime import configure,check_pause,atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_reference_cardinality'

def moments(x):return x.mean(1,keepdim=True),x.std(1,correction=0,keepdim=True).clamp_min(1e-6)
def margin(values,refs,indices,branch,nb,a,u):
    normal=[]
    for j,(x,ref) in enumerate(zip(values,refs)):
        sub=ref[:,indices] if branch in ['joint',str(j)] else ref
        mu,sd=moments(sub);normal.append((x-mu)/sd)
    b=normal[0][:,:nb]+a*normal[1];n=normal[0][:,nb:]+u*normal[2][:,nb:]
    return n.max(1).values-b.max(1).values

@torch.no_grad()
def main():
    device=configure();OUT.mkdir(exist_ok=True)
    config={'source_sha256':digest_file(Path(__file__)),'protocol_sha256':digest_file(ROOT/'experiments/expansion_shrinkage_protocol.md')}
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==config
    else:atomic_json(config,mp)
    if (OUT/'complete.json').exists():return
    for enc in ['l14','b32']:
        for ds in ['pets','cifar','imagenetr']:
            d,_=previous.load_data(ds,enc);Ef,Es,T=[d[k].to(device) for k in ['E_fit','E_support','T']]
            entries=json.loads((ROOT/f'experiments/kbs_revision/{enc}/{ds}/partitions.json').read_text())
            for entry in entries:
                check_pause();out=OUT/enc/ds/f"p{entry['seed']}.json"
                if out.exists():continue
                p=entry['seed'];source=ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{p}/selection.json'
                old=json.loads(source.read_text());qsource=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p{p}/selection.json'
                tuned=json.loads(qsource.read_text());rows=[]
                for stage in ['pseudo','full']:
                    base=np.asarray(entry['pseudo_base' if stage=='pseudo' else 'base'])
                    novel=np.asarray(entry['pseudo_novel' if stage=='pseudo' else 'novel']);nb=len(base)
                    W=core.fit_ridge(Ef,d['fit_y'],base,old['pseudo_ridge_lambda' if stage=='pseudo' else 'ridge_lambda'])
                    Pb=core.means(Ef,d['fit_y'],base);si=core.support_rows(d['support_y'],novel,5,7)
                    Pn=core.means(Es[si],d['support_y'][si],novel)
                    keep=np.isin(d['tune_y'],entry['base']);choices=np.flatnonzero(keep)
                    ix=np.random.default_rng(27102026+p).choice(choices,min(128,len(choices)),replace=False)
                    E=d['E_tune'][ix].to(device);ids=np.r_[base,novel]
                    values=[E@T[ids].T,core.linear(E,W),E@torch.cat([Pb,Pn]).T]
                    refs=[values[0][:,:nb],values[1],values[2][:,:nb]]
                    par=tuned['selections']['retained_query']['selected'];a,u=par['a'],par['u']
                    full=margin(values,refs,np.arange(nb),'joint',nb,a,u);fullmom=[moments(x) for x in refs]
                    sizes=sorted(set([min(k,nb) for k in [2,4,7,15,nb]]))
                    for seed in range(20):
                        order=np.random.default_rng(seed).permutation(nb)
                        for size in sizes:
                            ixref=order[:size]
                            for bank in ['0','1','2','joint']:
                                m=margin(values,refs,ixref,bank,nb,a,u)
                                mrms=float((m-full).square().mean().sqrt())
                                banks=range(3) if bank=='joint' else [int(bank)]
                                muerr=[];sderr=[]
                                for j in banks:
                                    mu,sd=moments(refs[j][:,ixref]);fmu,fsd=fullmom[j]
                                    muerr.append(float(((mu-fmu)/fsd).square().mean().sqrt()))
                                    sderr.append(float(((sd-fsd)/fsd).square().mean().sqrt()))
                                rows.append(dict(encoder=enc,dataset=ds,part=p,stage=stage,reference_count=nb,
                                                 subset_count=size,seed=seed,bank=bank,
                                                 normalized_mean_rms=float(np.mean(muerr)),relative_scale_rms=float(np.mean(sderr)),
                                                 margin_rms=mrms))
                atomic_json(dict(source_sha256=digest_file(source),tuning_sha256=digest_file(qsource),records=rows),out)
            del d,Ef,Es,T;gc.collect();torch.cuda.empty_cache()
            print('REFERENCE DIAGNOSTIC COMPLETE',enc,ds,flush=True)
    atomic_json(dict(complete=True,conditions=6,partitions=60,scope='Deterministic reference-subset perturbation; not iid moment-variance asymptotics.'),OUT/'complete.json')
if __name__=='__main__':main()
