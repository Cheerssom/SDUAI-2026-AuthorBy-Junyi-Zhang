"""NumPy score witness independent of the runner's normalization implementation."""
from pathlib import Path
import sys,json,gc,argparse,hashlib
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kbs_reviewer_revision as prior
from kbs_runtime import atomic_json,digest_file
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_prospective_validation'
def load(ds,enc):
    if ds in ['food','sun']:return torch.load(OUT/f'external/{ds}/{enc}/frozen.pt',map_location='cpu',weights_only=False)
    return prior.load_data(ds,enc)[0]
def prototypes(E,y,classes):
    p=np.stack([E[y==c].mean(0) for c in classes]);return p/np.linalg.norm(p,axis=1,keepdims=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--external-only',action='store_true');args=ap.parse_args();torch.set_num_threads(2);results=[]
    assert args.partial or not args.external_only
    for block,ds in [('mechanism','cifar'),('mechanism','pets'),('mechanism','imagenetr'),('cardinality','cifar'),('cardinality','imagenetr'),('external','food'),('external','sun')]:
        if args.external_only and block!='external':continue
        for enc in ['l14','b32']:
            folders=sorted((OUT/block/enc/ds).glob('*/p0'))
            if not folders:
                if args.partial:continue
                raise AssertionError((block,ds,enc))
            # Refit cardinality extremes as well as the intact catalog.
            folders=folders if block=='cardinality' else folders[:1]
            d=load(ds,enc);Ef=d['E_fit'].numpy();Es=d['E_support'].numpy();T=d['T'].numpy();yf,ys=d['fit_y'],d['support_y']
            for folder in folders:
                jp=folder/'o0_t5.json'
                if not jp.exists():continue
                s=json.loads((folder/'selection.json').read_text());entry=s['entry'];b,n=[np.array(entry[k]) for k in ['base','novel']]
                W=torch.load(folder/'weights.pt',map_location='cpu',weights_only=False)['W'].numpy();Pb=prototypes(Ef,yf,b)
                rng=np.random.default_rng(7);support=np.concatenate([rng.permutation(np.flatnonzero(ys==c))[:5] for c in n]);order=np.random.default_rng(1000).permutation(n);Pn=prototypes(Es[support],ys[support],order);ids=np.r_[b,order];B=len(b)
                with np.load(jp.with_suffix('.npz')) as zz:z={k:zz[k] for k in zz.files}
                methods=list(s['selections']);errors={};shuffled=[]
                for pool in ['cal','test']:
                    y=d['cal_y'] if pool=='cal' else d['test_y'];mask=np.isin(y,b if pool=='cal' else np.r_[b,n]);E=d['E_'+pool][mask].numpy()
                    arrays=[E@T[ids].T,E@W[:-1]+W[-1],E@np.r_[Pb,Pn].T]
                    moments=[(v[:,:B].mean(1,keepdims=True),v[:,:B].std(1,keepdims=True).clip(1e-6)) for v in arrays]
                    assert np.max(abs(moments[1][0]-1/B))<1e-4
                    for method in methods:
                        C=float(method[1:]) if method.startswith('C') else 30;eta=min(1.,B/C);normalized=[];index=np.arange(len(E))
                        if method.startswith('shuffle'):
                            index=np.random.default_rng(int(method[7:])+(23 if pool=='cal' else 37)).permutation(len(E))
                            for mu,sd in moments:assert np.array_equal(np.sort(mu.ravel()),np.sort(mu[index].ravel())) and np.array_equal(np.sort(sd.ravel()),np.sort(sd[index].ravel()))
                            shuffled.append(dict(pool=pool,method=method,mapping_sha256=hashlib.sha256(index.tobytes()).hexdigest()))
                        for i,v in enumerate(arrays):
                            mu,sd=moments[i];fm,fs=s['full_moments'][i];mu=eta*mu[index]+(1-eta)*fm;sd=eta*sd[index]+(1-eta)*fs
                            if method=='none':mu=0.
                            if method=='fixed':mu,sd=fm,fs
                            if method=='pm_only':
                                sd=fs
                                if i!=2:mu=fm
                            if method=='scale_only' and i==2:mu=fm
                            if method=='minus_learned_scale' and i==1:sd=fs
                            if method=='minus_text_scale' and i==0:sd=fs
                            normalized.append((v-mu)/np.maximum(sd,1e-6))
                        a,u=[s['selections'][method]['selected'][k] for k in ['a','u']]
                        sb=normalized[0][:,:B]+a*(normalized[2][:,:B] if method=='dense' else normalized[1]);sn=normalized[0][:,B:]+u*normalized[2][:,B:]
                        margin=sb.max(1).astype(float)-sn.max(1).astype(float);sample=np.arange(min(128,len(E)));err=float(np.max(abs(margin[sample]-z[f'{pool}_{method}_margin'][sample])))
                        assert err<.005,(block,enc,ds,len(b),method,pool,err);errors[pool+'_'+method]=err
                        if method=='full' and s['gates']:
                            # Rebuild all12 router inputs from the explicit NumPy score equations.
                            qt,ql,qp=arrays;zt,zl,zp=normalized;st,sl,sp=[v[1] for v in moments];mp=moments[2][0]
                            assert B>=30,'New-domain full banks use eta=1'
                            prob=np.exp(zl-zl.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
                            ent=-(prob*np.log(np.maximum(prob,1e-12))).sum(1)/np.log(B);top=np.sort(sb,axis=1)
                            un=(qt[:,B:]/st+u*qp[:,B:]/sp).max(1)-(qt[:,:B]/st+a*ql/sl).max(1)
                            X=np.stack([sn.max(1)-sb.max(1),un,zt[:,B:].max(1)-zt[:,:B].max(1),zp[:,B:].max(1)-zp[:,:B].max(1),sb.max(1),sn.max(1),zl.max(1),np.log(sl[:,0]),mp[:,0],np.log(sp[:,0]),ent,top[:,-1]-top[:,-2]],axis=1).astype(float)
                            for kind,gate in s['gates'].items():
                                xx=(X-np.array(gate['mean']))/np.array(gate['scale']);model=gate['model']
                                if kind=='logistic':pred=xx@np.array(model['coef'])+model['intercept']
                                else:
                                    g=model['state'];h=np.tanh(xx@np.array(g['0.weight']).T+np.array(g['0.bias']));pred=(h@np.array(g['2.weight']).T+np.array(g['2.bias'])).ravel()
                                error=float(np.max(abs(-pred[sample]-z[f'{pool}_{kind}_margin'][sample])))
                                assert error<.005,(enc,ds,pool,kind,error);errors[pool+'_'+kind]=error
                results.append(dict(block=block,encoder=enc,dataset=ds,B=B,maximum_margin_error=max(errors.values()),errors=errors,shuffled_marginals_unchanged=shuffled))
            del d;gc.collect()
    target='external_feature_witness.json' if args.external_only else 'feature_witness.json'
    atomic_json(dict(complete=not args.partial,partial=args.partial,conditions=results,max_error=max(r['maximum_margin_error'] for r in results)),OUT/target)
    print('NUMPY SCORE WITNESS',len(results),max(r['maximum_margin_error'] for r in results),flush=True)
if __name__=='__main__':main()
