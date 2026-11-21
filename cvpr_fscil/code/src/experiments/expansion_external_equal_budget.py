"""Explicit new-domain wrapper: equal36x101 gate-selection budgets."""
from pathlib import Path
import json,sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import expansion_prospective_validation as p
from kbs_runtime import atomic_json,digest_file

def select_gate(kind,X,c,y,base,new,seed):
    g=p.gates;rng=np.random.default_rng(281000+seed);train=np.zeros(len(y),bool)
    for cl in np.unique(y):
        ix=rng.permutation(np.flatnonzero(y==cl));train[ix[:len(ix)//2]]=True
    valid=~train;sc=StandardScaler().fit(X[train]);xx=sc.transform(X);labels=np.isin(y,new).astype(int);vc=g.subcompact(c,valid,y,base,new);trials=[]
    grid=np.logspace(-3,3,36) if kind=='logistic' else np.logspace(-5,0,36)
    for reg in grid:
        model=g.fit_model(kind,xx[train],labels[train],g.macro_weights(y[train]),float(reg),seed)
        cc=dict(vc,margin=-g.predict_model(model,xx[valid]));offsets=np.quantile(cc['margin'],np.linspace(0,1,101));hs=p.tune.tuning_scores(cc,offsets);j=int(np.argmax(hs))
        trials.append(dict(reg=float(reg),harmonic=float(hs[j]),offset=float(offsets[j]),offsets=offsets.tolist(),harmonics=hs.tolist()))
    best=max(range(36),key=lambda j:trials[j]['harmonic']);chosen=trials[best];full=StandardScaler().fit(X)
    model=g.fit_model(kind,full.transform(X),labels,g.macro_weights(y),chosen['reg'],seed);h,offset=g.group_h(c,-g.predict_model(model,full.transform(X)))
    return dict(model=model,mean=full.mean_.tolist(),scale=full.scale_.tolist(),trials=trials,selected=chosen,refit_offset=offset,refit_harmonic=h,features=g.FEATURES,training_rows=int(train.sum()),validation_rows=int(valid.sum()),training_ids_sha256=p.core.digest(np.flatnonzero(train)),selection_settings=3636,candidate_models=36,within_group_winners='shared locked Full fusion;independent trained group gate',tuning_labels='base-only pseudo episode')

if __name__=='__main__':
    assert '--block' in sys.argv and sys.argv[sys.argv.index('--block')+1]=='external'
    root=p.OUT/'external';root.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),p.ROOT/'experiments/expansion_external_equal_budget_addendum.md']
    manifest=dict(sources={f.relative_to(p.ROOT).as_posix():digest_file(f) for f in files},gate_settings=3636,gate_models=36)
    dest=root/'equal_budget_manifest.json'
    if dest.exists():assert json.loads(dest.read_text())==manifest
    else:atomic_json(manifest,dest)
    p.gates.select_gate=select_gate
    p.main()
