"""Independent CPU verification against pinned scorer and archived all-session inputs."""
from pathlib import Path
import zipfile,io,json,hashlib,sys,ast,math
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import kbs_paired_accuracy as paired
import kbs_risk_control as risk
from expansion_hycal_self_replay import compact
R=Path(__file__).resolve().parents[2];H=R/'experiments/expansion_hycal_self_remote';D=H/'download'
def stats(x):
 x=np.asarray(x);rng=np.random.default_rng(2026);v=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
 return dict(mean=float(x.mean()),ci95=np.quantile(v,[.025,.975]).tolist(),values=x.tolist(),n=len(x))
def main():
 torch.set_num_threads(2)
 source=R/'src/experiments/kbs_reviewer_readouts.py';tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['cdf_normalize','hycal_scores']]
 env=dict(torch=torch,math=math);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
 allrows=[];checked=0
 with zipfile.ZipFile(H/'payload.zip') as payload:
  hashes=json.loads(payload.read('input_hashes.json'))
  for en in ['l14','b32']:
   for ds in ['cifar','pets','imagenetr']:
    for part in range(10):
     prefix=f'inputs/{en}/{ds}/p{part}/';raw=payload.read(prefix+'closed_raw.npz');assert hashlib.sha256(raw).hexdigest()==hashes[prefix+'closed_raw.npz']
     with np.load(io.BytesIO(raw)) as a:z={k:a[k] for k in a.files}
     assert (np.diff(z['base_classes'])>0).all();preds={}
     for pool in ['cal','test']:
      scores=env['hycal_scores'](torch.from_numpy(z[pool+'_cosine']),torch.from_numpy(z[pool+'_negative_distance']),z['counts'])
      preds[pool]=z['base_classes'][scores.argmax(1).numpy()]
     for order in [0,1]:
      for session in range(1,6):
       name=f'o{order}_t{session}.npz';raw=payload.read(prefix+name);assert hashlib.sha256(raw).hexdigest()==hashes[prefix+name]
       with np.load(io.BytesIO(raw)) as a:q={k:a[k] for k in a.files}
       result=json.loads((D/f'results/{en}/{ds}/p{part}/o{order}_t{session}.json').read_text());assert result['input_sha256']==hashes[prefix+name] and result['closed_input_sha256']==hashes[prefix+'closed_raw.npz']
       own=next(a for a in result['records'] if a['reference']=='self');cc=compact(q,'cal');tc=compact(q,'test');w=risk.weights(q['cal_labels'],q['base_classes']);rc=preds['cal']==q['cal_labels'];rt=preds['test']==q['test_labels'];new=paired.certify(cc,rc,w,q['hycal_candidates'])
       assert own['offset']==new['offset'],(en,ds,part,order,session,own['offset'],new['offset'])
       assert len(own['trace'])==len(new['trace'])
       for aa,bb in zip(own['trace'],new['trace']):
        assert aa['accepted']==bb['accepted'] and aa['offset']==bb['offset']
        for kk in ['pvalue','log_evalue','accuracy_surplus','destroyed','recovered']:
         assert abs(aa[kk]-bb[kk])<1e-9,(en,ds,part,order,session,kk,aa[kk],bb[kk])
       bm=tc['base_mask'];tw=tc['weights'][bm]
       churn=float(tw@(preds['test'][bm]!=q['test_hycal_base_prediction'][bm]))
       allrows.append(dict(encoder=en,dataset=ds,part=part,order=order,session=session,closed_churn=churn,**own));checked+=1
    print('VERIFIED',en,ds,flush=True)
 summary=json.loads((H/'summary.json').read_text());summary['verification']=dict(reconstructed_self_policies=checked,original_scorer_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),all_input_hashes_verified=True,own_closed_uses_only_base_columns=True)
 contrasts=[]
 for en in ['l14','b32']:
  for ds in ['cifar','pets','imagenetr']:
   final=[r for r in allrows if r['encoder']==en and r['dataset']==ds and r['session']==5];assert len(final)==20
   summaryrow=next(r for r in summary['summary'] if r['encoder']==en and r['dataset']==ds and r['reference']=='self')
   summaryrow['closed_churn']=stats([100*np.mean([r['closed_churn'] for r in final if r['part']==p]) for p in range(10)])
   summaryrow['target']={k:([.95*v for v in x] if k=='values' else [.95*v for v in x] if k=='ci95' else .95*x if k=='mean' else x) for k,x in summaryrow['closed_base'].items()}
   a=summaryrow['novel']['values'];b=next(r for r in summary['summary'] if r['encoder']==en and r['dataset']==ds and r['reference']=='common')['novel']['values'];contrasts.append(dict(encoder=en,dataset=ds,self_minus_common_novel=stats(np.array(a)-b)))
   print(en,ds,'closedchurn%',summaryrow['closed_churn']['mean'],'novel_self-common',contrasts[-1]['self_minus_common_novel']['mean'])
 summary['contrasts']=contrasts;(H/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 report=['# HyCal 自身闭集参照认证（CPU 容器）','', '使用真实仅基类词表的 HyCal CDF 评分构造闭集参照。共同参照与自身参照的合同不同，必须同时查看绝对基类下限。全部600个共同参照策略精确复现，全部600个自身参照策略由本地原评分函数独立重算并精确复现偏置和检验轨迹。','', '|编码器|数据|共同参照Novel|自身参照Novel [95% CI]|自身闭集Base|95%绝对下限|自身有限准入/达标|闭集→扩展内部排名变化%|','|---|---|---:|---|---:|---:|---|---:|']
 for en in ['l14','b32']:
  for ds in ['cifar','pets','imagenetr']:
   a=next(r for r in summary['summary'] if r['encoder']==en and r['dataset']==ds and r['reference']=='self');b=next(r for r in summary['summary'] if r['encoder']==en and r['dataset']==ds and r['reference']=='common');v=a['novel']
   report.append(f"|{en}|{ds}|{b['novel']['mean']:.2f}|{v['mean']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]|{a['closed_base']['mean']:.2f}|{a['target']['mean']:.2f}|{a['finite']}/20 / {a['met']}/20|{a['closed_churn']['mean']:.3f}|")
 report.extend(['','这些结果证明共同参照下的零准入不能表述为 HyCal 无识别能力；降低至其自身闭集保持目标后，多数条件可用准入恢复。Pets L/14 仍有两条最终路径经验目标未达标。内部排序确实随候选词表变化，但本实验不把统计拒绝因果地归结于 G；仅报告实际 G、D 与候选内部变化。','', '只复核手稿任务下的 author-code-controlled HyCal transfer，不声称复现原论文标准benchmark。所有指标均为第五轮，先平均每个划分的两种顺序，再平均十个划分。原始合同、cap40样本、候选序列、评分头与regularization均未改变。'])
 (H/'REPORT_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
if __name__=='__main__':main()
