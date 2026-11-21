from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'experiments/expansion_followup_diagnostics'
def read(folder):
 rows=[]
 for p in folder.rglob('p*.json'):rows+=json.loads(p.read_text())['records']
 return rows
def stats(values):
 a=np.array(values,float);rng=np.random.default_rng(28102026);means=a[rng.integers(len(a),size=(20000,len(a)))].mean(1)
 return dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(a))
def fmt(s):return f"{s['mean']:.2f} [{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}]"
def summarize():
 rows=read(H/'coverage');domain=read(H/'domain');cov=[];dom=[]
 for enc in ['l14','b32']:
  for B in [5,10,15,30]:
   for cap in sorted({5,10,20,40,150//B}):
    rr=[r for r in rows if (r['encoder'],r['B'],r['cap'])==(enc,B,cap)]
    if not rr:continue
    rec=dict(encoder=enc,B=B,cap=cap,cells=len(rr),actual_total_range=[min(r['total'] for r in rr),max(r['total'] for r in rr)],equal_total150=all(r['equal_total150'] for r in rr))
    metrics={k:[] for k in ['capacity','finite','novel','conditional_novel','met','base']}
    for part in range(10):
     a=[r for r in rr if r['part']==part];finite=[r for r in a if not r['policy']['fallback']]
     if not a:continue
     metrics['capacity'].append(100*np.mean([r['capacity']['new'] for r in a]));metrics['finite'].append(100*len(finite)/len(a));metrics['novel'].append(100*np.mean([r['policy']['new'] for r in a]));metrics['met'].append(100*np.mean([r['policy']['test_target_met'] for r in a]));metrics['base'].append(100*np.mean([r['policy']['old'] for r in a]))
     if finite:metrics['conditional_novel'].append(100*np.mean([r['policy']['new'] for r in finite]))
    rec.update({k:stats(v) if v else None for k,v in metrics.items()});cov.append(rec)
  for ds in ['food','sun']:
   for group in ['base','novel']:
    for stratum in ['all','base_winners_agree','base_winners_disagree']:
     rr=[r for r in domain if (r['encoder'],r['dataset'],r['group'],r['stratum'])==(enc,ds,group,stratum)]
     rec=dict(encoder=enc,dataset=ds,group=group,stratum=stratum)
     for key in ['mass','full_correct','dense_correct','correction','destruction','contribution','gain','full_auc','dense_auc']:
      vals=[]
      for part in range(10):
       a=[r for r in rr if r['part']==part]
       if key in ['full_correct','dense_correct','correction','destruction','gain']:
        a=[r for r in a if r['mass']>0]
        if not a:continue
       fun=(lambda r:r['mass']*(r['full_correct']-r['dense_correct'])) if key=='contribution' else ((lambda r:r['full_correct']-r['dense_correct']) if key=='gain' else lambda r:r[key])
       vals.append((1 if 'auc' in key else 100)*np.mean([fun(r) for r in a]))
      rec[key]=stats(vals)
     dom.append(rec)
 equal=[]
 for enc in ['l14','b32']:
  joint=set.intersection(*[{r['part'] for r in rows if r['encoder']==enc and r['B']==B and r['equal_total150']} for B in [5,10,15,30]])
  for B in [5,10,15,30]:
   rr=[r for r in rows if r['encoder']==enc and r['B']==B and r['equal_total150'] and r['part'] in joint]
   equal.append(dict(encoder=enc,B=B,cap=150//B,partitions=sorted(joint),finite=stats([100*np.mean([not r['policy']['fallback'] for r in rr if r['part']==part]) for part in sorted(joint)]),novel=stats([100*np.mean([r['policy']['new'] for r in rr if r['part']==part]) for part in sorted(joint)])))
 report=dict(coverage=cov,domain=dom,equal_total_joint_partitions=equal,raw_coverage_records=len(rows),raw_domain_records=len(domain),scope='partition bootstrap; subset/order repetitions averaged within each partition; conditional means use partitions with admission')
 (H/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# 固定模型补充诊断结果','', '所有区间以类别划分为重采样单位；顺序和校准子集先在划分内平均。条件Novel仅在有准入的划分中汇总，不把重复子集视为独立样本。','', '## 1. 校准覆盖','', '|编码器|B|cap|实际总量|容量|准入%|Novel%|条件Novel%|达标%|','|---|---|---|---|---|---|---|---|---|']
 for r in cov:
  lines.append('|'+ '|'.join([r['encoder'],str(r['B']),str(r['cap']),str(r['actual_total_range']),fmt(r['capacity']),fmt(r['finite']),fmt(r['novel']),fmt(r['conditional_novel']) if r['conditional_novel'] else '--',fmt(r['met'])])+'|')
 lines+=['','## 2. 等总量150的比较','','只有实际总量在全部重复中均为150的行进入下表；容量差异仍反映不同B任务，并非同模型干预。','', '|编码器|B|每类数量|准入%|Novel%|','|---|---|---|---|---|']
 for r in equal:
  lines.append(f"|{r['encoder']}|{r['B']}|{r['cap']}|{fmt(r['finite'])}|{fmt(r['novel'])}|")
 lines+=['','等总量比较固定为所有B共同具备预算的四个类别划分；对应划分编号写入summary.json。其余类别划分因B=5可用独立样本不足150而排除，未混用不同划分集合。']
 lines+=['','## 3. Food/SUN 固定指标错误诊断','','在Dense策略实际Base准确率下匹配两条纯识别前沿。按两方法基类赢家是否一致预分组；Contribution为该组对总体Novel差值的加权贡献（pp）。这是描述性分解，不是因果检验。','', '|编码器|数据|查询|组|占比%|纠正%|损坏%|组内增益pp|总体贡献pp|','|---|---|---|---|---|---|---|---|---|']
 for r in dom:lines.append('|'+ '|'.join([r['encoder'],r['dataset'],r['group'],r['stratum']]+[fmt(r[k]) for k in ['mass','correction','destruction','gain','contribution']])+'|')
 lines+=['','## 4. 路由成本','','同一CPU两线程、512个缓存查询，报告完整读出与额外门控前向；不含图像编码。标准化常数不计为可训练参数。完整选参为计时重放，不替换准确率模型。','']
 folder=H/'router_inference'
 if not (folder/'complete.json').exists():folder=H/'router_serial'
 if not (folder/'complete.json').exists():folder=H/'router'
 for f in sorted(folder.glob('l14.json'))+sorted(folder.glob('b32.json')):
  r=json.loads(f.read_text());lines.append(f"- {r['encoder']} Full：完整读出中位数 {r['full']['median_ms']:.3f} ms；额外门控参数0，路由器拟合0。")
  for v in r['rows']:lines.append(f"  - {v['kind']}：{v['trainable_router_parameters']}参数；完整读出 {v['complete_readout']['median_ms']:.3f} ms；预计算输入的门控 {v['gate_forward']['median_ms']:.4f} ms；单设置拟合中位数 {np.median(v['selected_setting_fit_seconds']):.3f}s；36模型完整选参 {v['full_selection_seconds']:.3f}s。")
 (H/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
 fig,axs=plt.subplots(2,2,figsize=(7.2,5),sharex=True)
 for i,enc in enumerate(['l14','b32']):
  for B in [5,10,15,30]:
   rr=[r for r in cov if r['encoder']==enc and r['B']==B and r['cap'] in [5,10,20,40]]
   for j,key in enumerate(['finite','novel']):axs[i,j].plot([r['cap'] for r in rr],[r[key]['mean'] for r in rr],marker='o',label=f'B={B}')
  axs[i,0].set_ylabel(enc.upper()+' Admission (%)');axs[i,1].set_ylabel('Novel accuracy (%)')
 for ax in axs.flat:ax.set_ylim(0,102);ax.grid(alpha=.2);ax.set_xticks([5,10,20,40]);ax.set_xlabel('Calibration cap per class')
 axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(H/'coverage.pdf',bbox_inches='tight');plt.close(fig)
 print('Summary ready:',len(cov),'coverage conditions,',len(dom),'domain strata')
if __name__=='__main__':summarize()
