"""Statistical completeness audit; no fitting or threshold reselection."""
from pathlib import Path
import sys,json,math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import expansion_prospective_validation as p
import kbs_calibration_budget as budget
from summarize_expansion_followup import stats,fmt
H=p.ROOT/'experiments/expansion_followup_diagnostics'
def main():
 summary=json.loads((H/'summary.json').read_text());groups=[];budget_effects=[];power=[]
 for enc in ['l14','b32']:
  for B in [5,10,15,30]:
   rows=[]
   for part in range(10):rows+=json.loads((H/f'coverage/{enc}/B{B}/p{part}.json').read_text())['records']
   for lo,hi in [(10,20),(20,40),(10,40)]:
    changes={k:[] for k in ['finite','novel']}
    for part in range(10):
     a={(r['order'],r['subset']):r for r in rows if r['part']==part and r['cap']==lo};b={(r['order'],r['subset']):r for r in rows if r['part']==part and r['cap']==hi};assert a.keys()==b.keys()
     for k in changes:
      value=lambda r:float(not r['policy']['fallback']) if k=='finite' else r['policy']['new']
      changes[k].append(100*np.mean([value(b[t])-value(a[t]) for t in a]))
    budget_effects.append(dict(encoder=enc,B=B,low_cap=lo,high_cap=hi,**{k:stats(v) for k,v in changes.items()}))
   if B==5:
    for part in range(10):
     for order in [0,1]:
      z=p.gates.loadz(p.OUT/f'cardinality/{enc}/imagenetr/B5/p{part}/o{order}_t5.npz');y=z['cal_labels'];base=z['base_classes'];R=z['cal_reference_correct'];c=budget.compact(z,'C30',y,base,z['novel_classes'],'cal_')
      assert np.array_equal(c['base_correct'],R),'Zero-recovery ceiling requires same closed-base predictions'
      for seed in range(10):
       rng=np.random.default_rng(41000+seed+100*part);ix={cl:rng.permutation(np.flatnonzero(y==cl)) for cl in base}
       for cap in [10,20,40]:
        ids=np.sort(np.concatenate([v[:cap] for v in ix.values()]));w=p.risk.weights(y[ids],base);pv,_=p.paired.pvalue(R[ids],R[ids],w)
        power.append(dict(encoder=enc,part=part,order=order,subset=seed,cap=cap,pvalue=pv,closed_reference_passes=pv<=.01))
  for ds in ['food','sun']:
   for group in ['base','novel']:
    for stratum in ['all','base_winners_agree','base_winners_disagree']:
     counts=[];class_counts=[]
     for part in range(10):
      for order in [0,1]:
       z=p.gates.loadz(p.OUT/f'external/{enc}/{ds}/catalog/p{part}/o{order}_t5.npz');y=z['test_labels'];agree=z['test_full_base_prediction']==z['test_dense_base_prediction'];m=np.isin(y,z['base_classes' if group=='base' else 'novel_classes'])
       if stratum!='all':m&=agree if stratum=='base_winners_agree' else ~agree
       counts.append(int(m.sum()));class_counts.append(int(len(np.unique(y[m]))))
     rec=next(r for r in summary['domain'] if (r['encoder'],r['dataset'],r['group'],r['stratum'])==(enc,ds,group,stratum))
     groups.append(dict(**rec,partition_order_cells=20,nonempty_cells=sum(v>0 for v in counts),query_count_range=[min(counts),max(counts)],class_count_range=[min(class_counts),max(class_counts)],sum_query_appearances=sum(counts),independent_resampling_units=rec['gain']['n']))
 timing=[]
 for enc in ['l14','b32']:
  r=json.loads((H/f'router_inference/{enc}.json').read_text());timing.append(dict(encoder=enc,method='Full',**r['full']))
  for row in r['rows']:timing.append(dict(encoder=enc,method=row['kind'],**row['complete_readout']))
 result=dict(complete=True,budget_effects=budget_effects,groups=groups,timing=timing,power_ceiling=power,scope='Paired bootstrap of ten partition means. Query counts describe appearances, not independent observations. Timing P10-P90 is empirical run variation, not a confidence interval. Full-search timing is one replay, selected-setting fits have three repeats.')
 (H/'statistical_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# 统计完整性核验','', '预算差值及分组识别差值均使用20,000次配对bootstrap，先在每个类别划分内平均两个顺序和校准子集，再重采样有效类别划分（最多十个；空组处理见下文）。','', '## 校准预算增加的配对效果','', '|编码器|B|cap变化|准入率变化pp [95%CI]|Novel变化pp [95%CI]|','|---|---|---|---|---|']
 for r in budget_effects:lines.append(f"|{r['encoder']}|{r['B']}|{r['low_cap']}→{r['high_cap']}|{fmt(r['finite'])}|{fmt(r['novel'])}|")
 lines+=['','B=5时cap10/20下，两个编码器的所有200个最终划分/顺序/子集记录中，C=R的最好保留耦合均不能通过检验。由于检查确认该Full模型的闭集正确性与共同参考一致，任何准入只能使C≤R；因此此处的检验功效不足不是由新类识别容量低推断而来。cap40可改善部分准入，但配对区间包含零，不能声称B=5存在稳定显著的预算收益，也不能把剩余差距全部归于样本量。','', '## 分组样本量及配对区间','', '查询数和类别数为每个划分/顺序单元的范围，重采样独立单位始终为10个类别划分；同一图片跨顺序或划分的重复出现不增加有效样本数。','', '|编码器|数据|组|分层|查询数范围|类别数范围|有效划分n|Full−Dense pp [95%CI]|','|---|---|---|---|---|---|---|---|']
 for r in groups:lines.append(f"|{r['encoder']}|{r['dataset']}|{r['group']}|{r['stratum']}|{r['query_count_range']}|{r['class_count_range']}|{r['gain']['n']}|{fmt(r['gain'])}|")
 lines+=['','Food两个编码器各有两个划分的“赢家不同”组为空。条件准确率与条件增益排除这些空组，使用8个非空划分；占比及总体加权贡献仍使用全部10个划分，空组贡献为0。不能把空组条件准确率写为0。机器可读结果逐行提供非空单元数和bootstrap n。','', '## Router完整读出时间波动','', '同一CPU两线程，512个缓存查询，5次预热、30次计时。范围为P10–P90经验分位数，不是95%置信区间；不含图像编码、数据加载或标签相关的评价指标构建。','', '|编码器|读出|中位数ms|P10–P90 ms|','|---|---|---|---|']
 for r in timing:lines.append(f"|{r['encoder']}|{r['method']}|{r['median_ms']:.3f}|{r['p10_ms']:.3f}–{r['p90_ms']:.3f}|")
 lines+=['','完整36模型选参耗时仅测了一次，不能赋予重复测量区间；单设置拟合有三次重复。计时比较不是随机交错的严格配对性能检验，不据此声称某个很小的时差具有统计显著性。']
 (H/'STATISTICAL_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('Statistical audit complete')
if __name__=='__main__':main()
