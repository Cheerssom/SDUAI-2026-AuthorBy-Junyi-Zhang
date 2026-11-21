"""Collect complete C10/C50 extension alongside immutable C20/C30 controls."""
from pathlib import Path
import json,zipfile,numpy as np,sys
R=Path(__file__).resolve().parents[2];H=R/'experiments/expansion_constant_sensitivity_remote';D=H/'download'
def stat(v):
 a=np.asarray(v);rng=np.random.default_rng(2026);b=a[rng.integers(len(a),size=(20000,len(a)))].mean(1)
 return dict(mean=float(a.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),n=len(a),values=a.tolist())
def main():
 global H,D
 cloud='--cloud' in sys.argv
 if cloud:H=R;D=R
 else:
  with zipfile.ZipFile(H/'results_bundle.zip') as z:assert z.testzip() is None;z.extractall(D)
 roots={10:D/'results/cardinality',50:D/'results/cardinality',20:(R/'controls' if cloud else R/'experiments/expansion_prospective_validation/cardinality'),30:(R/'controls' if cloud else R/'experiments/expansion_prospective_validation/cardinality')}
 rows=[];contrasts=[];witness=0
 for enc in ['l14','b32']:
  for B in [5,10,15,20,30,50,100]:
   cache={}
   for C in [10,20,30,50]:
    partcap=[];partnov=[];f=met=0;closed=[]
    for part in range(10):
     caps=[];nov=[]
     for order in [0,1]:
      jp=roots[C]/enc/'imagenetr'/f'B{B}'/f'p{part}'/f'o{order}_t5.json';data=json.loads(jp.read_text());r=next(v for v in data['records'] if v['method']==f'C{C}');cap=r['capacity95'];pol=r['policy']
      assert cap['feasible'];caps.append(100*cap['new']);nov.append(100*pol['new']);f+=int(not pol['fallback']);met+=int(pol['test_target_met']);closed.append(100*data['reference_closed'])
     partcap.append(float(np.mean(caps)));partnov.append(float(np.mean(nov)))
    cache[C]=dict(capacity=partcap,novel=partnov)
    rows.append(dict(encoder=enc,B=B,C=C,capacity=stat(partcap),novel=stat(partnov),finite=f,met=met,closed_mean=float(np.mean(closed))))
   for C in [10,50]:contrasts.append(dict(encoder=enc,B=B,C=C,capacity_gain=stat(np.array(cache[C]['capacity'])-cache[30]['capacity']),policy_gain=stat(np.array(cache[C]['novel'])-cache[30]['novel'])))
  # B100 both pseudo bank>=50: controls algebraically coincide with C30.
  for part in range(10):
   old=(R/f'inputs/{enc}/tasks/B100/p{part}/selection.json' if cloud else R/f'experiments/expansion_prospective_validation/cardinality/{enc}/imagenetr/B100/p{part}/selection.json');new=D/f'results/cardinality/{enc}/imagenetr/B100/p{part}/selection.json';a=json.loads(old.read_text())['selections']['C30']['selected'];b=json.loads(new.read_text())['selections']
   for C in [10,50]:
    assert a['a']==b[f'C{C}']['selected']['a'] and a['u']==b[f'C{C}']['selected']['u'];assert abs(a['offset']-b[f'C{C}']['selected']['offset'])<2e-4;witness+=1
 report=dict(complete=True,conditions=len(rows),new_constants=[10,50],main_constant=30,verified_B100_equivalent_selections=witness,rows=rows,contrasts=contrasts)
 (H/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# 常数敏感性扩展：C=10/20/30/50','', '固定主方法C30，新增C10和C50独立3636设置选参。C20/C30直接读取原冻结结果，不重跑、不改参数。真实B=5/10/15/20/30/50/100的Ridge权重、数据划分与C30共同保底参照均保持一致。下表为第五轮、两顺序先在十划分内平均；有限准入/达标均计20条最终路径。','', '|Encoder|B|C|Capacity|Calibrated Novel|Finite/Met|','|---|---:|---:|---:|---:|---|']
 for r in rows:lines.append(f"|{r['encoder']}|{r['B']}|{r['C']}|{r['capacity']['mean']:.2f}|{r['novel']['mean']:.2f}|{r['finite']}/20 / {r['met']}/20|")
 lines+=['','全部配对置信区间、划分值与相对C30差值见summary.json。此实验只分析预先规定常数的敏感性，不按测试表现更换主C30。B100时两种新增规则在pseudo/full阶段均为eta=1，与C30的选参一致性作为数值核验。']
 (H/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');plot(rows,H);print('COMPLETE',len(rows),'B100 witnesses',witness)
def plot(rows,H):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
 for i,en in enumerate(['l14','b32']):
  for j,key in enumerate(['capacity','novel']):
   ax=axes[i,j]
   for C,color in zip([10,20,30,50],['#d95f02','#7570b3','#1b9e77','#e7298a']):
    q=sorted([r for r in rows if r['encoder']==en and r['C']==C],key=lambda r:r['B']);ax.plot([r['B'] for r in q],[r[key]['mean'] for r in q],marker='o',label=f'C={C}',color=color,linewidth=2 if C==30 else 1)
   ax.set_title(f"{en.upper()} / "+('Oracle capacity' if key=='capacity' else 'Calibrated novel'))
   ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2);ax.set_xscale('log');ax.set_xticks([5,10,15,20,30,50,100]);ax.set_xticklabels([5,10,15,20,30,50,100]);ax.set_xlabel('Base class count B')
 axes[0,0].legend(ncol=2);fig.tight_layout();fig.savefig(H/'constant_sensitivity.pdf',bbox_inches='tight');fig.savefig(H/'constant_sensitivity.png',dpi=160,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
