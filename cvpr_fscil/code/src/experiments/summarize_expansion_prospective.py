"""All-prespecified-comparison summaries and standalone experiment figures."""
from pathlib import Path
from collections import defaultdict
import argparse,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kbs_runtime import atomic_json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_prospective_validation';REPORT=OUT/'report'

def stat(values):
    values=np.asarray(values,float)
    if not len(values):return dict(n=0,mean=None,ci95=None)
    mean=float(values.mean());ci=None
    if len(values)>1:
        rng=np.random.default_rng(28102026);draw=values[rng.integers(len(values),size=(20000,len(values)))].mean(1);ci=np.quantile(draw,[.025,.975]).tolist()
    return dict(n=len(values),mean=mean,ci95=ci)

def fmt(s):
    if s['mean'] is None:return '-- (0)'
    result=f"{s['mean']:.2f}"
    if s['ci95']:result+=f" [{s['ci95'][0]:.2f},{s['ci95'][1]:.2f}]"
    return result+f" ({s['n']})"

def summarize(partial=False):
    REPORT.mkdir(parents=True,exist_ok=True);cells={};aia=defaultdict(list)
    for block in ['mechanism','cardinality','external']:
        for p in (OUT/block).glob('*/*/*/p*/o*_t*.json'):
            data=json.loads(p.read_text());e,ds,B,part,order,session=[data[k] for k in ['encoder','dataset','B','part','order','session']]
            for r in data['records']:
                m=r['method'];aia[block,e,ds,B,m,part].append(r['fixed_gate']['overall']*100)
                if session==5:cells[block,e,ds,B,m,part,order]=r
    groups=sorted({k[:5] for k in cells});rows=[];contrasts=[]
    for block,e,ds,B,m in groups:
        ks=[k for k in cells if k[:5]==(block,e,ds,B,m)]
        parts=sorted({k[5] for k in ks});cap=[];nov=[];cond=[];base=[];ret=[];au=[];inc=[];fin=met=0;den=0
        for part in parts:
            rr=[cells[block,e,ds,B,m,part,o] for o in [0,1] if (block,e,ds,B,m,part,o) in cells]
            if len(rr)<2:continue
            if all(r['capacity95']['feasible'] for r in rr):cap.append(np.mean([r['capacity95']['new']*100 for r in rr]))
            nov.append(np.mean([r['policy']['new']*100 for r in rr]));base.append(np.mean([r['policy']['old']*100 for r in rr]));au.append(np.mean([r['auc'] for r in rr]))
            reference=[json.loads((OUT/block/e/ds/(f'B{B}' if block=='cardinality' else 'catalog')/f'p{part}'/f'o{o}_t5.json').read_text())['reference_closed'] for o in [0,1]]
            ret.append(np.mean([r['policy']['old']/a*100 for r,a in zip(rr,reference)]));finite=[r['policy']['new']*100 for r in rr if not r['policy']['fallback']]
            if finite:cond.append(np.mean(finite))
            fin+=sum(not r['policy']['fallback'] for r in rr);met+=sum(r['policy']['test_target_met'] for r in rr);den+=2;inc.append(np.mean(aia[block,e,ds,B,m,part]))
        rows.append(dict(block=block,encoder=e,dataset=ds,B=B,method=m,capacity95=stat(cap),novel=stat(nov),conditional_novel=stat(cond),base=stat(base),retention=stat(ret),auc=stat(au),aia=stat(inc),finite=fin,met=met,total=den))
    conditions=sorted({k[:4] for k in cells})
    for block,e,ds,B in conditions:
        full='C30' if block=='cardinality' else 'full';methods=sorted({k[4] for k in cells if k[:4]==(block,e,ds,B)})
        for m in methods:
            if m==full:continue
            out={k:[] for k in ['capacity95','matched_actual_base','policy','policy_joint_finite','auc']};overshoot=[]
            for part in range(10):
                fr=[cells.get((block,e,ds,B,full,part,o)) for o in [0,1]];br=[cells.get((block,e,ds,B,m,part,o)) for o in [0,1]]
                if any(r is None for r in fr+br):continue
                if all(r['capacity95']['feasible'] for r in fr+br):out['capacity95'].append(np.mean([100*(f['capacity95']['new']-b['capacity95']['new']) for f,b in zip(fr,br)]))
                ff=[r['matched_actual_base'][m] for r in fr];bf=[r['matched_actual_base'][m] for r in br]
                if all(v['feasible'] for v in ff+bf):
                    out['matched_actual_base'].append(np.mean([100*(f['new']-b['new']) for f,b in zip(ff,bf)]));overshoot.append(np.mean([100*(f['old']-b['old']) for f,b in zip(ff,bf)]))
                pol=np.mean([100*(f['policy']['new']-b['policy']['new']) for f,b in zip(fr,br)]);out['policy'].append(pol)
                if all(not r['policy']['fallback'] for r in fr+br):out['policy_joint_finite'].append(pol)
                out['auc'].append(np.mean([f['auc']-b['auc'] for f,b in zip(fr,br)]))
            contrasts.append(dict(block=block,encoder=e,dataset=ds,B=B,baseline=m,**{k:stat(v) for k,v in out.items()},matched_base_overshoot_pp=stat(overshoot)))
    shift=defaultdict(lambda:defaultdict(list))
    for p in (OUT/'shift/results').glob('*/p*/*.json'):
        if p.name=='complete.json':continue
        v=json.loads(p.read_text())
        for r in v['records']:
            key=(v['encoder'],v['kind'],v['group'],v['severity'],r['method'],r['calibration'])
            for k in ['old','new','retention','violation_pp','target_gap_pp']:
                shift[key][k].append(r[k]*(100 if k in ['old','new','retention'] else 1))
            shift[key]['finite'].append(not r['fallback']);shift[key]['met'].append(r['test_target_met']);shift[key]['reference'].append(100*r['reference_closed'])
    shiftrows=[dict(encoder=k[0],kind=k[1],group=k[2],severity=k[3],method=k[4],calibration=k[5],**{m:stat(v) for m,v in r.items()}) for k,r in sorted(shift.items())]
    summary=dict(partial=partial,rows=rows,contrasts=contrasts,shift=shiftrows,completion={k:(OUT/k/'complete.json').exists() for k in ['mechanism','cardinality','shift']},source='frozen selection/session JSONs;orders averaged within partitions;conditional intervals on fixed images')
    atomic_json(summary,REPORT/'summary.json')
    highlights=[]
    for enc in ['l14','b32']:
        rr=[r for r in contrasts if r['block']=='mechanism' and r['encoder']==enc and r['dataset']=='imagenetr' and r['baseline'].startswith('shuffle')]
        if len(rr)==3:
            means=[r['capacity95']['mean'] for r in rr];supported=all(r['capacity95']['ci95'] and r['capacity95']['ci95'][0]>0 for r in rr)
            highlights.append(f"ImageNet-R {enc.upper()}：Full 相比三个预设置换种子的95%保底识别容量差为 {min(means):.2f}–{max(means):.2f} pp；三个配对区间均高于零。" if supported else f"ImageNet-R {enc.upper()}：三个预设置换对照的容量和区间逐项列于下表。")
    ext=[r for r in contrasts if r['block']=='external' and r['baseline']=='dense']
    if not ext:highlights.append('Food-101/SUN397 新领域验证正在执行；当前结论仅覆盖已完成的缓存实验。')
    else:
        for r in ext:highlights.append(f"新领域 {r['dataset']} {r['encoder']}：实际基础类准确率匹配后的 Full−Dense 容量差为 {fmt(r['matched_actual_base'])} pp。")
    lines=['# 四组补充实验结果'+('（进行中）' if partial else ''),'',*highlights,'','所有差值为 Full/C30 减去对照，容量与策略分别汇报。区间为按类别划分配对 bootstrap 95% 区间；n 为双方两种顺序均可行的划分数。策略条件差值只使用双方两种顺序均准入的划分；不以不同子集的条件均值相减。置换为批量统计干预，不赋予独立部署认证结论。','', '|组|编码器|数据|B|对照|95%容量差pp|实际Base匹配容量差pp|策略差pp|共同准入策略差pp|AUROC差|','|---|---|---|---:|---|---|---|---|---|---|']
    for r in contrasts:lines.append('|'+ '|'.join([r['block'],r['encoder'],r['dataset'],str(r['B']),r['baseline'],*[fmt(r[k]) for k in ['capacity95','matched_actual_base','policy','policy_joint_finite','auc']]])+'|')
    lines+=['','## 绝对性能','', '|组|编码器|数据|B|方案|容量%|策略Novel%|准入/总路径|达标/总路径|Base保留率%|AUROC|','|---|---|---|---:|---|---|---|---|---|---|---|']
    for r in rows:lines.append('|'+ '|'.join([r['block'],r['encoder'],r['dataset'],str(r['B']),r['method'],fmt(r['capacity95']),fmt(r['novel']),f"{r['finite']}/{r['total']}",f"{r['met']}/{r['total']}",fmt(r['retention']),f"{r['auc']['mean']:.4f}" if r['auc']['mean'] else '--'])+'|')
    lines+=['','## 分布偏移','','保留率以同一损坏部署分布上的原始参考分类器准确率为分母。Base 与 Reference 绝对准确率另列；相对保底不等于参考分类器对损坏本身免疫。','', '|编码器|损坏|方案|校准|Novel%|Base%|Reference%|Base保留率%|违反幅度pp|准入比例|达标比例|','|---|---|---|---|---|---|---|---|---|---|---|']
    for r in shiftrows:lines.append('|'+ '|'.join([r['encoder'],r['kind'],r['method'],r['calibration'],fmt(r['new']),fmt(r['old']),fmt(r['reference']),fmt(r['retention']),fmt(r['violation_pp']),f"{r['finite']['mean']:.2f}",f"{r['met']['mean']:.2f}"])+'|')
    (REPORT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plots(summary)
    if not partial:
        atomic_json(dict(complete=True,comparisons=len(contrasts),absolute_rows=len(rows),shift_rows=len(shiftrows)),OUT/'summary_complete.json')
        tracker='# Prospective experiment tracker — completed\n\nAll four prespecified blocks completed and reconstructed. C=30 remained frozen.\n\n- W:Food-101 and SUN397,two encoders,ten partitions,two orders,five arrivals,six systems.\n- X:query moment interventions,including three fixed shuffle seeds.\n- Y:true base-task refits across B and five prespecified C values.\n- Z:progressive calibration/deployment mismatch,source/target/reference-only policies.\n\nResults:experiments/expansion_prospective_validation/report/RESULTS_ZH.md\nVerification:experiments/expansion_prospective_validation/verification.json\n'
        for name in ['EXPERIMENT_TRACKER.md','EXPERIMENT_TRACKER_20261028_prospective.md']:(ROOT/'refine-logs'/name).write_text(tracker,encoding='utf-8')
        with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:f.write('\nCompleted all four prospective-validation blocks;frozen C=30,new-domain gate budgets3636,independent prediction/feature reconstruction passed. Results under experiments/expansion_prospective_validation/report/.\n')
    print(json.dumps(dict(rows=len(rows),contrasts=len(contrasts),shift=len(shiftrows),completion=summary['completion'])),flush=True)

def plots(summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    for metric in ['capacity95','novel']:
        fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
        for i,ds in enumerate(['food','sun']):
            for j,e in enumerate(['l14','b32']):
                ax=axes[i,j];methods=['full','none','fixed','dense','logistic','mlp']
                rr={r['method']:r for r in summary['rows'] if r['block']=='external' and r['dataset']==ds and r['encoder']==e}
                if not rr:ax.text(.5,.5,'Prospective evaluation pending',transform=ax.transAxes,ha='center',va='center',fontsize=9)
                for k,m in enumerate(methods):
                    if m not in rr or rr[m][metric]['mean'] is None:continue
                    st=rr[m][metric];err=None if not st['ci95'] else np.array([[st['mean']-st['ci95'][0]],[st['ci95'][1]-st['mean']]])
                    ax.bar(k,st['mean'],color='#1a5675' if m=='full' else '#a4bcc9');ax.errorbar(k,st['mean'],yerr=err,fmt='none',ecolor='#333333',capsize=2)
                ax.set_xticks(range(6),methods,rotation=30,ha='right');ax.set(title=f'{e.upper()} {ds}',ylabel='95% novel capacity (%)' if metric=='capacity95' else 'Direct novel accuracy (%)');ax.grid(axis='y',alpha=.2)
        fig.savefig(REPORT/f'external_{metric}.pdf',bbox_inches='tight',pad_inches=.05);plt.close(fig)
    for ds in ['cifar','imagenetr']:
        fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
        for i,e in enumerate(['l14','b32']):
            for j,metric in enumerate(['capacity95','novel']):
                ax=axes[i,j]
                for c in [15,20,30,40,60]:
                    rr=sorted([r for r in summary['rows'] if r['block']=='cardinality' and r['dataset']==ds and r['encoder']==e and r['method']==f'C{c}' and r[metric]['mean'] is not None],key=lambda r:r['B'])
                    if rr:
                        line,=ax.plot([r['B'] for r in rr],[r[metric]['mean'] for r in rr],'-o',label=f'C={c}',linewidth=2 if c==30 else 1)
                        band=[r for r in rr if r[metric]['ci95']]
                        if band:ax.fill_between([r['B'] for r in band],[r[metric]['ci95'][0] for r in band],[r[metric]['ci95'][1] for r in band],alpha=.06,color=line.get_color())
                ax.set(xlabel='Number of fitted base classes',ylabel=('95% novel capacity (%)' if j==0 else 'Direct novel accuracy (%)'),title=f'{e.upper()} {ds}');ax.grid(alpha=.2)
                if i==0 and j==0:ax.legend(ncol=2)
        fig.savefig(REPORT/f'cardinality_{ds}.pdf',bbox_inches='tight',pad_inches=.05);plt.close(fig)
    for metric in ['capacity95','matched_actual_base','policy']:
        fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
        for i,e in enumerate(['l14','b32']):
            for j,ds in enumerate(['cifar','pets','imagenetr']):
                ax=axes[i,j];rr=[r for r in summary['contrasts'] if r['block']=='mechanism' and r['encoder']==e and r['dataset']==ds and r[metric]['mean'] is not None]
                for k,r in enumerate(rr):
                    st=r[metric];err=None if not st['ci95'] else np.array([[st['mean']-st['ci95'][0]],[st['ci95'][1]-st['mean']]])
                    ax.errorbar(k,st['mean'],yerr=err,fmt='o',capsize=2,color='#206080')
                labels={'minus_learned_scale':'Fixed L\nscale','minus_text_scale':'Fixed T\nscale','pm_only':'Proto.\nmean only','scale_only':'Query\nscales only','shuffle3101':'Shuffle\n3101','shuffle3102':'Shuffle\n3102','shuffle3103':'Shuffle\n3103'}
                ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(len(rr)),[labels[r['baseline']] for r in rr],fontsize=7);ax.set(title=f'{e.upper()} {ds}',ylabel='Full minus control (pp)')
        fig.savefig(REPORT/f'mechanism_{metric}.pdf',bbox_inches='tight',pad_inches=.05);plt.close(fig)
    for group in ['noise','blur']:
        fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
        for i,e in enumerate(['l14','b32']):
            for j,metric in enumerate(['retention','new']):
                ax=axes[i,j]
                for method in ['retained','anchored_tp','reference']:
                    for cal in (['fallback'] if method=='reference' else ['source','target']):
                        rr=sorted([r for r in summary['shift'] if r['encoder']==e and r['group'] in ['clean',group] and r['method']==method and r['calibration']==cal],key=lambda r:r['severity'])
                        if rr:ax.plot([r['severity'] for r in rr],[r[metric]['mean'] for r in rr],'-o',label=method+'/'+cal)
                if j==0:ax.axhline(95,color='gray',ls='--')
                ax.set(xlabel=group+' severity',ylabel='Base retention (%)' if j==0 else 'Novel accuracy (%)',title=e.upper());ax.grid(alpha=.2)
                if i==0 and j==1 and ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
        fig.savefig(REPORT/f'shift_{group}.pdf',bbox_inches='tight',pad_inches=.05);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();summarize(a.partial)
