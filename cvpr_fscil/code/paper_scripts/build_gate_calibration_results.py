"""Aggregate the four additional experiment blocks, with paired partition intervals."""
from pathlib import Path
import sys,json,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/experiments'))
import expansion_gate_calibration as ex
import fcs_retention_value as rv
from kbs_runtime import atomic_json,digest_file
ROOT=ex.ROOT;OUT=ex.OUT;H=Path(__file__).resolve().parent;DEST=H/'gate_calibration'
LABEL={'retained':'Retained','anchored_tp':'Dense','raw':'Raw','fixed_temperature':'Fixed','logistic':'Logistic gate','mlp':'MLP gate'}
DS={'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'};ENC={'l14':'L/14','b32':'B/32'}

def interval(v):
    v=np.array(v,float);rng=np.random.default_rng(28102026)
    if not len(v):return None
    if len(v)==1:return dict(mean=float(v[0]),lo=None,hi=None,n=1)
    draws=v[rng.integers(0,len(v),(20000,len(v)))].mean(1)
    return dict(mean=float(v.mean()),lo=float(np.quantile(draws,.025)),hi=float(np.quantile(draws,.975)),n=len(v))

def avg(v):
    v=np.array(v,float)
    return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.,n=len(v)) if len(v) else None

def match_summary():
    result=[]
    for enc in ex.ENCODERS:
        for ds in ex.DATASETS:
            allrows=[r for p in range(10) for r in json.loads((OUT/f'matched/{enc}/{ds}/p{p}.json').read_text())['records'] if r['session']==5]
            for target in ['dense_realized','retained_realized','floor95','floor99']:
                for method in ['dense_query','retained_none','retained_fixed','dense_none','dense_fixed']:
                    gain=[];over=[];targets=[]
                    for p in range(10):
                        vals=[];ovs=[];ts=[]
                        for order in [0,1]:
                            pick=lambda m:next(r for r in allrows if r['part']==p and r['order']==order and r['target_kind']==target and r['method']==m)
                            a=pick('retained_query');b=pick(method)
                            if a['frontier']['feasible'] and b['frontier']['feasible']:
                                vals.append(100*(a['frontier']['novel']-b['frontier']['novel']))
                                ovs.append(100*a['frontier']['overshoot']);ts.append(100*a['target'])
                        if len(vals)==2:gain.append(np.mean(vals));over.append(np.mean(ovs));targets.append(np.mean(ts))
                    result.append(dict(encoder=enc,dataset=ds,target_kind=target,comparator=method,
                        paired_gain=interval(gain),target=avg(targets),overshoot=avg(over)))
    return result

def gate_summary():
    result=[];contrasts=[];matched=[]
    for enc in ex.ENCODERS:
        for ds in ex.DATASETS:
            records=[r for p in range(10) for jp in (OUT/f'gates/{enc}/{ds}/p{p}').glob('o*_t*.json') for r in json.loads(jp.read_text())['records']]
            for method in ex.GATES:
                aia=[];cap=[];nov=[];base=[];finite=met=0;pg=[];cg=[];fg=[]
                for p in range(10):
                    rr=[r for r in records if r['part']==p and r['method']==method]
                    aia.append(100*np.mean([r['overall'] for r in rr if r['policy']=='fixed_gate']))
                    cs=[r for r in rr if r['policy']=='matched_test' and r['retention']==.95 and r['session']==5]
                    if all(r['feasible'] for r in cs):cap.append(100*np.mean([r['novel'] for r in cs]))
                    r=next(r for r in rr if r['policy']=='paired_accuracy' and r['session']==5 and r['order']==0)
                    nov.append(100*r['new']);base.append(100*r['old']);finite+=not r['fallback'];met+=r['test_target_met']
                    ppg=[];ccg=[];ffg=[]
                    for order in [0,1]:
                        q=next(r for r in json.loads((ROOT/f'experiments/kbs_reviewer_revision/fairness/{enc}/{ds}/p{p}/o{order}_t5.json').read_text())['records'] if r['method']=='retained' and r['policy']=='paired_accuracy')
                        g=next(r for r in rr if r['policy']=='paired_accuracy' and r['session']==5 and r['order']==order)
                        z=ex.loadz(OUT/f'gates/{enc}/{ds}/p{p}/o{order}_t5.npz')
                        legacy=ex.loadz(ROOT/f'experiments/kbs_independent_risk/{enc}/{ds}/p{p}/o{order}_t5.npz')
                        bc=z['base_classes'];nc=z['novel_classes'];y=z['test_labels'];ref=z['test_reference_correct']
                        qc=ex.cm(legacy,'retained',y,bc,nc,'test_');gc=ex.cm(z,method,y,bc,nc,'test_')
                        ppg.append(100*(q['new']-g['new']))
                        fq=ex.operating(qc,g['old']);ff=ex.operating(gc,g['old'])
                        if fq['feasible'] and ff['feasible']:ffg.append(100*(fq['novel']-ff['novel']))
                        closed=float(np.mean([ref[y==cl].mean() for cl in bc]));cq=ex.operating(qc,.95*closed);cc=ex.operating(gc,.95*closed)
                        if cq['feasible'] and cc['feasible']:ccg.append(100*(cq['novel']-cc['novel']))
                        matched.append(dict(encoder=enc,dataset=ds,part=p,order=order,method=method,target=g['old'],retained=fq,gate=ff))
                    pg.append(np.mean(ppg))
                    if len(ccg)==2:cg.append(np.mean(ccg))
                    if len(ffg)==2:fg.append(np.mean(ffg))
                result.append(dict(encoder=enc,dataset=ds,method=method,aia=avg(aia),capacity=avg(cap),novel=avg(nov),base=avg(base),finite=int(finite),met=int(met)))
                contrasts.append(dict(encoder=enc,dataset=ds,comparator=method,policy_gain=interval(pg),capacity_gain=interval(cg),realized_base_gain=interval(fg)))
    return result,contrasts,matched

def stability_summary():
    groups=collections.defaultdict(list)
    for p in (OUT/'stability').rglob('p*.json'):
        for r in json.loads(p.read_text())['records']:groups[r['encoder'],r['dataset'],r['cap'],r['method'],r['policy']].append(r)
    result=[]
    for (enc,ds,cap,m,pol),rows in groups.items():
        means=[];within=[];threshold=[]
        for part in range(10):
            rr=[r for r in rows if r['part']==part];v=[r['novel'] for r in rr];means.append(100*np.mean(v))
            if len(v)>1:within.append(100*np.std(v,ddof=1))
            off=[r['offset'] for r in rr if r['finite']]
            if len(off)>1:threshold.append(float(np.std(off,ddof=1)))
        admitted=[r for r in rows if r['finite']]
        result.append(dict(encoder=enc,dataset=ds,cap=cap,method=m,policy=pol,cells=len(rows),
            admission_fraction=float(np.mean([r['finite'] for r in rows])),met_fraction=float(np.mean([r['test_met'] for r in rows])),
            novel=avg(means),novel_ci=interval(means),conditional_novel=100*float(np.mean([r['novel'] for r in admitted])) if admitted else None,
            within_partition_subset_sd=float(np.mean(within)) if within else 0.,threshold_subset_sd=float(np.mean(threshold)) if threshold else None,
            calibration_labels=float(np.mean([r['labels'] for r in rows]))))
    return result

def real_coupling():
    result=[]
    for enc in ex.ENCODERS:
        for ds in ex.DATASETS:
            for m in ['retained_query','dense_query','retained_none','retained_fixed']:
                rows=[r for p in range(10) for r in json.loads((OUT/f'matched/{enc}/{ds}/p{p}.json').read_text())['records'] if r['method']==m and r['target_kind']=='floor95' and r['session']==5 and r['order']==0]
                actual=[];interventions=[]
                for p in range(10):
                    folder=ROOT/f'experiments/kbs_full_fusion_retuning/{enc}/{ds}/p{p}'
                    z=ex.loadz(folder/'o0_t5.npz');y=z['cal_labels'];bc=z['base_classes'];nc=z['novel_classes'];ref=z['cal_reference_correct']
                    policy=next(r for r in json.loads((folder/'o0_t5.json').read_text())['records'] if r['method']==m and r['policy']=='paired_accuracy')
                    c=ex.cm(z,m,y,bc,nc,'cal_');d=policy['offset']
                    stat=ex.accuracy_stats(c,ref,y,bc,d);actual.append(stat)
                    diagnostic_d=policy['trace'][-1]['offset'] if policy['trace'] else d
                    correct=ref.copy() if diagnostic_d is None else c['base_correct']&~ex.choose(c,diagnostic_d)
                    coupled=correct.copy()
                    for cl in bc:
                        g=np.flatnonzero((y==cl)&~ref&correct);h=np.flatnonzero((y==cl)&ref&~correct);n=min(len(g),len(h))
                        coupled[g[:n]]=False;coupled[h[:n]]=True
                        assert correct[y==cl].sum()==coupled[y==cl].sum()
                    w=ex.risk.weights(y,bc)
                    po,detail=ex.paired.pvalue(ref,correct,w);pn,newdetail=ex.paired.pvalue(ref,coupled,w)
                    assert pn<=po+1e-12
                    interventions.append(dict(part=p,admitted=d is not None,diagnostic_offset=diagnostic_d,
                        first_rejected_or_last_tested=True,original_pvalue=po,min_recovery_pvalue=pn,
                        original_recovery=detail['recovered'],min_recovery=newdetail['recovered'],
                        removed_recovery=detail['recovered']-newdetail['recovered'],
                        log_evalue_gain=newdetail['log_evalue']-detail['log_evalue'],
                        original_pass=po<=.01,min_recovery_pass=pn<=.01))
                result.append(dict(encoder=enc,dataset=ds,method=m,base=100*np.mean([r['selected_base'] for r in rows]),novel=100*np.mean([r['selected_novel'] for r in rows]),
                    recovered=100*np.mean([r['error_structure']['recovered'] for r in rows]),destroyed=100*np.mean([r['error_structure']['destroyed'] for r in rows]),
                    calibration_recovery=100*np.mean([r['recovered'] for r in actual]),
                    mean_variance=float(np.mean([r['calibration_mean_variance'] for r in actual])),
                    coupling_interventions=interventions))
    return result

def fmt(v):return '--' if v is None else f'{v:.2f}'
def ci(v):
    if v is None:return '--'
    if v['n']==1:return f"{v['mean']:+.2f} (1; no CI)"
    return f"{v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] ({v['n']})"
def table(name,spec,header,lines):
    text='\\begin{tabular}{@{}'+spec+'@{}}\n\\toprule\n'+header+r' \\'+ '\n\\midrule\n'+'\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n'
    (H/name).write_text(text,encoding='utf-8')

def tables(s):
    lines=[]
    for r in s['actual_base']:
        if r['comparator']!='dense_query':continue
        lines.append(f"{ENC[r['encoder']]} / {DS[r['dataset']]} & {r['target_kind'].replace('_',' ')} & {fmt(r['target']['mean'] if r['target'] else None)} & {ci(r['paired_gain'])} & {fmt(r['overshoot']['mean'] if r['overshoot'] else None)} "+r'\\')
    table('table_gate_actual_base.tex','llrrr','Condition & Target & Base (\\%) & Retained $-$ dense (pp) & Overshoot',lines)
    lines=[]
    for r in s['gates']:
        cap=r['capacity']
        lines.append(f"{ENC[r['encoder']]} / {DS[r['dataset']]} & {LABEL[r['method']]} & {r['aia']['mean']:.2f} & {fmt(cap['mean'] if cap else None)} ({cap['n'] if cap else 0}) & {r['novel']['mean']:.2f} & {r['finite']}/10 & {r['met']}/10 "+r'\\')
    table('table_gate_baselines.tex','llrrrrr','Condition & Gate & AIA & Capacity ($n$) & Novel & Finite & Met',lines)
    lines=[]
    for r in s['gate_contrasts']:
        lines.append(f"{ENC[r['encoder']]} / {DS[r['dataset']]} & {LABEL[r['comparator']]} & {ci(r['capacity_gain'])} & {ci(r['policy_gain'])} & {ci(r['realized_base_gain'])} "+r'\\')
    table('table_gate_contrasts.tex','llrrr','Condition & Comparator & Capacity & Direct policy & Actual-base frontier',lines)
    lines=[]
    for r in s['real_coupling']:
        lines.append(f"{ENC[r['encoder']]} / {DS[r['dataset']]} & {r['method'].replace('_',' ')} & {r['base']:.2f} & {r['novel']:.2f} & {r['destroyed']:.2f} & {r['recovered']:.2f} & {r['mean_variance']:.6f} "+r'\\')
    table('table_gate_error_structure.tex','llrrrrr','Condition & Readout & Base & Novel & $D$ & $G$ & Var. mean',lines)
    lines=[]
    for r in s['real_coupling']:
        rr=r['coupling_interventions']
        rescued=sum(not x['original_pass'] and x['min_recovery_pass'] for x in rr)
        lines.append(f"{ENC[r['encoder']]} / {DS[r['dataset']]} & {r['method'].replace('_',' ')} & {100*np.mean([x['removed_recovery'] for x in rr]):.2f} & {np.mean([x['log_evalue_gain'] for x in rr]):.2f} & {rescued}/10 "+r'\\')
    table('table_gate_coupling_intervention.tex','llrrr','Condition & Readout & Removed $G$ (pp) & $\\Delta\\log E$ & Rejected $\\to$ passed',lines)
    for pol,tag in [('paired_accuracy','direct'),('protection','protection')]:
        lines=[]
        for r in sorted(s['stability'],key=lambda r:(r['encoder'],r['dataset'],r['cap'])):
            if r['method']!='retained' or r['policy']!=pol:continue
            lines.append(f"{ENC[r['encoder']]} / {DS[r['dataset']]} & {r['cap']} & {100*r['admission_fraction']:.1f} & {100*r['met_fraction']:.1f} & {r['novel']['mean']:.2f} & {r['within_partition_subset_sd']:.2f} & {fmt(r['conditional_novel'])} "+r'\\')
        table(f'table_gate_stability_{tag}.tex','lrrrrrr','Condition & Cap & Finite (\\%) & Met (\\%) & Novel & Subset SD & Given finite',lines)

def figures(s):
    plt.rcParams.update({'font.family':'serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'retained':'#1b5579','anchored_tp':'#d65f00','logistic':'#b24e92','mlp':'#008a83'}
    for pol,tag in [('paired_accuracy','direct'),('protection','protection')]:
        fig,axs=plt.subplots(2,3,figsize=(10,5.5),constrained_layout=True)
        for i,enc in enumerate(ex.ENCODERS):
            for j,ds in enumerate(ex.DATASETS):
                ax=axs[i,j]
                for m,color in colors.items():
                    rows=sorted([r for r in s['stability'] if r['encoder']==enc and r['dataset']==ds and r['method']==m and r['policy']==pol],key=lambda r:r['cap'])
                    x=[r['cap'] for r in rows];v=np.array([r['novel']['mean'] for r in rows]);sd=np.array([r['within_partition_subset_sd'] for r in rows])
                    ax.plot(x,v,'o-',ms=3,lw=1,c=color,label=LABEL[m]);ax.fill_between(x,np.maximum(0,v-sd),np.minimum(100,v+sd),color=color,alpha=.1)
                ax.set_title(f'{ENC[enc]} {DS[ds]}');ax.set_xlabel('Images per base class');ax.set_ylim(0,100)
                if j==0:ax.set_ylabel('Novel accuracy (%)')
        handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
        fig.savefig(H/f'gate_budget_{tag}.pdf',bbox_inches='tight',pad_inches=.06);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(10,5.3),constrained_layout=True)
    for i,enc in enumerate(ex.ENCODERS):
        for j,ds in enumerate(ex.DATASETS):
            ax=axs[i,j]
            for cap in [5,10,20,40]:
                rows=[r for r in s['coupling_simulation'] if r['encoder']==enc and r['dataset']==ds and r['cap']==cap]
                ax.errorbar([r['recovered']*100 for r in rows],[r['acceptance']*100 for r in rows],
                    yerr=[1.96*r['mc_standard_error']*100 for r in rows],marker='o',ms=3,lw=.9,label=f'cap {cap}')
            ax.set_title(f'{ENC[enc]} {DS[ds]}');ax.set_xlabel('Recovery G (%)');ax.set_ylim(-3,103)
            if j==0:ax.set_ylabel('Fixed-candidate acceptance (%)')
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(H/'gate_coupling_power.pdf',bbox_inches='tight',pad_inches=.06);plt.close(fig)

def main():
    DEST.mkdir(exist_ok=True);verification=json.loads((OUT/'verification.json').read_text());assert verification['complete']
    gates,contrasts,matched=gate_summary()
    s=dict(protocol_sha256=digest_file(ex.PROTOCOL),verification=verification,
        feature_witness=json.loads((OUT/'feature_witness.json').read_text()),
        restart_verification=json.loads((OUT/'restart_verification.json').read_text()),
        actual_base=match_summary(),gates=gates,
        gate_contrasts=contrasts,gate_actual_base=matched,stability=stability_summary(),real_coupling=real_coupling(),
        coupling_simulation=[r for p in (OUT/'coupling').glob('*.json') for r in json.loads(p.read_text())['records']])
    atomic_json(s,DEST/'summary.json');tables(s);figures(s)
    report=['# Four supplementary experiment blocks','','## Actual-base frontier: retained minus dense']
    for r in s['actual_base']:
        if r['comparator']=='dense_query':report.append(f"- {r['encoder']} {r['dataset']} {r['target_kind']}: {ci(r['paired_gain'])}")
    report+=['','## Retained minus learned gate (paired across orders)']
    for r in contrasts:report.append(f"- {r['encoder']} {r['dataset']} {r['comparator']}: capacity {ci(r['capacity_gain'])}; policy {ci(r['policy_gain'])}; actual-base {ci(r['realized_base_gain'])}.")
    (DEST/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8');print('\n'.join(report))

if __name__=='__main__':main()
