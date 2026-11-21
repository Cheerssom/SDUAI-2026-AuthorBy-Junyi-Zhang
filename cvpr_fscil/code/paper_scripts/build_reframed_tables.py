from pathlib import Path
import json,shutil
H=Path(__file__).resolve().parent;R=H.parents[1];E=H/'revision_evidence';E.mkdir(exist_ok=True)
files={'prospective':H/'prospective/summary.json','sun_cache':R/'experiments/expansion_sun_controls_remote/summary.json','shared':R/'experiments/expansion_shared_head_remote/summary.json','cub':R/'experiments/expansion_cub_remote/summary.json','constants':R/'experiments/expansion_constant_sensitivity_remote/summary.json','references':R/'experiments/expansion_reference_completion/summary.json','cache_references':R/'experiments/expansion_reference_completion/cache_summary.json','source_references':R/'experiments/expansion_self_reference_remote/summary.json','hycal_references':R/'experiments/expansion_hycal_self_remote/summary.json'}
D={k:json.loads(p.read_text()) for k,p in files.items()}
for k,p in files.items():shutil.copy2(p,E/(k+'.json'))
shutil.copy2(R/'experiments/expansion_constant_sensitivity_remote/download/constant_sensitivity.pdf',H/'constant_sensitivity_new.pdf')
labels={'full':'Full','none':'None','fixed':'Fixed','dense':'Dense + stacking','cache_mean':'Class-mean cache','SimpleCIL_transfer':'SimpleCIL transfer','RanPAC_frozen_transfer':'RanPAC transfer','BiMC_transfer':'BiMC transfer','HyCal_transfer':'HyCal transfer','retained':'Retained source','anchored_tp':'Dense source','hycal':'HyCal source'}
lines=[r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',r'& & \multicolumn{3}{c}{CLIP L/14} & \multicolumn{3}{c}{CLIP B/32} \\',r'\cmidrule(lr){3-5}\cmidrule(l){6-8}',r'Data & Readout & AIA & Base & Novel (finite) & AIA & Base & Novel (finite) \\',r'\midrule'];witness=[]
for ds,methods in [('sun',['full','dense','cache_mean']),('food',['full','dense']),('cub',['full','none','fixed','dense','SimpleCIL_transfer','RanPAC_frozen_transfer','BiMC_transfer','HyCal_transfer'])]:
 for j,m in enumerate(methods):
  row=[{'sun':'SUN397','food':'Food-101','cub':'CUB-200'}[ds] if j==0 else '',labels[m]]
  for en in ['l14','b32']:
   if ds=='cub':
    a=next(r for r in D['cub']['summary'] if r['encoder']==en and r['track']=='standard' and r['method']==m);b=next(r for r in D['cub']['summary'] if r['encoder']==en and r['track']=='certified' and r['method']==m)['common'];vals=[a['aia_top1'],b['base'],b['novel']];count=f"{b['finite']}/3";assert b['met']==3
   elif m=='cache_mean':
    a=next(r for r in D['sun_cache']['summary'] if r['encoder']==en and r['method']==m);vals=[a['aia']['mean'],a['base']['mean'],a['novel']['mean']];count=f"{a['finite']}/20";assert a['met']==20
   else:
    a=next(r for r in D['prospective']['rows'] if r['block']=='external' and r['dataset']==ds and r['encoder']==en and r['method']==m);vals=[a['aia']['mean'],a['base']['mean'],a['novel']['mean']];count=f"{a['finite']}/20";assert a['met']==20
   row += [f'{vals[0]:.2f}',f'{vals[1]:.2f}',f'{vals[2]:.2f} ({count})'];witness.append(dict(dataset=ds,method=m,encoder=en,values=vals,finite=count))
  lines.append(' & '.join(row)+r' \\')
 lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}');(H/'table_cvpr_primary.tex').write_text('\n'.join(lines)+'\n');(E/'primary_display.json').write_text(json.dumps(witness,indent=2)+'\n')
lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'& & \multicolumn{2}{c}{CLIP L/14} & \multicolumn{2}{c}{CLIP B/32} \\',r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',r'Data & Readout & Common & Self & Common & Self \\',r'\midrule'];witness=[]
for ds,methods in [('sun',['full','dense','cache_mean']),('sketch',['retained','anchored_tp','hycal']),('cub',['full','dense','SimpleCIL_transfer','RanPAC_frozen_transfer','BiMC_transfer','HyCal_transfer'])]:
 for j,m in enumerate(methods):
  row=[{'sun':'SUN397','sketch':'Sketch','cub':'CUB-200'}[ds] if j==0 else '',labels[m]]
  for en in ['l14','b32']:
   for kind in ['common','self']:
    if ds=='cub':a=next(r for r in D['cub']['summary'] if r['encoder']==en and r['track']=='certified' and r['method']==m)[kind];val=a['novel'];floor=.95*a['closed'];c=f"{a['finite']}/3"
    else:
     source=D['cache_references']['summary'] if m=='cache_mean' else D['references']['summary'];a=next(r for r in source if r['dataset']==ds and r['encoder']==en and r['method']==m and r['reference']==kind and (ds!='sketch' or r.get('calibration')=='source_cal'));val=a['novel'];floor=.95*a['closed_base'];c=f"{a['finite']}/{a['total']}"
    row.append(r'\shortstack{'+f'{val:.2f} ({c})'+r'\\'+r'\scriptsize $\tau='+f'{floor:.2f}'+r'$}');witness.append(dict(dataset=ds,method=m,encoder=en,reference=kind,novel=val,floor=floor,finite=c))
  lines.append(' & '.join(row)+r' \\')
 lines.append(r'\midrule')
lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}');(H/'table_cvpr_references.tex').write_text('\n'.join(lines)+'\n');(E/'reference_display.json').write_text(json.dumps(witness,indent=2)+'\n')
(H/'table_cvpr_tnorm_compact.tex').write_text(r'''\begin{tabular}{@{}lrr@{}}
\toprule
Contrast (pp) & L/14 & B/32 \\
\midrule
Full $-$ T-norm: capacity & +4.89 & +3.43 \\
\bottomrule
\end{tabular}
''')
print('primary and references:',len(D['cub']['summary']),len(witness))
