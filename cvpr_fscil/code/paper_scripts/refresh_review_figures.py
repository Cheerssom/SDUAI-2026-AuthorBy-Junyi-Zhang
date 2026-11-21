"""Refresh legend terminology and plot frozen L/14 source trajectories."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
source=ROOT/'experiments/fcs/results.json';D=json.loads(source.read_text());R=D['records']
LABELS={'text':'CLIP text','prototype':'Visual prototypes','text_prototype':'Text + prototypes',
        'tip_adapter':'Tip-Adapter (tuned)','reference':'Reference-anchored readout','dynamic':'Candidate-normalized readout'}
def stat(method,k,key):
    x=[np.mean([r[key] for r in R if r['part']==p and r['k']==k and r['method']==method]) for p in (0,1,2)]
    return np.mean(x)*100,np.std(x,ddof=1)*100
# Same plotting operations and frozen arrays, with readout names in legends.
original=(H.parent/'fcs/build_results.py').read_text(encoding='utf-8')
block=original[original.index('plt.rcParams.update'):original.index("print((ROOT/'table_expansion.tex')")]
exec(compile(block,'frozen_auxiliary_plots','exec'),dict(plt=plt,np=np,D=D,R=R,LABELS=LABELS,stat=stat,FIG=H))
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(10.8,5.3))
hashes={};colors={'retained':'#1B4F72','anchored_tp':'#D55E00'}
names={'retained':'Retained','anchored_tp':'Dense prototypes'}
for (enc,d),ax in zip([(e,d) for e in ['l14','b32'] for d in ['cifar','pets','imagenetr']],axes.flat):
    path=ROOT/f'experiments/kbs_mechanism_followup/{enc}/{d}/p0/o0.json'
    r=json.loads(path.read_text());raw=path.with_suffix('.npz')
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['raw_sha256']
    hashes[enc+'/'+d]={'record_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'arrays_sha256':r['raw_sha256']}
    with np.load(raw) as z:
        ceiling=max(row['reference'] for row in r['rows'])
        ceiling=max(ceiling,z['anchored_tp_base'][0])
        for m in colors:
            ax.plot(100*z[m+'_base'],100*z[m+'_novel'],color=colors[m],lw=1.25,marker='.',ms=1.0,label=names[m])
    for row in r['rows']:
        ax.scatter(100*row['base'],100*row['novel'],marker='o' if row['policy']=='paired_accuracy' else '^',
                   s=30,color=colors[row['method']],facecolors='none' if row['fallback'] else colors[row['method']],zorder=4)
    target=r['rows'][0]['target'];ax.axvline(100*target,color='#777777',ls=':',lw=1)
    ax.set_xlim(max(0,100*target-3),min(100,100*ceiling+1));ax.set_ylim(-2,102)
    ax.set_yticks([0,50,100]);ax.grid(axis='y',alpha=.15)
    ax.set_xlabel('Actual base accuracy (%)');ax.set_ylabel('Novel accuracy (%)')
    ax.set_title(('L/14 ' if enc=='l14' else 'B/32 ')+{'cifar':'CIFAR-100','pets':'Pets','imagenetr':'ImageNet-R'}[d])
handles,_=axes[0,0].get_legend_handles_labels()
handles += [Line2D([],[],marker='o',color='#444444',ls='',label='Direct test'),
            Line2D([],[],marker='^',color='#444444',ls='',label='Protection test'),
            Line2D([],[],color='#777777',ls=':',label='95% target'),
            Line2D([],[],marker='o',color='#444444',markerfacecolor='none',ls='',label='Reference fallback')]
fig.legend(handles=handles,ncol=3,loc='lower center',frameon=False,fontsize=9)
fig.tight_layout(rect=[0,.10,1,1])
for ext in ['pdf','png']:fig.savefig(H/f'source_operating_full.{ext}',dpi=250,bbox_inches='tight',pad_inches=0.05)
plt.close(fig)
report={'auxiliary_input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_frontier_inputs':hashes,'figure_values_reused_without_refitting':True,
        'legend_changes_only':['stability.pdf','shots.pdf','tradeoff.pdf']}
(H/'figure_revision_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Updated three auxiliary legends and restored the source trajectory row.')
