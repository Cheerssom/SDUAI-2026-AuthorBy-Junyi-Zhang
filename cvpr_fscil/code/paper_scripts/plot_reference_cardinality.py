"""Plot descriptive reference-subset sensitivity, without iid rate claims."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;S=json.loads((H/'shrinkage/cardinality_diagnostic.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(8.2,5.5))
colors={'pets':'#A33D64','cifar':'#D55E00','imagenetr':'#1B4F72'}
for row,stage in enumerate(['pseudo','full']):
    for enc in ['l14','b32']:
        for ds in ['pets','cifar','imagenetr']:
            rr=sorted([r for r in S['conditions'] if r['stage']==stage and r['encoder']==enc and r['dataset']==ds],key=lambda r:r['subset_count'])
            for j,key in enumerate(['normalized_mean_rms','relative_scale_rms']):
                axes[row,j].plot([r['subset_count'] for r in rr],[r[key] for r in rr],
                                 color=colors[ds],ls='-' if enc=='l14' else '--',marker='o' if enc=='l14' else 's',ms=3,
                                 label={'pets':'Pets','cifar':'CIFAR-100','imagenetr':'ImageNet-R'}[ds]+' '+('L/14' if enc=='l14' else 'B/32'))
    for j in range(2):
        axes[row,j].set_xscale('log',base=2);axes[row,j].set_xlabel('Reference subset size')
        axes[row,j].set_ylabel(['Normalized mean RMS deviation','Relative scale RMS deviation'][j])
        axes[row,j].set_title(('Pseudo-base' if stage=='pseudo' else 'Full base')+' reference bank');axes[row,j].grid(alpha=.15)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',frameon=False,fontsize=8)
fig.tight_layout(rect=[0,.11,1,1])
for ext in ['pdf','png']:fig.savefig(H/f'reference_cardinality.{ext}',dpi=220,bbox_inches='tight',pad_inches=.05)
plt.close(fig)
print('Plotted fixed-model reference-subset sensitivity.')
