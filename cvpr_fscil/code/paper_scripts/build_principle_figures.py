"""Vector figures for the principle and frozen query-moment evidence."""
from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
H=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
f,axs=plt.subplots(1,2,figsize=(3.25,1.95),gridspec_kw={'width_ratios':[1,1.05]})
for ax in axs:ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
a=axs[0];a.text(.02,.98,'(a) An existing pair can flip',va='top',fontsize=6.8,weight='bold');a.text(.02,.83,'Text: (0, 2); learned: (+.25, −.25)',fontsize=5.8)
a.text(.02,.71,'Fused old scores',fontsize=6.3,weight='bold');a.text(.57,.71,'c₁',fontsize=6.3);a.text(.78,.71,'c₂',fontsize=6.3)
for y,label,scores,color in [(.56,'Before',('−.75','+.75'),'#333333'),(.39,'Append 100',('−.48','−.94'),'#b95425'),(.21,'Fixed base',('−.75','+.75'),'#165777')]:
 a.text(.02,y,label,fontsize=6.0,color=color);a.text(.56,y,scores[0],fontsize=6,color=color);a.text(.79,y,scores[1],fontsize=6,color=color)
a.text(.02,.04,'Candidate frame: c₂ → c₁\nFixed reference: c₂ → c₂',fontsize=6.1,color='#165777')
a=axs[1];a.text(.02,.98,'(b) Adapt to x, freeze the bank',va='top',fontsize=6.8,weight='bold')
def box(y,text,color):
 a.add_patch(FancyBboxPatch((.04,y),.93,.145,boxstyle='round,pad=.015',linewidth=.6,edgecolor=color,facecolor='#f7fafc'));a.text(.505,y+.0725,text,ha='center',va='center',fontsize=6.1)
box(.70,'Frozen feature  f(x)','#667085');box(.47,'Text / Ridge / visual references\nFixed base cohort  ℛₑ','#165777');box(.24,'Query moments  μₑ(x), σₑ(x)\nAnchor and fuse evidence','#165777');box(.01,'Base score  vs.  novel score + d','#b95425')
for y in [.70,.47,.24]:a.annotate('',xy=(.5,y-.08),xytext=(.5,y-.025),arrowprops={'arrowstyle':'->','lw':.7})
f.subplots_adjust(left=.01,right=.99,top=.99,bottom=.03,wspace=.13);f.savefig(H/'principle_framework.pdf',bbox_inches='tight',pad_inches=.035);f.savefig(H/'principle_framework.png',dpi=220,bbox_inches='tight',pad_inches=.035);plt.close(f)
D=json.loads((H/'prospective/summary.json').read_text());f,axes=plt.subplots(1,2,figsize=(6.9,2.25),sharey=True)
for en,ax in zip(['l14','b32'],axes):
 rr=[r for r in D['contrasts'] if r['block']=='mechanism' and r['dataset']=='imagenetr' and r['encoder']==en and r['baseline'].startswith('shuffle')];rr=sorted(rr,key=lambda r:r['baseline']);m=[r['capacity95']['mean'] for r in rr];lo=[r['capacity95']['ci95'][0] for r in rr];hi=[r['capacity95']['ci95'][1] for r in rr]
 ax.bar(np.arange(3),m,color='#1b6384',width=.58);ax.errorbar(np.arange(3),m,yerr=[np.array(m)-lo,np.array(hi)-m],fmt='none',color='#333333',capsize=3,lw=1);ax.set_xticks(range(3),['3101','3102','3103']);ax.set_xlabel('Prespecified permutation seed',fontsize=8);ax.set_title('CLIP '+('L/14' if en=='l14' else 'B/32'),fontsize=9);ax.set_ylim(0,11);ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Full − shuffled-moment capacity (pp)',fontsize=8);f.tight_layout();f.savefig(H/'query_shuffle_main.pdf',bbox_inches='tight',pad_inches=.05);plt.close(f)
