"""Original method, BiMC-inspired visual organization. No borrowed diagram content.
Reference canvas 1048x554; original architecture roles and equations retained.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

H=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
 'mathtext.fontset':'stix','font.size':17,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
W=14.;HH=W*554/1048
f,a=plt.subplots(figsize=(W,HH));f.subplots_adjust(0,0,1,1)
a.set(xlim=(0,W),ylim=(0,HH));a.set_axis_off()
C={'text':'#b48a20','textfill':'#fff7dc','visual':'#648148','visualfill':'#eff5e7',
   'learn':'#b66c42','learnfill':'#fceadc','blue':'#467b95','bluefill':'#edf5fa',
   'ink':'#30343a','red':'#a44e48','redfill':'#fff2ef'}
texts=[]
def txt(x,y,s,size=17,color=None,ha='center',weight='normal',**kw):
 t=a.text(x,y,s,fontsize=size,color=color or C['ink'],ha=ha,va='center',weight=weight,**kw);texts.append(t);return t
def box(x,y,w,h,ec,fc='white',dash=False,r=.12,lw=1.05):
 q=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',edgecolor=ec,facecolor=fc,linewidth=lw,linestyle=(0,(8,5)) if dash else '-');a.add_patch(q);return q
def arrow(points,color=None,lw=1.1,dashed=False):
 for p,q in zip(points[:-2],points[1:-1]):a.plot([p[0],q[0]],[p[1],q[1]],color=color or C['ink'],lw=lw,ls='--' if dashed else '-')
 a.annotate('',xy=points[-1],xytext=points[-2],arrowprops={'arrowstyle':'-|>','color':color or C['ink'],'lw':lw,'mutation_scale':12,'linestyle':'--' if dashed else '-'},zorder=4)
def encoder(x,y,w,h,label,col,fill):
 a.add_patch(Polygon([(x,y),(x+w,y+.12),(x+w,y+h-.12),(x,y+h)],closed=True,facecolor=fill,edgecolor=col,lw=1.2));txt(x+w/2,y+h/2,label,17,weight='bold',color=col)
def stack(x,y,label,col,fill):
 for j in [2,1,0]:a.add_patch(Rectangle((x+j*.055,y+j*.045),.95,.20,facecolor=fill,edgecolor=col,lw=.8,alpha=1-j*.18))
 txt(x+.50,y-.21,label,19)
def snow(x,y):
 for ang in np.arange(0,np.pi,.5235988):
  dx,dy=.09*np.cos(ang),.09*np.sin(ang);a.plot([x-dx,x+dx],[y-dy,y+dy],color=C['blue'],lw=1)

# Reference-style dashed modality regions; each branch has one reading direction.
box(.18,5.02,7.15,1.72,'#e1c875',dash=True)
box(.18,3.12,7.15,1.72,'#dfb198',dash=True)
box(.18,1.22,7.15,1.72,'#bed19f',dash=True)
for y,title,col in [(6.48,'Semantic evidence',C['text']),(4.58,'Retained base evidence',C['learn']),(2.68,'Visual support evidence',C['visual'])]:txt(.40,y,title,18,col,ha='left',weight='bold')

box(.45,5.47,1.55,.58,'#d4c9a6','#fffdf7');txt(1.225,5.76,'Class names',16)
encoder(2.40,5.35,1.48,.84,'Text\nencoder',C['text'],C['textfill']);snow(2.60,6.04)
arrow([(2.,5.77),(2.36,5.77)]);arrow([(3.90,5.77),(4.27,5.77)])
stack(4.36,5.67,r'$t_b,\ t_n$',C['text'],C['textfill'])
arrow([(5.42,5.77),(6.00,5.77)]);txt(6.48,5.77,r'$q_T(x,c)$',21)

box(.45,3.55,1.55,.60,'#dfb198','#fff9f5');txt(1.225,3.85,'Base features\n+ labels',16.5)
encoder(2.40,3.45,1.48,.84,'Ridge\nfit',C['learn'],C['learnfill'])
arrow([(2,3.86),(2.36,3.86)],C['learn']);arrow([(3.90,3.86),(4.27,3.86)],C['learn'])
stack(4.36,3.76,r'$W_B,\ b_B$',C['learn'],C['learnfill']);txt(4.85,4.28,'Retain after fitting',16.5,C['learn'])
arrow([(5.42,3.86),(6.0,3.86)]);txt(6.48,3.86,r'$q_L(x,b)$',21)

# Use the user's original diagram thumbnails, not imagery from the reference paper.
from PIL import Image
imgs=[Image.open(H/'figure_assets'/f'input_{i}.png').convert('RGB') for i in range(3)]
for j,im in enumerate(imgs[:3]):
 x=.48+j*.30;y=1.64+j*.06;a.imshow(im,extent=(x,x+.63,y,y+.56),zorder=2+j,aspect='auto')
txt(1.24,1.40,r'Base / $k$-shot images',16.5)
encoder(2.40,1.55,1.48,.84,'Visual\nencoder',C['visual'],C['visualfill']);snow(2.60,2.24)
arrow([(1.91,1.96),(2.36,1.96)]);arrow([(3.90,1.96),(4.27,1.96)]);txt(4.10,2.32,'Mean',16.5)
stack(4.36,1.86,r'$p_b,\ p_n$',C['visual'],C['visualfill']);arrow([(5.42,1.96),(6.0,1.96)]);txt(6.48,1.96,r'$q_P(x,c)$',21)

# All evidence uses the same frozen query feature; record this once instead of crossing all lanes.
box(.20,.41,7.13,.56,'#b7cdd8','#f5f9fc',r=.08)
snow(.48,.69);txt(.73,.69,r'Query image $x\ \longrightarrow\ f(x)=v$; same frozen visual encoder',16,ha='left')

# A narrow, BiMC-like fusion area visually joins the three branches.
box(7.62,1.22,1.22,5.52,'#e4b4ae',C['redfill'],dash=True)
for yy in [5.77,3.86,1.96]:arrow([(7.10,yy),(7.94,yy)])
for yy,col,fill in [(5.77,C['text'],C['textfill']),(3.86,C['learn'],C['learnfill']),(1.96,C['visual'],C['visualfill'])]:
 a.add_patch(Rectangle((7.99,yy-.12),.49,.24,facecolor=fill,edgecolor=col,lw=.9))
txt(8.23,4.90,'Reference\nalignment',17,C['red'],weight='bold')
txt(8.23,2.90,'Evidence\nfusion',17,C['red'],weight='bold')
arrow([(8.24,5.56),(8.24,5.30)],C['red']);arrow([(8.24,3.63),(8.24,3.27)],C['red'])

# Mechanism inset, replacing generic geometric illustrations with the actual operator.
box(9.19,1.22,4.61,5.52,'#a9cddc','#ffffff',dash=True)
txt(11.50,6.46,'Fixed references, adaptive responses',19,C['blue'],weight='bold')
txt(11.50,6.09,r'Base banks $\mathcal{R}_T,\ \mathcal{R}_L,\ \mathcal{R}_P$',18)
for x,label,col,fill in [(9.68,r'$T_B$',C['text'],C['textfill']),(10.99,r'$W_B,b_B$',C['learn'],C['learnfill']),(12.30,r'$P_B$',C['visual'],C['visualfill'])]:
 box(x,5.40,1.02,.43,col,fill,r=.03);txt(x+.51,5.62,label,18)
arrow([(11.50,5.34),(11.50,5.06)],C['blue'])
box(9.61,4.25,3.79,.76,'#a9cddc',C['bluefill'],r=.07)
txt(11.50,4.76,r'Query moments $\mu_e(x),\ s_e(x)$',18)
txt(11.50,4.46,r'Full: $\eta_B=\min(1,B/30)$',17)
arrow([(11.50,4.19),(11.50,3.94)],C['blue'])
txt(11.50,3.66,r'$\widehat q_e(x,c)=\frac{q_e(x,c)-\widetilde\mu_e(x)}{\widetilde s_e(x)}$',24)
arrow([(11.50,3.24),(11.50,3.02)],C['blue'])
box(9.61,1.53,3.79,1.42,'#e4b4ae','#fff7f5',r=.08)
txt(9.82,2.64,r'Base: $\widehat q_T+a\widehat q_L$',20,ha='left')
txt(9.82,2.21,r'Novel: $\widehat q_T+u\widehat q_P+d$',20,ha='left')
txt(11.50,1.76,r'Prediction: $\arg\max_c s(x,c)$',19)
arrow([(8.87,3.86),(9.13,3.86)],C['red'],dashed=True)

txt(4.50,7.08,'(a) Reference-anchored asymmetric readout',19,weight='bold')
txt(11.50,7.08,'(b) Alignment and admission',19,weight='bold')
txt(11.50,.83,r'Append $t_n,p_n$; retain base weights',16)
txt(11.50,.48,'and reference banks across arrivals.',16)

f.savefig(H/'framework_refined.pdf',facecolor='white')
f.savefig(H/'framework_refined.svg',facecolor='white')
f.savefig(H/'framework_refined.png',dpi=190,facecolor='white')
f.canvas.draw();renderer=f.canvas.get_renderer()
overflow=[]
for t in texts:
 bb=t.get_window_extent(renderer)
 if bb.x0<0 or bb.y0<0 or bb.x1>f.bbox.width or bb.y1>f.bbox.height:overflow.append(t.get_text())
assert not overflow,overflow
print('Rendered PDF/SVG/PNG; canvas and text bounds pass.')
