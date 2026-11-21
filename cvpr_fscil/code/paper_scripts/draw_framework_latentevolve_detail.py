from pathlib import Path
import xml.etree.ElementTree as E
import base64
H=Path(__file__).resolve().parent
mx=E.Element('mxfile',host='app.diagrams.net');d=E.SubElement(mx,'diagram',name='Reference-anchored expansion');m=E.SubElement(d,'mxGraphModel',page='1',pageWidth='912',pageHeight='543',background='#ffffff');r=E.SubElement(m,'root');E.SubElement(r,'mxCell',id='0');E.SubElement(r,'mxCell',id='1',parent='0')
def box(k,x,y,w,h,v='',fill='none',size=18,extra=''):
 c=E.SubElement(r,'mxCell',id=k,value=v,vertex='1',parent='1',style=f'html=1;fontFamily=Georgia;fontSize={size};align=center;verticalAlign=middle;whiteSpace=wrap;spacing=3;fillColor={fill};strokeColor=#353333;strokeWidth=2;rounded=1;arcSize=12;sketch=1;jiggle=0.6;fillStyle=solid;'+extra)
 E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'});return c
def txt(k,x,y,w,h,v,size=18,color='#242323'):
 return box(k,x,y,w,h,v,size=size,extra=f'strokeColor=none;fontColor={color};sketch=0;')
def edge(k,pts,dashed=False,color='#292929'):
 c=E.SubElement(r,'mxCell',id=k,edge='1',parent='1',style=f'strokeColor={color};strokeWidth=2.3;endArrow=classic;endSize=7;rounded=1;sketch=1;jiggle=0.6;'+('dashed=1;' if dashed else ''))
 g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
 for n,(x,y) in zip(['sourcePoint','targetPoint'],[pts[0],pts[-1]]):E.SubElement(g,'mxPoint',x=str(x),y=str(y),attrib={'as':n})
 if len(pts)>2:
  a=E.SubElement(g,'Array',attrib={'as':'points'})
  for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
def icon(k,file,x,y,w,h):
 a=base64.b64encode((H/'latent_style_assets'/file).read_bytes()).decode()
 return box(k,x,y,w,h,extra='shape=image;imageAspect=1;sketch=0;image=data:image/png,'+a+';')
def cube(k,x,y,fill='#BFDB95',w=28,h=24):
 return icon(k,'cube_light.png',x,y,w,h)
def snow(k,x,y):
 icon(k,'snow.png',x-5,y-5,40,40)
def photo(k,file,x,y,w,h):
 a=base64.b64encode((H/'figure_assets'/file).read_bytes()).decode()
 box(k,x,y,w,h,extra='shape=image;imageAspect=1;sketch=0;image=data:image/png,'+a+';')

box('canvas',0,0,912,543,fill='#FFFFFF',extra='strokeColor=none;sketch=0;')
box('leftbg',28,38,609,495,fill='#FFFAEC',extra='strokeColor=none;rounded=0;sketch=0;')
box('rightbg',637,38,263,495,fill='#EFF1E3',extra='strokeColor=none;rounded=0;sketch=0;')
txt('headA',36,40,570,36,'<b><i>(a) Query-adaptive evidence alignment</i></b>',22,'#925015')
txt('headB',641,41,254,35,'<b><i>(b) Retention control</i></b>',22,'#145886')
# Top input strip, matching the long task strip of the reference.
box('input',95,86,519,49,fill='#F6F6F3',extra='strokeColor=#686760;')
photo('query_image','query_pet.png',101,91,37,38)
txt('input_words',148,91,456,38,'<b>Query x</b> &nbsp; + &nbsp; class names &nbsp; + &nbsp; few-shot supports',18)
# Encoder and query feature strip.
box('visual_encoder',54,173,117,144,'<b>Visual<br>encoder<br><br>(frozen)</b>','#D5E6C7',21,extra='strokeColor=#526B45;')
snow('frozen_icon',34,147)
edge('input_encoder',[(116,136),(116,169)],color='#768A61')
box('feature_strip',181,171,423,58,fill='#A5CEC4',extra='strokeColor=#4C6860;')
txt('feature_title',182,140,410,28,'<i>Frozen query representation  v = f(x)</i>',19)
for j in range(8):
 x=195+j*48
 box('feature_token'+str(j),x,194,29,27,fill='#CEE2F2',extra='strokeWidth=1.5;arcSize=2;')
 txt('tokenlabel'+str(j),x-2,172,33,20,'<i>v<sub>'+str(j+1)+'</sub></i>',15)
edge('encoder_feature',[(171,197),(181,197)])
# Compact model cards: existing base supervision and few-shot extension.
box('retained',181,255,190,61,fill='#A5CEC4',extra='strokeColor=#4C6860;')
txt('retained_label',187,261,131,49,'Retained head<br><i>W<sub>B</sub>, b<sub>B</sub></i>',14)
cube('ridge_cube',324,271,'#C7DA90',35,32)
box('novel_sources',397,255,177,61,fill='#A5CEC4',extra='strokeColor=#4C6860;')
txt('novel_label',406,262,103,47,'Names / supports<br><i>t<sub>c</sub>, p<sub>c</sub></i>',12)
cube('prototype_cube',532,270,'#9EC6A6',33,31)
edge('feature_to_readouts',[(390,230),(390,242),(277,242),(277,253)])
edge('feature_to_novel',[(390,242),(500,242),(500,253)])
# Bottom evidence region matches the dashed blue optimization area in the reference.
box('evidence_region',54,351,369,170,fill='#A6C6D0',extra='dashed=1;dashPattern=5 4;strokeColor=#66808A;arcSize=13;')
txt('evidence_heading',64,323,350,28,'<i>Asymmetric score construction</i>',19)
# Three icon families represent fitted weights, labels and visual prototypes.
for j,y in enumerate([370,417,464]):
 color=['#DABED0','#C7DE9E','#ACD3BC'][j]
 if j==1:
  box('tag',73,y+2,38,26,'<i>c</i>',color,18,extra='shape=mxgraph.basic.rect;')
 else:cube('evidence_icon'+str(j),75,y+1,color,35,29)
 txt('branch_name'+str(j),116,y-1,140,29,['Learned head','Semantic anchors','Visual prototypes'][j],17)
 box('score_card'+str(j),278,y-1,122,32,'<i>q<sub>'+['L','T','P'][j]+'</sub>(x,c)</i>','#E8F0E6',19,extra='strokeWidth=1.5;')
 edge('branch_score'+str(j),[(255,y+15),(277,y+15)])
edge('head_to_evidence',[(277,317),(277,342),(235,342),(235,350)])
# Center narrow moment card, in the reference momentum-transfer position.
box('moments',444,364,102,145,fill='#E4E2D8',extra='strokeColor=none;rounded=0;')
txt('moments_title',446,368,98,47,'<i>Per-query<br>moments</i>',19)
for j,(x,y,col) in enumerate([(455,426,'#D4E696'),(497,426,'#C3D993'),(455,468,'#B1CFAC'),(497,468,'#90B7A3')]):cube('moment_cube'+str(j),x,y,col,30,25)
# Alignment hub: native concentric icon instead of an unrelated brain / yarn image.
box('alignment_hub',579,236,119,113,fill='#E8C9D6',extra='ellipse;shape=ellipse;strokeColor=#76494E;strokeWidth=3;')
for j in range(3):
 edge('slider'+str(j),[(616,252+j*9),(661,252+j*9)],color='#6D424F')
 box('slider_knob'+str(j),[628,645,632][j],248+j*9,7,8,fill='#AC6C83',extra='strokeWidth=1;sketch=0;')
txt('alignment_text',581,285,115,57,'<b>Reference<br>alignment</b>',20)
edge('scores_to_moments',[(423,438),(442,438)])
edge('moments_hub',[(495,364),(495,341),(575,307)])
# Upper-right pink reference card, same silhouette as the reference consolidation box.
box('reference_card',654,86,231,140,fill='#E9CAD8',extra='strokeColor=#AD5854;strokeWidth=2.6;')
snow('bank_frozen',660,93)
txt('reference_title',693,91,183,39,'<b>Fixed base references</b>',16)
for j,x in enumerate([676,746,816]):cube('bankcube'+str(j),x,143,['#C9DE9E','#B2CEA2','#9FC4AF'][j],39,31)
for j,x in enumerate([665,735,805]):txt('banklabel'+str(j),x,177,61,28,['<i>T<sub>B</sub></i>','<i>W<sub>B</sub>, b<sub>B</sub></i>','<i>P<sub>B</sub></i>'][j],17)
txt('fixed_rule',665,201,210,21,'<i>Unchanged across arrivals</i>',15)
edge('refs_to_hub',[(719,227),(687,254)])
# Normalization inset occupies the reference instantiation panel.
box('normalization',721,245,164,102,fill='#DCDCD7',extra='strokeColor=none;rounded=0;')
txt('normalization_title',727,248,153,26,'<i>Moment alignment</i>',16)
txt('normalization_formula',726,277,155,29,'q̂<sub>e</sub> = (q<sub>e</sub> − μ̃<sub>e</sub>)/s̃<sub>e</sub>',14)
txt('shrinkage',726,311,155,27,'η<sub>B</sub> = min(1,B/30)',16)
edge('hub_formula',[(699,293),(720,293)])
# Independent calibration set: cylinder and small held-out sample cards.
box('calibration',565,377,153,144,fill='#DECEE5',extra='shape=cylinder;size=0.12;strokeColor=#66516D;')
txt('cal_title',569,378,145,32,'<b>Calibration set</b>',18)
for j,x in enumerate([581,621,661]):
 box('cal_sample'+str(j),x,421,29,34,'<i>x<sub>'+str(j+1)+'</sub></i>','#EFE6D7',16,extra='shape=note;size=0.2;strokeWidth=1.3;')
txt('cal_stat',575,463,135,43,'<i>Retention test<br>selects offset d</i>',18)
# Layered final score cards, similar to reference high-confidence subset.
box('output_layer',747,374,140,117,fill='#83BCAF',extra='strokeColor=none;rounded=0;')
box('output_card',735,386,140,113,fill='#A1CCC0',extra='strokeColor=#52796B;')
txt('baseout',738,389,133,42,'<b>Base scores</b><br>q̂<sub>T</sub> + a q̂<sub>L</sub>',16)
txt('novelout',738,443,133,46,'<b>Novel scores</b><br>q̂<sub>T</sub> + u q̂<sub>P</sub> + d',16)
edge('norm_output',[(803,348),(803,372)])
edge('cal_offset',[(719,468),(733,468)],True)
txt('prediction',728,502,166,27,'<b><i>Predict by argmax</i></b>',16)
# Explicit scope of the ridge branch.
for c in r:
 if c.get('id')=='score_card0':c.set('value','<i>q<sub>L</sub>(x,b)</i>')
# Source-PDF frame artwork, with text removed; native labels remain editable.
frame_map={'input':('task_frame.png',95,86,519,49), 'retained':('green_card.png',181,255,190,61), 'novel_sources':('green_card.png',397,255,177,61)}
for c in list(r):
 k=c.get('id')
 if k in frame_map:
  file,x,y,w,h=frame_map[k]
  idx=list(r).index(c)
  value=c.get('value','')
  r.remove(c)
  new=icon(k,file,x,y,w,h)
  r.remove(new);r.insert(idx,new)
  if value:txt(k+'_label',x+4,y+5,w-8,h-10,value,20)
for c in list(r):
 if c.get('id') in ['ridge_cube','prototype_cube']:r.remove(c)
E.ElementTree(mx).write(H/'framework_latentevolve_detail.drawio',encoding='utf-8',xml_declaration=True)
