from pathlib import Path
import xml.etree.ElementTree as E
import base64
H=Path(__file__).resolve().parent
mx=E.Element('mxfile',host='app.diagrams.net');d=E.SubElement(mx,'diagram',name='Reference-anchored expansion');m=E.SubElement(d,'mxGraphModel',page='1',pageWidth='1200',pageHeight='405',background='#ffffff');r=E.SubElement(m,'root');E.SubElement(r,'mxCell',id='0');E.SubElement(r,'mxCell',id='1',parent='0')
def box(k,x,y,w,h,v='',fill='none',size=18,extra=''):
 c=E.SubElement(r,'mxCell',id=k,value=v,vertex='1',parent='1',style=f'html=1;fontFamily=Times New Roman;fontSize={size};align=center;verticalAlign=middle;whiteSpace=wrap;spacing=3;fillColor={fill};strokeColor=#353333;strokeWidth=2;rounded=1;arcSize=12;sketch=1;jiggle=1;fillStyle=solid;'+extra)
 E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'});return c
def txt(k,x,y,w,h,v,size=18,color='#242323'):
 return box(k,x,y,w,h,v,size=size,extra=f'strokeColor=none;fontColor={color};sketch=0;')
def edge(k,pts,dashed=False,color='#292929'):
 c=E.SubElement(r,'mxCell',id=k,edge='1',parent='1',style=f'strokeColor={color};strokeWidth=2.3;endArrow=classic;endSize=7;rounded=1;sketch=1;jiggle=1;'+('dashed=1;' if dashed else ''))
 g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
 for n,(x,y) in zip(['sourcePoint','targetPoint'],[pts[0],pts[-1]]):E.SubElement(g,'mxPoint',x=str(x),y=str(y),attrib={'as':n})
 if len(pts)>2:
  a=E.SubElement(g,'Array',attrib={'as':'points'})
  for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
def cube(k,x,y,fill='#BFDB95',w=28,h=24):
 return box(k,x,y,w,h,fill=fill,extra='shape=cube;size=7;direction=east;')
def snow(k,x,y):txt(k,x,y,28,30,'❄',27,'#3983A0')
def photo(k,file,x,y,w,h):
 a=base64.b64encode((H/'figure_assets'/file).read_bytes()).decode()
 box(k,x,y,w,h,extra='shape=image;imageAspect=1;sketch=0;image=data:image/png,'+a+';')
box('canvas',0,0,1200,405,fill='#FFFFFF',extra='strokeColor=none;sketch=0;')
box('panelA',7,9,689,379,fill='#FFFAEB',extra='strokeColor=#D8D4C9;strokeWidth=1;')
box('panelB',710,9,482,379,fill='#EDF5F5',extra='strokeColor=#CDDADA;strokeWidth=1;')
txt('titleA',20,12,660,34,'<b><i>(a) Frozen representations &amp; asymmetric evidence</i></b>',23,'#65462F')
txt('titleB',721,12,458,34,'<b><i>(b) Reference alignment &amp; admission</i></b>',23,'#214E6E')
# Frozen feature and three evidence sources.
photo('query','query_pet.png',30,68,66,66)
txt('querylabel',24,136,81,23,'<i>Query x</i>',17)
box('encoder',125,64,133,87,'<b>Visual encoder<br>(frozen)</b>','#D9E8CA',19);snow('freeze1',231,49)
edge('queryencoder',[(97,101),(124,101)])
box('queryfeatures',290,70,185,66,fill='#9BCBC4')
for j in range(4):cube('feature'+str(j),306+38*j,87,'#CAE4EC',27,27)
txt('featurelabel',291,137,184,24,'<i>Query feature v = f(x)</i>',18)
edge('encodefeature',[(259,101),(288,101)])
# Three distinct rows, with colored source cards and aligned score cards.
ys=[187,250,313]
labels=['<b>Retained Ridge head</b>','<b>Textual encoder</b>','<b>Class-mean prototypes</b>']
sub=['W<sub>B</sub>, b<sub>B</sub>','Class names → t<sub>c</sub>','Base / few-shot images → p<sub>c</sub>']
for j,(y,title,v) in enumerate(zip(ys,labels,sub)):
 box('source'+str(j),27,y,252,49,title+'<br><i>'+v+'</i>',['#E9D4E1','#DAE7C8','#D1E5DD'][j],17)
 cube('expert'+str(j),299,y+11,['#D5B9D2','#BBCE8B','#92C1AC'][j],31,28)
 edge('sourcearrow'+str(j),[(280,y+25),(297,y+25)])
 edge('expertscore'+str(j),[(332,y+25),(361,y+25)])
 box('score'+str(j),363,y+1,135,47,'<i>q<sub>'+['L','T','P'][j]+'</sub>(x,'+['b','c','c'][j]+')</i>','#A9CCC8',22)
 edge('querytoscore'+str(j),[(475,101),(520,101),(520,169),(350,169),(350,y+7),(361,y+7)])
 edge('scorecollect'+str(j),[(499,y+24),(565,y+24)],False)
box('evidence',566,188,108,173,'<b>Score<br>branches</b><br><br><i>learned<br>semantic<br>visual</i>','#EEE1B9',18)

# Fixed reference bank: frozen across vocabulary expansion.
box('bank',728,59,235,111,'<b><i>Fixed base references</i></b><br><br><i>T<sub>B</sub> · W<sub>B</sub>, b<sub>B</sub> · P<sub>B</sub></i>','#D7E6AF',18,extra='shape=cylinder;size=0.12;')
snow('freeze_bank',931,54)
txt('bankfixed',737,144,216,21,'<i>Held fixed across arrivals</i>',15)
box('moments',986,64,184,98,'<b><i>Query moments</i></b><br>μ<sub>e</sub>(x), s<sub>e</sub>(x)<br><i>η<sub>B</sub> = min(1,B/30)</i>','#E9CCD7',18)
edge('bankmoments',[(964,111),(984,111)])
edge('scoresmoments',[(675,269),(701,269),(701,180),(979,180),(979,144),(985,144)])
box('normalize',745,199,416,49,'<b><i>Aligned evidence</i></b> &nbsp; q̂<sub>e</sub> = (q<sub>e</sub> − μ̃<sub>e</sub>) / s̃<sub>e</sub>','#8FC1B9',20)
edge('momentnorm',[(1078,180),(1078,197)])
box('base',734,270,201,49,'<b>Base</b><br>q̂<sub>T</sub> + a q̂<sub>L</sub>','#CDDDED',18)
box('novel',964,270,210,49,'<b>Novel</b><br>q̂<sub>T</sub> + u q̂<sub>P</sub> + d','#F0D5BD',18)
edge('normbase',[(836,249),(836,268)])
edge('normnovel',[(1078,249),(1078,268)])
box('calibration',735,337,239,34,'<i>Independent calibration → d</i>','#E9D6E6',17)
edge('caliboffset',[(855,372),(855,381),(1181,381),(1181,295),(1175,295)],True)
txt('argmax',985,334,182,32,'<b><i>Predict: argmax</i></b>',20)
# Export native, editable components.
E.ElementTree(mx).write(H/'framework_drawio.drawio',encoding='utf-8',xml_declaration=True)
