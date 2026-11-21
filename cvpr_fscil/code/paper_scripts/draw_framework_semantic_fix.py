from pathlib import Path
import xml.etree.ElementTree as E
H=Path(__file__).resolve().parent
src=H.parent/'cvpr2027_revisions/20261105_before_semantic_figure_fix/framework_drawio.drawio'
t=E.parse(src);r=t.find('.//root')
remove={'upper_route','proto_score','top_collect','mid_collect','lower_route','semantic_base','upper_score','fusion_score','lower_score','upper_eq','mid_eq','lower_eq','unit_title'}
start=False
for c in list(r):
 k=c.get('id')
 if k=='rank_calibration':start=True
 if k in remove or (start and k not in ['caption_a','caption_b']):r.remove(c)
def node(k,x,y,w,h,v='',extra=''):
 c=E.SubElement(r,'mxCell',id=k,value=v,vertex='1',parent='1',style='fontFamily=Palatino Linotype;fontSize=13;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;spacing=0;strokeColor=none;fillColor=none;'+extra)
 E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'});return c
def label(k,x,y,w,h,v,size=13):return node(k,x,y,w,h,'<b><i>'+v+'</i></b>',f'fontSize={size};')
def edge(k,pts,color='#E65B00',arrow=True,dash=False):
 c=E.SubElement(r,'mxCell',id=k,edge='1',parent='1',style=f'endArrow={"classic" if arrow else "none"};endSize=4;strokeWidth=1.3;strokeColor={color};'+('dashed=1;' if dash else ''))
 g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
 for name,(x,y) in zip(['sourcePoint','targetPoint'],[pts[0],pts[-1]]):E.SubElement(g,'mxPoint',x=str(x),y=str(y),attrib={'as':name})
 if len(pts)>2:
  a=E.SubElement(g,'Array',attrib={'as':'points'})
  for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
# Three heterogeneous scores share the encoded query.
edge('support_mean',[(414,50),(472,50),(472,72)])
label('mean',420,28,100,20,'Class means',12)
for k,y,s in [('P',136,'q<sub>P</sub>(x,c)'),('T',201,'q<sub>T</sub>(x,c)'),('L',282,'q<sub>L</sub>(x,b)')]:
 edge('score_in'+k,[(538 if k!='L' else 414,y),(554,y)])
 label('score'+k,558,y-15,85,30,s,16)
 edge('collect'+k,[(644,y),(651,y),(651,201)],arrow=False)
label('query_note',421,159,215,22,'All scores use the query feature',12)
edge('to_moments',[(651,201),(668,201),(668,148),(951,148),(951,173)])
# Lower panel: references, per-query moments and aligned evidence.
node('banks',677,170,208,135,extra='rounded=1;arcSize=18;fillColor=#DEEBF7;')
label('banks_title',680,174,202,21,'Fixed base reference banks',13)
label('bank_semantic',682,200,198,18,'Semantic + learned',12)
node('bank_row1',687,220,188,27,'<b><i>T<sub>B</sub>, W<sub>B</sub>, b<sub>B</sub>; μ<sub>L</sub>(x)=1/B</i></b>','strokeColor=#999999;fontSize=12;whiteSpace=nowrap;')
label('bank_visual',682,250,198,18,'Visual + fixed moments',12)
node('bank_row2',687,270,188,27,'<b><i>P<sub>B</sub>; μ̄<sub>e</sub>, s̄<sub>e</sub>; η<sub>B</sub>=min(1,B/30)</i></b>','strokeColor=#999999;fontSize=11;whiteSpace=nowrap;')
node('moments_box',911,175,113,34,'<b><i>Query moments</i></b>','rounded=1;fillColor=#D8D9D9;')
node('alignment_box',911,228,113,38,'<b><i>Reference<br>alignment</i></b>','rounded=1;fillColor=#D8D9D9;')
edge('bank_to_moments',[(886,195),(909,195)])
edge('moments_to_alignment',[(969,210),(969,226)])
node('norm',891,278,149,24,'q̂<sub>e</sub>=(q<sub>e</sub>−μ̃<sub>e</sub>)/s̃<sub>e</sub>','fontSize=13;')
# Schematic scores: identical base bars; scalar d moves every novel bar equally.
label('before_title',681,20,142,20,'Aligned group scores',13)
label('after_title',879,20,155,20,'After novel offset d',13)
for prefix,x,hs in [('before',695,[42,28,25,17]),('after',895,[42,28,49,41])]:
 edge(prefix+'axis',[(x,48),(x,105),(x+111,105)],'#333333',False)
 for j,h in enumerate(hs):node(prefix+'bar'+str(j),x+12+j*24,104-h,12,h,extra='fillColor='+('#456684' if j<2 else '#EE9856')+';')
 label(prefix+'base',x+4,108,52,17,'Base',11)
 label(prefix+'novel',x+57,108,58,17,'Novel',11)
edge('offset_arrow',[(823,78),(878,78)])
label('offset_label',825,55,52,20,'+ d',14)
label('schematic',704,127,322,17,'Schematic scores · prediction = argmax',11)
edge('alignment_to_scores',[(1025,246),(1040,246),(1040,10),(751,10),(751,18)])
# Actual branch equations under the left score lanes.
label('equations',421,305,231,21,'Base: q̂<sub>T</sub> + a q̂<sub>L</sub> &nbsp; Novel: q̂<sub>T</sub> + u q̂<sub>P</sub>',11)
for c in r:
 if c.get('id')=='caption_a':c.find('mxGeometry').set('y','327')
 if c.get('id')=='caption_b':c.find('mxGeometry').set('y','327')
# Extend canvas slightly to keep captions clear.
for c in r:
 if c.get('id')=='canvas':c.find('mxGeometry').set('height','359')
t.write(H/'framework_drawio.drawio',encoding='utf-8',xml_declaration=True)
