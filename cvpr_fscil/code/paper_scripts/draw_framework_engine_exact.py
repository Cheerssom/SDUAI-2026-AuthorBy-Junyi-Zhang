"""Editable draw.io method figure. Visual reference: ENGINE Fig. 2.
All method labels, operators and inputs describe this paper, not ENGINE.
"""
from pathlib import Path
import base64, math, xml.etree.ElementTree as E

H=Path(__file__).resolve().parent
W,HGT=1048,343
mx=E.Element('mxfile',host='app.diagrams.net',version='29.0.0')
d=E.SubElement(mx,'diagram',id='framework',name='Architecture')
m=E.SubElement(d,'mxGraphModel',dx=str(W),dy=str(HGT),grid='0',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(HGT),background='#ffffff',math='0',shadow='0')
r=E.SubElement(m,'root');E.SubElement(r,'mxCell',id='0');E.SubElement(r,'mxCell',id='1',parent='0')
count=0
ORANGE='#E65B00'; INK='#161616';GRAY='#D9DADA';BLUE='#DDEBF7'
inventory=[]
def node(name,x,y,w,h,text='',style=''):
    global count
    count+=1
    c=E.SubElement(r,'mxCell',id=name,value=text,style='fontFamily=Palatino Linotype;fontSize=14;fontColor='+INK+';html=1;whiteSpace=wrap;spacing=0;'+style,vertex='1',parent='1')
    E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    inventory.append((name,x,y,w,h,'native',text))
    return name
def text(name,x,y,w,h,s,sz=18,style=''):
    return node(name,x,y,w,h,s,'text;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize='+str(sz)+';'+style)
def box(name,x,y,w,h,s='',fill=GRAY,stroke='none',sz=18):
    return node(name,x,y,w,h,s,f'rounded=1;arcSize=22;fillColor={fill};strokeColor={stroke};fontSize={sz};')
def line(name,pts,color=ORANGE,arrow=True,dash=False,width=1.4):
    c=E.SubElement(r,'mxCell',id=name,style=f'edgeStyle=none;rounded=0;html=1;endArrow={"classic" if arrow else "none"};endFill=1;endSize=4;strokeColor={color};strokeWidth={width};'+('dashed=1;dashPattern=5 4;' if dash else ''),edge='1',parent='1')
    g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    E.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),attrib={'as':'sourcePoint'})
    E.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),attrib={'as':'targetPoint'})
    if len(pts)>2:
        a=E.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
    inventory.append((name,pts[0],pts[-1],'-','-','native connector',''))
def image(name,path,x,y,w,h):
    data=base64.b64encode(path.read_bytes()).decode()
    node(name,x,y,w,h,style='shape=image;imageAspect=1;aspect=fixed;image=data:image/png,'+data+';')
def snow(name,x,y):
    for k in range(3):
        ang=k*math.pi/3;dx=7*math.cos(ang);dy=7*math.sin(ang)
        line(name+str(k),[(x-dx,y-dy),(x+dx,y+dy)],'#5B9BD5',False,width=1.5)
        for sign in [-1,1]:
            xx=x+sign*dx*.55;yy=y+sign*dy*.55
            for side in [-1,1]:
                aa=ang+sign*math.pi+side*.7
                line(name+str(k)+str(sign)+str(side),[(xx,yy),(xx+3*math.cos(aa),yy+3*math.sin(aa))],'#5B9BD5',False,width=1)
def blocks(name,x,y,fill,stroke):
    for j in range(3):box(name+str(j),x+j*19,y,13,30,fill=fill,stroke=stroke)
    text(name+'dots',x+57,y,22,30,'⋯',23)


import sys
original='--reference' in sys.argv
A=H/'engine_layout_assets'
def choose(ref,ours):return ref if original else ours
def label(name,x,y,w,h,s,sz=14,color='#000000'):
    sz={14:12.9134,17:16.4416}.get(sz,sz)
    return text(name,x,y,w,h,'<b><i>'+s+'</i></b>',sz,'fontColor='+color+';')
def pic(n,x,y,w,h):image(n,A/(n+'.png'),x,y,w,h)
def opaque(name,x,y,w,h):node(name,x,y,w,h,style='fillColor=#ffffff;strokeColor=none;')
node('canvas',0,0,1048,343,style='fillColor=#ffffff;strokeColor=none;')
line('divider',[(660,24),(660,327)],'#BFBFBF',False,width=1.8)
# Original coordinates: four identical 141 x 34 encoder modules.
for n,y,t in [('visual1',34,'Visual Encoder'),('visual2',120,'Visual Encoder'),('text1',185,'Textual Encoder'),('text2',265,choose('Textual Encoder','Ridge Head'))]:
    box(n,273,y,141,34,'<b><i>'+t+'</i></b>','#D8D9D9',sz=14)
    image(n+'_snow',A/'snow.png',392,y-2,22,23)
if original:
    pic('top_cat',178,23,55,56);pic('bottom_cat',178,108,55,55)
else:
    image('top_support',H/'figure_assets/support_pet_2.png',179,24,53,53)
    image('query_image',H/'figure_assets/query_pet.png',179,109,53,53)
line('image_top',[(232,50),(272,50)])
line('image_bottom',[(232,135),(272,135)])
if original:
    line('augmentation',[(205,109),(205,78)])
    label('augmentation_label',101,81,105,22,'Augmentation')
else:
    label('support_label',109,79,124,22,'Supports / query',13)
line('shared_visual',[(343,68),(343,118)],'#303030',False,True,1.6)
label('shared_visual_label',351,83,61,23,'Shared')
if original:
    line('shared_text',[(344,220),(344,264)],'#303030',False,True,1.6)
    label('shared_text_label',351,228,61,23,'Shared')
else:
    # The fourth module is a fitted head, not a second shared text encoder.
    label('fitted_label',346,231,72,22,'Frozen fit',13)
# Input text, retained in exactly the two reference boxes.
node('class_text',161,185,78,34,choose('<b><i>A photo of<br>a cat</i></b>','<b><i>Class<br>names</i></b>'),'fillColor=#ffffff;strokeColor=#999999;align=left;spacingLeft=6;fontSize=14;')
line('text_input',[(240,202),(272,202)])
node('long_input',56,257,192,50,choose('<b><i>Almond-shaped eyes with reflective pupils.<br>Pointed, upright ears that swivel.<br>Sleek, flexible tail used for balance.<br>...</i></b>','<b><i>Base features + labels<br>Weighted ridge fit<br>Retain W<sub>B</sub>, b<sub>B</sub> across arrivals</i></b>'),'fillColor=#ffffff;strokeColor=#999999;fontSize='+('10' if original else '13')+';')
line('ridge_input',[(248,282),(272,282)])
pic('left_query_loop',81,180,83,77)
if not original:
    # Keep the reference loop shape; replace its LLM-specific symbol and word.
    opaque('remove_llm_left',81,207,43,42)
    for j in range(3):
        box('fit_data_'+str(j),85+j*3,211+j*3,26,27,fill='#DDEBF7',stroke='#555555')
    for j in range(3):line('fit_rows'+str(j),[(94,222+j*5),(113,222+j*5)],'#54748B',False,width=1)
    opaque('remove_query_left',112,177,44,16)
opaque('query_text_clean',112,177,49,17)
label('left_query_label',108,171,67,23,choose('Query','Base data'),14,'#FF0000')
# Preserve the green prototype card and dot glyphs from the provided reference.
pic('prototypes',425,72,94,37)
label('prototypes_heading',421,52,99,20,'Prototypes')
# Original curved prototype-to-response arrow (native curve with same anchors).
c=E.SubElement(r,'mxCell',id='prototype_response',style='curved=1;rounded=1;endArrow=classic;endSize=4;endFill=1;strokeColor=#E65B00;strokeWidth=1.4;',edge='1',parent='1')
g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});E.SubElement(g,'mxPoint',x='443',y='108',attrib={'as':'sourcePoint'});E.SubElement(g,'mxPoint',x='425',y='134',attrib={'as':'targetPoint'});ar=E.SubElement(g,'Array',attrib={'as':'points'});E.SubElement(ar,'mxPoint',x='442',y='123');E.SubElement(ar,'mxPoint',x='424',y='121')
for n,y in [('visual_units',119),('text_units',183)]:
    for j in range(3):box(n+str(j),445+24*j,y,16,35,fill='#F6B68D',stroke='#E67834')
    text(n+'_dots',514,y+6,23,25,'••',17)
line('visual_units_in',[(414,136),(445,136)])
line('text_units_in',[(414,201),(445,201)])
label('unit_title',386,156,217,22,choose('Expanding Injection Units','Prototype / semantic responses'),14)
# Copy all three routed orange paths with their original bend coordinates.
line('upper_route',[(414,50),(565,50),(565,74)])
line('proto_score',[(538,138),(565,138),(565,109)])
line('top_collect',[(565,138),(629,138),(629,156)])
line('mid_collect',[(537,201),(629,201),(629,183)])
line('lower_route',[(414,282),(565,282),(565,259)])
line('semantic_base',[(537,201),(565,201),(565,227)])
label('upper_score',520,73,80,35,choose('Similarity<br>Loss','Novel<br>evidence'),14)
label('fusion_score',588,152,68,32,choose('Contrastive<br>Loss','Score<br>branches'),13)
label('lower_score',525,226,80,33,choose('Similarity<br>Loss','Base<br>evidence'),14)
label('upper_eq',576,103,40,20,choose('(Eq. 6)','q<sub>P</sub>'),12,'#FF0000')
label('mid_eq',571,175,41,20,choose('(Eq. 9)','q<sub>T</sub>'),12,'#FF0000')
label('lower_eq',579,256,39,20,choose('(Eq. 5)','q<sub>L</sub>'),12,'#FF0000')
# Right reference panel: retain its chart/curve/gradient-arrow silhouette.
pic('rank_calibration',710,51,288,71)
label('rank_title',704,27,157,24,choose('Top k Prediction Results','Aligned class scores'),14)
label('calibration_title',889,123,89,24,choose('Calibration','Admission'),14)
label('calibration_eq',917,144,49,18,choose('(Eq. 14)','offset d'),12,'#FF0000')
pic('red_query_arrow',752,117,42,43)
pic('llm',795,116,41,41)
if not original:
    opaque('erase_red_query',752,117,42,22)
    label('bank_label',749,119,49,19,'Fixed',13,'#FF0000')
    opaque('erase_right_llm',795,116,41,41)
    for j in range(3):
        box('bank_tile'+str(j),800+j*5,120+j*4,26,24,fill='#F7D2B5',stroke='#555555')
    for j in range(3):line('bank_rows'+str(j),[(811,133+j*4),(829,133+j*4)],'#C17739',False,width=1)
node('reference_panel',677,157,215,153,style='rounded=1;arcSize=20;absoluteArcSize=1;fillColor=#DEEBF7;strokeColor=none;')
label('panel_heading',681,157,211,24,choose('Pair-Wise Discriminative Features','Fixed base reference banks'),14)
pic('pair_icon1',686,182,22,20);pic('pair_icon2',746,182,22,20)
pic('pair_icon3',686,241,23,22);pic('pair_icon4',746,241,22,22)
if original:
    label('has1',709,181,33,22,'has,',13)
    label('nothave1',767,181,113,22,'does not have:',13)
if original:
    label('has2',709,240,33,22,'has,',13)
    label('nothave2',767,240,113,22,'does not have:',13)
if not original:
    # Same small icon positions; no paired positive/negative claims in our method.
    opaque('hide_pair_icon2',746,182,22,20);opaque('hide_pair_icon4',746,241,22,22)
    # Overlay with the same blue as the parent; redraw labels above the cover.
    node('cover_icon2',746,182,22,20,style='fillColor=#DEEBF7;strokeColor=none;')
    node('cover_icon4',746,241,22,22,style='fillColor=#DEEBF7;strokeColor=none;')
    label('our_pair_label1',710,181,177,22,'Semantic + learned',13)
    label('our_pair_label2',710,240,177,22,'Visual + fixed moments',13)
node('pair_text1',692,205,189,33,choose('<b><i>Pointed ears that are usually<br>upright and expressive</i></b>','<i>T<sub>B</sub>, W<sub>B</sub>, b<sub>B</sub><br>μ<sub>L</sub>(x) = 1/B</i>'),'fillColor=none;strokeColor=#999999;fontSize=13;align=left;spacingLeft=4;')
node('pair_text2',692,264,189,34,choose('<b><i>Stockier, heavier tail with a tuft<br>of hair at the end</i></b>','<i>P<sub>B</sub>; μ̄<sub>e</sub>, s̄<sub>e</sub><br>Full: η<sub>B</sub> = min(1, B/30)</i>'),'fillColor=none;strokeColor=#999999;fontSize=13;align=left;spacingLeft=4;')
text('panel_dots',687,293,25,20,'••',17)
pic('outline_arrow',889,203,29,37)
box('small_visual',912,183,70,35,'<b><i>'+choose('Visual<br>Encoder','Query<br>moments')+'</i></b>','#D8D9D9',sz=14)
box('small_text',912,227,70,35,'<b><i>'+choose('Textual<br>Encoder','Reference<br>alignment')+'</i></b>','#D8D9D9',sz=14)
if not original:
    text('normalization',903,268,130,35,'q̂<sub>e</sub> = (q<sub>e</sub> − μ̃<sub>e</sub>) / s̃<sub>e</sub>',12)
    label('schematic',857,28,162,20,'(schematic)',11)
label('caption_a',217,311,389,28,choose('(a) On-the-fly Knowledge Injection','(a) Asymmetric evidence construction'),17,'#FF0000')
label('caption_b',690,311,347,28,choose('(b) Post-tuning Knowledge Injection','(b) Reference-anchored admission'),17,'#FF0000')
stem='engine_layout_reference' if original else 'framework_engine_exact'
E.ElementTree(mx).write(H/(stem+'.drawio'),encoding='utf-8',xml_declaration=True)
import json
(H/(stem+'_geometry.json')).write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
print(stem)
