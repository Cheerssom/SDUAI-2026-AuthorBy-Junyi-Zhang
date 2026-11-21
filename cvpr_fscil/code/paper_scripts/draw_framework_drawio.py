"""Editable draw.io method figure. Visual reference: ENGINE Fig. 2.
All method labels, operators and inputs describe this paper, not ENGINE.
"""
from pathlib import Path
import base64, math, xml.etree.ElementTree as E

H=Path(__file__).resolve().parent
W,HGT=1140,374
mx=E.Element('mxfile',host='app.diagrams.net',version='29.0.0')
d=E.SubElement(mx,'diagram',id='framework',name='Architecture')
m=E.SubElement(d,'mxGraphModel',dx=str(W),dy=str(HGT),grid='0',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(HGT),background='#ffffff',math='0',shadow='0')
r=E.SubElement(m,'root');E.SubElement(r,'mxCell',id='0');E.SubElement(r,'mxCell',id='1',parent='0')
count=0
ORANGE='#E55B0B'; INK='#161616';GRAY='#D9DADA';BLUE='#DDEBF7'
inventory=[]
def node(name,x,y,w,h,text='',style=''):
    global count
    count+=1
    c=E.SubElement(r,'mxCell',id=name,value=text,style='fontFamily=Times New Roman;fontSize=18;fontColor='+INK+';html=1;whiteSpace=wrap;spacing=0;'+style,vertex='1',parent='1')
    E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    inventory.append((name,x,y,w,h,'native',text))
    return name
def text(name,x,y,w,h,s,sz=18,style=''):
    return node(name,x,y,w,h,s,'text;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize='+str(sz)+';'+style)
def box(name,x,y,w,h,s='',fill=GRAY,stroke='none',sz=18):
    return node(name,x,y,w,h,s,f'rounded=1;arcSize=22;fillColor={fill};strokeColor={stroke};fontSize={sz};')
def line(name,pts,color=ORANGE,arrow=True,dash=False,width=1.6):
    c=E.SubElement(r,'mxCell',id=name,style=f'edgeStyle=none;rounded=0;html=1;endArrow={"classic" if arrow else "none"};endFill=1;endSize=6;strokeColor={color};strokeWidth={width};'+('dashed=1;dashPattern=5 4;' if dash else ''),edge='1',parent='1')
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

# Page-sized white background fixes the same 3.05:1 ratio as the supplied image.
node('canvas',0,0,W,HGT,style='fillColor=#ffffff;strokeColor=none;')
line('panel_separator',[(714,18),(714,341)],'#BFBFBF',False,width=1.8)

# Four tight lanes, with gray rounded encoders, as in the user's reference.
image('query_image',H/'figure_assets/query_pet.png',80,21,54,54)
text('query_label',37,76,140,24,'<i>Query image x</i>',18)
box('query_encoder',185,31,166,36,'<b><i>Visual Encoder</i></b>');snow('snow_query',337,38)
line('query_to_encoder',[(135,49),(183,49)])
line('query_to_feature',[(352,49),(400,49)])
box('feature_vector',402,32,67,34,'<i>v = f(x)</i>','#C7DDB5',sz=20)
line('feature_to_scores',[(470,49),(509,49)])
text('query_use',516,25,174,47,'Shared query feature<br>for all three scores',18)

text('fit_inputs',38,110,115,45,'Base features<br>+ labels',18)
box('ridge',185,115,166,36,'<b><i>Base Ridge Head</i></b>')
line('fit_to_ridge',[(154,133),(183,133)])
line('ridge_to_weights',[(352,133),(400,133)])
blocks('weights',403,118,'#F7B78B','#E57028')
text('weight_label',387,151,115,24,'<i>W<sub>B</sub>, b<sub>B</sub></i>',19)
text('retained_label',380,94,129,23,'<i>Retained weights</i>',17)
line('weights_to_score',[(482,133),(535,133)])
text('score_L',539,116,110,34,'<i>q<sub>L</sub>(x,b)</i>',22)

box('class_names',38,195,115,37,'<i>Class names</i>','#ffffff','#999999')
box('text_encoder',185,196,166,36,'<b><i>Textual Encoder</i></b>');snow('snow_text',337,203)
line('names_to_text',[(154,214),(183,214)])
line('text_to_anchors',[(352,214),(400,214)])
blocks('anchors',403,199,'#F7B78B','#E57028')
text('text_label',389,232,111,24,'<i>t<sub>b</sub>, t<sub>n</sub></i>',19)
line('anchors_to_score',[(482,214),(535,214)])
text('score_T',539,197,110,34,'<i>q<sub>T</sub>(x,c)</i>',22)

image('support0',H/'figure_assets/support_pet_0.png',57,275,41,41)
image('support1',H/'figure_assets/support_pet_1.png',82,269,41,41)
image('support2',H/'figure_assets/support_pet_2.png',107,263,41,41)
text('support_label',16,319,160,22,'Base / <i>k</i>-shot images',17)
box('visual_encoder',185,278,166,36,'<b><i>Visual Encoder</i></b>');snow('snow_support',337,285)
line('supports_to_visual',[(150,296),(183,296)])
line('visual_to_prototypes',[(352,296),(400,296)])
text('mean_label',349,270,49,23,'<i>Mean</i>',17)
blocks('prototypes',403,281,'#F7B78B','#E57028')
text('proto_label',389,314,111,24,'<i>p<sub>b</sub>, p<sub>n</sub></i>',19)
line('protos_to_scores',[(482,296),(535,296)])
text('score_P',539,279,110,34,'<i>q<sub>P</sub>(x,c)</i>',22)

# Collect branch evidence with explicit straight orthogonal routes.
for key,y in [('L',133),('T',214),('P',296)]:
    line('collect_'+key,[(650,y),(684,y),(684,214)],arrow=False)
line('evidence_to_alignment',[(684,214),(735,214),(735,156),(756,156)])

# Right inset: fixed cohorts supply moments, not the growing candidate list.
text('bank_heading',770,12,334,28,'<b><i>Fixed base-class references</i></b>',20)
for key,x,label,fill in [('t',779,'T<sub>B</sub>','#F8D7B9'),('l',891,'W<sub>B</sub>, b<sub>B</sub>','#F8D7B9'),('p',1003,'P<sub>B</sub>','#C7DDB5')]:
    box('bank_'+key,x,45,90,34,'<i>'+label+'</i>',fill,sz=19)
line('banks_to_operator',[(938,79),(938,103)])
box('alignment',756,105,362,110,fill=BLUE)
text('alignment_title',772,111,330,25,'<b><i>Reference-anchored alignment</i></b>',20)
text('moments',769,140,336,29,'Query moments μ<sub>e</sub>(x), s<sub>e</sub>(x)',20)
text('interpolation',768,173,338,28,'Full: η<sub>B</sub> = min(1, B/30)',20)
line('aligned_to_scores',[(938,216),(938,240)])
text('formula',771,239,332,36,'q̂<sub>e</sub> = (q<sub>e</sub> − μ̃<sub>e</sub>) / s̃<sub>e</sub>',23)
box('base_score',758,283,160,42,'Base: q̂<sub>T</sub> + a q̂<sub>L</sub>','#F2F2F2',sz=19)
box('novel_score',931,283,187,42,'Novel: q̂<sub>T</sub> + u q̂<sub>P</sub> + d','#FCE5D5',sz=19)
line('fork_base',[(937,274),(937,278),(839,278),(839,282)],width=1.4)
line('fork_novel',[(937,278),(1023,278),(1023,282)],width=1.4)

# Red italic panel captions are a defining cue of the supplied ENGINE figure.
text('panel_a',16,347,682,26,'<b><i>(a) Frozen representations and asymmetric evidence</i></b>',21,'fontColor=#D32F2F;')
text('panel_b',727,347,400,26,'<b><i>(b) Query alignment and admission</i></b>',21,'fontColor=#D32F2F;')

out=H/'framework_drawio.drawio'
E.ElementTree(mx).write(out,encoding='utf-8',xml_declaration=True)
notes=['# Draw.io architecture revision','',
'Reference: user-supplied ENGINE-style figure, 1048 × 343. Output: 1140 × 374 (ratio difference < 0.3%).',
'Style inventory: white canvas; 63%/37% split; gray rounded encoders; orange 1.4–1.6 px arrows; serif italic/bold labels; salmon feature bars; blue alignment panel; red italic bottom captions.',
'Content adaptation: omit ENGINE-specific LLM icons, losses and pairwise descriptions; use this paper’s actual query, Ridge weights, semantic anchors, prototypes, fixed base moments and group scores.',
'Media: native draw.io text/shapes/connectors; existing local Oxford-IIIT Pets example photographs are embedded PNGs. No figure-sized background image.',
'','## Visible element inventory','| id | geometry | medium | content |','|---|---|---|---|']
for name,x,y,w,h,medium,content in inventory:notes.append(f'| {name} | {x}, {y}, {w}, {h} | {medium} | {content} |')
notes+=['','## Visual audit','Pending exported-image and manuscript-size inspection.']
(H/'framework_drawio.audit.md').write_text('\n'.join(notes),encoding='utf-8')
print(out)
