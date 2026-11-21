from pathlib import Path
import fitz,base64,xml.etree.ElementTree as E
H=Path(__file__).resolve().parent;A=H/'latent_exact_components';A.mkdir(exist_ok=True)
d=fitz.open(H/'latent_style_assets/source_without_text.pdf');pg=d[0]
def clear(rect,color):pg.draw_rect(fitz.Rect(*(v/1.5 for v in rect)),color=None,fill=color,overlay=True)
# Clean original method-specific symbols while retaining each original frame.
clear((88,64,723,114),(.96,.96,.95))
clear((199,169,672,190),(.64,.79,.75))
clear((198,280,308,338),(.64,.79,.75));clear((419,280,521,338),(.64,.79,.75))
clear((54,407,475,578),(.65,.77,.81))
clear((788,66,1064,231),(.90,.83,.88))
clear((818,258,1060,377),(.84,.84,.83))
clear((675,448,846,586),(.88,.82,.90))
clear((520,551,652,572),(.84,.84,.83))
components={'task':(80,57,734,123),'encoder':(29,162,180,354),'feature':(188,160,682,242),'ridge':(187,268,384,349),'novel':(410,269,599,349),'evidence':(28,384,499,604),'moments':(519,378,653,571),'hub':(685,238,838,386),'references':(780,56,1075,241),'formula':(840,245,1069,386),'calibration':(667,405,856,605),'output':(904,408,1076,545),'arrow1':(137,137,204,190),'arrow2':(652,363,719,415)}
for k,rect in components.items():pg.get_pixmap(matrix=fitz.Matrix(4,4),clip=fitz.Rect(*(v/1.5 for v in rect)),alpha=True).save(A/(k+'.png'))
mx=E.Element('mxfile',host='app.diagrams.net');di=E.SubElement(mx,'diagram',name='Source-matched framework');m=E.SubElement(di,'mxGraphModel',page='1',pageWidth='1090',pageHeight='617');r=E.SubElement(m,'root');E.SubElement(r,'mxCell',id='0');E.SubElement(r,'mxCell',id='1',parent='0')
def cell(k,x,y,w,h,v='',style=''):
 c=E.SubElement(r,'mxCell',id=k,vertex='1',parent='1',value=v,style='html=1;fontFamily=Georgia;fontSize=20;align=center;verticalAlign=middle;whiteSpace=wrap;spacing=0;fillColor=none;strokeColor=none;'+style);E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'});return c
def text(k,x,y,w,h,v,n=20):cell(k,x,y,w,h,v,'fontSize='+str(n)+';')
def img(k,file,x,y,w,h):cell(k,x,y,w,h,style='shape=image;imageAspect=0;image=data:image/png,'+base64.b64encode(file.read_bytes()).decode()+';')
def edge(k,pts):
 c=E.SubElement(r,'mxCell',id=k,edge='1',parent='1',style='endArrow=classic;endSize=6;strokeWidth=2.4;strokeColor=#272727;sketch=1;jiggle=0.7;rounded=1;')
 g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
 for n,(x,y) in zip(['sourcePoint','targetPoint'],[pts[0],pts[-1]]):E.SubElement(g,'mxPoint',x=str(x),y=str(y),attrib={'as':n})
 if len(pts)>2:
  a=E.SubElement(g,'Array',attrib={'as':'points'})
  for x,y in pts[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
cell('bgleft',0,0,761,617,style='fillColor=#FFFAEB;');cell('bgright',761,0,329,617,style='fillColor=#EDF3FB;')
for k,(x,y,x2,y2) in components.items():img(k,A/(k+'.png'),x,y,x2-x,y2-y)
# All component geometry above matches the source PDF, rather than inferred layout.
text('titlea',16,4,735,42,'<b><i>Query-Adaptive Evidence Alignment</i></b>',29)
text('titleb',765,4,315,42,'<b><i>Retention Control</i></b>',28)
text('tasklabel',93,68,621,40,'<b>Query x</b> &nbsp; · &nbsp; class names &nbsp; · &nbsp; few-shot supports',21)
text('encoderlabel',35,207,139,135,'<b>Visual<br>Encoder<br><br>(frozen)</b>',25)
text('featurelabel',213,129,463,29,'<i>Frozen query representation v = f(x)</i>',23)
for j,x in enumerate([210,250,292,335,377,420,522,563,606,646]):text('v'+str(j),x,170,25,21,'<i>v<sub>'+str(j+1)+'</sub></i>',16)
text('ridgetext',194,284,119,48,'Base classifier<br><i>W<sub>B</sub>, b<sub>B</sub></i>',17)
text('noveltext',416,284,106,48,'Class evidence<br><i>t<sub>c</sub>, p<sub>c</sub></i>',17)
text('scoretitle',88,352,396,31,'<i>Asymmetric score construction</i>',23)
for j,y in enumerate([425,476,527]):
 img('scoreicon'+str(j),H/'latent_style_assets/cube_light.png',68,y-4,36,36)
 text('scorename'+str(j),111,y-4,170,36,['Learned head','Semantic anchors','Visual prototypes'][j],18)
 cell('scorebox'+str(j),310,y-3,147,37,'<i>q<sub>'+['L','T','P'][j]+'</sub>(x,'+['b','c','c'][j]+')</i>','fillColor=#E2EEE6;strokeColor=#526871;strokeWidth=1.8;rounded=1;sketch=1;fillStyle=solid;')
 edge('scorearrow'+str(j),[(280,y+15),(308,y+15)])
text('momenttitle',523,381,126,61,'<i>Per-query<br>Moments</i>',22)
text('hubtitle',695,299,134,67,'<b>Reference<br>Alignment</b>',22)
text('reftitle',801,70,253,36,'<b>Fixed base references</b>',21)
for j,x in enumerate([811,896,981]):
 img('refcube'+str(j),H/'latent_style_assets/cube_light.png',x,121,49,48)
 text('reflabel'+str(j),x-12,173,75,30,['<i>T<sub>B</sub></i>','<i>W<sub>B</sub>, b<sub>B</sub></i>','<i>P<sub>B</sub></i>'][j],18)
text('refnote',798,207,253,23,'<i>Fixed across arrivals</i>',17)
text('formtitle',840,253,214,31,'<i>Moment alignment</i>',23)
text('formulaeq',841,293,214,35,'q̂<sub>e</sub> = (q<sub>e</sub> − μ̃<sub>e</sub>)/s̃<sub>e</sub>',19)
text('eta',841,337,214,32,'η<sub>B</sub> = min(1, B/30)',19)
text('caltitle',674,408,175,32,'Calibration Set',21)
for j,x in enumerate([684,738,792]):cell('cal'+str(j),x,464,40,43,'<i>x<sub>'+str(j+1)+'</sub></i>','shape=note;size=0.2;fillColor=#EFE6DC;strokeColor=#594D53;strokeWidth=1.4;')
text('calnote',679,529,163,56,'Independent test<br><i>selects offset d</i>',20)
text('outlabel',909,554,166,50,'<b>Base / Novel</b><br><i>argmax with d</i>',20)
# Explicit forward flow; arrows from the source that imply training are not reused.
edge('queryenc',[(110,123),(110,160)])
edge('featheads',[(391,244),(391,255),(286,255),(286,267)])
edge('featnovel',[(391,255),(500,255),(500,268)])
edge('headscores',[(286,350),(286,378)])
edge('scoresmoments',[(499,489),(517,489)])
edge('momentshub',[(588,378),(588,358),(683,320)])
edge('refhub',[(858,240),(825,265)])
edge('hubformula',[(839,315),(852,315)])
edge('formoutput',[(1015,387),(1015,406)])
edge('caloffset',[(857,494),(903,494)])
for c in r:
 if c.get('id')=='titlea':c.set('style',c.get('style')+';fontColor=#9F5412;')
 if c.get('id')=='titleb':c.set('style',c.get('style')+';fontColor=#165A9D;')
# Frozen icon is reused at its original position.
img('snow',H/'latent_style_assets/snow.png',0,131,79,74)
E.ElementTree(mx).write(H/'framework_source_matched.drawio',encoding='utf-8',xml_declaration=True)
