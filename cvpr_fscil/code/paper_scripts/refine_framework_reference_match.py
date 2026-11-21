from pathlib import Path
import runpy, xml.etree.ElementTree as E, base64
H=Path(__file__).resolve().parent
runpy.run_path(str(H/'draw_framework_engine_exact.py'))
t=E.parse(H/'framework_engine_exact.drawio');r=t.find('.//root')
# Keep the measured reference layout; replace illustrative reference photographs.
for c in list(r):
 k=c.get('id','')
 if k=='prototypes':
  r.remove(c)
 if k=='support_label': c.set('value','<b><i>Support images</i></b>')
 if k=='schematic': c.set('value','<b><i>Illustrative scores</i></b>')
def cell(k,x,y,w,h,value='',style=''):
 c=E.SubElement(r,'mxCell',id=k,value=value,vertex='1',parent='1',style=style)
 E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
cell('prototype_card',425,76,94,31,style='rounded=1;arcSize=18;fillColor=#C7DDB5;strokeColor=none;')
for j in range(3):
 data=base64.b64encode((H/f'figure_assets/support_pet_{j}.png').read_bytes()).decode()
 cell('prototype_photo'+str(j),429+27*j,79,23,23,style='shape=image;imageAspect=1;image=data:image/png,'+data+';')
# Place the prototype card behind its curved response connector.
card=[c for c in list(r) if c.get('id','').startswith('prototype_card') or c.get('id','').startswith('prototype_photo')]
for c in card:r.remove(c)
idx=next(i for i,c in enumerate(r) if c.get('id')=='prototype_response')
for j,c in enumerate(card):r.insert(idx+j,c)
t.write(H/'framework_drawio.drawio',encoding='utf-8',xml_declaration=True)
