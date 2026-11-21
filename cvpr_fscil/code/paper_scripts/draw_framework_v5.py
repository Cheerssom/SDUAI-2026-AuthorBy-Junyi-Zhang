from pathlib import Path
import xml.etree.ElementTree as E
H=Path(__file__).resolve().parent
t=E.parse(H/'framework_v4.drawio')
colors={'v3':('#DED0E4','#67546C'),'v13':('#D7E5C9','#526A44'),'v40':('#D7E5C9','#526A44'),'v64':('#D7E5C9','#526A44'),'v83':('#D9E7B8','#63794C'),'v103':('#DBE7C5','#63794C'),'v125':('#A7CEC5','#4D6D65'),'v128':('#A7CEC5','#4D6D65'),'v152':('#E5D0DE','#A85A5D'),'v156':('#F4F0E8','#756A69'),'v159':('#F6E4E5','#B74B4B'),'v162':('#ABD1C6','#537E73'),'v163':('#DED0E4','#67546C'),'v11':('#F7F5F0','#77716B')}
for c in t.iter('mxCell'):
 st=c.get('style','');k=c.get('id');v=c.get('value','')
 if c.get('vertex')!='1' or 'shape=image' in st:continue
 parts=st.split(';');cfg={};flags=[]
 for z in parts:
  if '=' in z:
   a,b=z.split('=',1);cfg[a]=b
  elif z:flags.append(z)
 cfg['fontFamily']='Palatino Linotype'
 if k in ['v3','v13','v40','v64']:cfg['fontSize']='15'
 if k in ['v130','v165']:cfg['fontSize']='17'
 if k in ['v1','v38','v91','v102','v153']:cfg['fontSize']='13'
 if k in colors:
  fill,stroke=colors[k];cfg.update(fillColor=fill,strokeColor=stroke,strokeWidth='2',sketch='1',jiggle='0.7',curveFitting='0.5',fillStyle='solid',rounded='1',arcSize='9',spacing='3')
 if k in ['v5','v6','v7','v30','v31','v32','v34']:
  cfg.update(sketch='1',jiggle='0.6',fillStyle='solid',strokeWidth='1.4',arcSize='4')
 if k=='v152':cfg.update(arcSize='10',strokeWidth='2.3')
 if k in ['v156','v159']:cfg.update(arcSize='3',strokeWidth='1.4',spacing='2')
 c.set('style',';'.join(flags+[a+'='+b for a,b in cfg.items()])+';')
t.write(H/'framework_v5.drawio',encoding='utf-8',xml_declaration=True)
