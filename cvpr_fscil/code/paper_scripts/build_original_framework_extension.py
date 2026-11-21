from pathlib import Path
import fitz,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':17,'pdf.fonttype':42})
f,a=plt.subplots(figsize=(2.8,7.55));a.set_axis_off();a.set_xlim(0,1);a.set_ylim(0,1)
a.text(.02,.94,'Ranking\ncounterexample',fontsize=17,weight='bold')
a.text(.02,.84,'Old text scores\n(0, 2)\n\nLearned contributions\n(+0.25, −0.25)',va='top',linespacing=1.55,fontsize=17)
a.text(.02,.52,'Fused old scores',weight='bold',fontsize=17)
for y,label,values,color in [(.45,'Before append','−0.75, +0.75','#333333'),(.31,'Append text score 100','−0.48, −0.94','#ac4c25'),(.17,'Fixed base references','−0.75, +0.75','#275987')]:
 a.text(.02,y,label,fontsize=16,color=color);a.text(.02,y-.055,values,fontsize=18,color=color)
a.text(.02,.025,'Candidate frame: 2 → 1\nFixed references: 2 → 2',fontsize=16,color='#275987',linespacing=1.5)
f.subplots_adjust(left=.03,right=.98,top=.99,bottom=.01);f.savefig(H/'ranking_counterexample_panel.pdf');plt.close(f)
orig=fitz.open(H/'framework.pdf');panel=fitz.open(H/'ranking_counterexample_panel.pdf');out=fitz.open();pg=out.new_page(width=1110,height=544.08)
pg.show_pdf_page(fitz.Rect(0,0,204,544.08),panel,0)
pg.draw_line((213,12),(213,532),color=(.7,.7,.7),width=.7)
pg.show_pdf_page(fitz.Rect(224,0,1110,544.08),orig,0)
out.save(H/'framework_with_counterexample.pdf',garbage=4,deflate=True)
pg.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(H/'framework_with_counterexample.png')
print('Original architecture SHA256:',hashlib.sha256((H/'framework.pdf').read_bytes()).hexdigest())
