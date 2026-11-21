from pathlib import Path
import fitz,hashlib
H=Path(__file__).resolve().parent
source=H.parent/'cvpr2027_revisions/20261101_before_evidence_reframe/framework.pdf'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='3f894e83e93e495abf1dfbbad5fd3e54fb13360c705f08577812e6ef22bb48c9'
d=fitz.open(source);p=d[0]
changes={'Gated evidence':'Aligned evidence','+ availability mask':'+ availability mask','Fixed-reference normalization':'Fixed-base reference normalization','Fixed references and gates.':'Query-adaptive moments; fixed banks.','Append rows; retain old scores.':'Append classes; preserve old scores.'}
spans=[s for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if s['text'] in changes]
assert len(spans)==5
for s in spans:p.add_redact_annot(s['bbox'],fill=False)
p.apply_redactions(images=0,graphics=0)
font=fitz.Font(fontfile='C:/Windows/Fonts/times.ttf');p.insert_font(fontname='EditedTimes',fontbuffer=font.buffer)
for s in spans:
 text=changes[s['text']];sz=s['size'];x=747.9-font.text_length(text,fontsize=sz)/2
 p.insert_text((x,s['origin'][1]),text,fontsize=sz,fontname='EditedTimes',color=(.161,.161,.161))
assert p.rect==fitz.Rect(0,0,886.0799560546875,544.0800170898438)
d.save(H/'framework.pdf',garbage=4,deflate=True)
p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(H/'framework_inplace.png')
print('Edited five text lines in the original vector PDF; page size and diagram retained.')
