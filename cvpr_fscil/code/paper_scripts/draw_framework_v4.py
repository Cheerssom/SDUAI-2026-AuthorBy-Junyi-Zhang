"""Editable draw.io main figure, v4.
Visual style follows the two-panel reference figure supplied by the authors
(gray encoders with snowflakes, orange evidence blocks, green prototype bank,
blue explanation panel, red equation tags). All content describes this paper.
All icons are drawn natively in draw.io; no third-party icon files are used.
Output: framework_v4.drawio (File > Export as > PDF, Crop).
"""
from pathlib import Path
import base64, math, sys, xml.etree.ElementTree as E

H = Path(__file__).resolve().parent
A = H / 'figure_assets_v3'
W, HGT = 1140, 362
NO_IMAGES = '--no-images' in sys.argv
OUT = H / ('framework_v4_preview.drawio' if NO_IMAGES else 'framework_v4.drawio')

mx = E.Element('mxfile', host='app.diagrams.net', version='29.0.0')
dg = E.SubElement(mx, 'diagram', id='framework_v4', name='Framework')
gm = E.SubElement(dg, 'mxGraphModel', dx=str(W), dy=str(HGT), grid='0', page='1', pageScale='1',
                  pageWidth=str(W), pageHeight=str(HGT), background='#ffffff', math='1', shadow='0')
root = E.SubElement(gm, 'root')
E.SubElement(root, 'mxCell', id='0'); E.SubElement(root, 'mxCell', id='1', parent='0')

INK, ORANGE, RED = '#1E1E1E', '#D2691E', '#C00000'
GRAY = '#D9D9D9'
BLOCK, BLOCK_S = '#F4B183', '#C55A11'
NEW, NEW_S = '#BDD7EE', '#2E75B6'
PROTO = '#D5E8C4'
PANEL, PANEL_S = '#DEEAF6', '#9DC3E6'
FONT = f'fontFamily=Palatino Linotype;fontColor={INK};html=1;whiteSpace=wrap;'

n = 0
def nid(p):
    global n; n += 1; return f'{p}{n}'

def cell(value, style, x, y, w, h):
    c = E.SubElement(root, 'mxCell', id=nid('v'), value=value, style=style, vertex='1', parent='1')
    E.SubElement(c, 'mxGeometry', x=f'{x:.1f}', y=f'{y:.1f}', width=f'{w:.1f}', height=f'{h:.1f}', attrib={'as': 'geometry'})

def box(x, y, w, h, value='', fill=GRAY, stroke='none', sz=18, extra='', sw=1.2):
    cell(value, FONT + f'rounded=1;arcSize=24;fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontSize={sz};' + extra, x, y, w, h)

def module(x, y, w, h, name, sz=18):
    box(x, y, w, h, f'<b><i>{name}</i></b>', GRAY, 'none', sz)

def text(x, y, w, h, value, sz=17, align='center', color=INK, extra=''):
    cell(value, FONT + f'text;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;fontSize={sz};fontColor={color};' + extra, x, y, w, h)

def eqtag(x, y, s, w=70):
    text(x, y, w, 20, f'<i>({s})</i>', 15, 'center', RED)

def edge(pts, color=ORANGE, sw=1.6, arrow='classic', dashed=False, curved=False, extra=''):
    st = f'html=1;edgeStyle=none;rounded=0;endArrow={arrow};endFill=1;endSize=6;strokeColor={color};strokeWidth={sw};'
    st += ('dashed=1;dashPattern=6 4;' if dashed else '') + ('curved=1;' if curved else '') + extra
    c = E.SubElement(root, 'mxCell', id=nid('e'), style=st, edge='1', parent='1')
    g = E.SubElement(c, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
    E.SubElement(g, 'mxPoint', x=str(pts[0][0]), y=str(pts[0][1]), attrib={'as': 'sourcePoint'})
    E.SubElement(g, 'mxPoint', x=str(pts[-1][0]), y=str(pts[-1][1]), attrib={'as': 'targetPoint'})
    if len(pts) > 2:
        a = E.SubElement(g, 'Array', attrib={'as': 'points'})
        for x, y in pts[1:-1]:
            E.SubElement(a, 'mxPoint', x=str(x), y=str(y))

def hollow_arrow(pts, width=10):
    """Red outlined block arrow, as in the reference."""
    edge(pts, color=RED, sw=2, arrow='block',
         extra=f'shape=flexArrow;width={width};endWidth={width + 12};endSize=6;fillColor=#FFFFFF;')

def gradient_arrow(pts):
    edge(pts, color='none', sw=1, arrow='block', curved=True,
         extra='shape=flexArrow;width=16;endWidth=30;endSize=9;fillColor=#2F5597;gradientColor=#FFFFFF;gradientDirection=east;opacity=85;')

def image(name, x, y, s):
    if NO_IMAGES:
        box(x, y, s, s, '', '#B08A6E', '#555555', extra='ellipse;' if 'circ' in name else '')
        return
    data = base64.b64encode((A / name).read_bytes()).decode()
    cell('', 'shape=image;imageAspect=1;aspect=fixed;html=1;image=data:image/png,' + data + ';', x, y, s, s)

def snowflake(cx, cy, r=8, color='#5B9BD5'):
    for k in range(3):
        a = k * math.pi / 3; dx, dy = r * math.cos(a), r * math.sin(a)
        edge([(cx - dx, cy - dy), (cx + dx, cy + dy)], color=color, sw=2, arrow='none')
        for sgn in (-1, 1):
            px, py = cx + sgn * dx * .6, cy + sgn * dy * .6
            for side in (-1, 1):
                b = a + (0 if sgn > 0 else math.pi) + side * 2.4
                edge([(px, py), (px + 3.6 * math.cos(b), py + 3.6 * math.sin(b))], color=color, sw=1.3, arrow='none')

def blocks(x, y, nbase, nnew=0, dots=True):
    xs = x
    for _ in range(nbase):
        box(xs, y, 14, 32, '', BLOCK, BLOCK_S, extra='arcSize=30;'); xs += 20
    if dots:
        text(xs - 2, y, 24, 32, '<b>··</b>', 18); xs += 22
    for _ in range(nnew):
        box(xs, y, 14, 32, '', NEW, NEW_S, sw=1.6, extra='arcSize=30;'); xs += 20
    return xs

def dot(x, y, color=ORANGE):
    box(x - 4.5, y - 4.5, 9, 9, '', color, '#FFFFFF', extra='ellipse;', sw=1)

# ====================================================================== panel (a)
MX, MW = 142, 150            # module column
BX = 318                     # block column
BUS = 470                    # shared query-feature bus
QX = 486                     # score labels
rows = {'L': 42, 'T': 120, 'V': 198, 'P': 292}

# Row L: retained base classifier
text(14, rows['L'] - 22, 116, 44, 'Base features<br>+ labels', 16)
edge([(126, rows['L']), (MX - 2, rows['L'])])
module(MX, rows['L'] - 17, MW, 34, 'Base Ridge Head')
edge([(MX + MW + 2, rows['L']), (BX - 2, rows['L'])])
e = blocks(BX, rows['L'] - 16, 3)
text(BX - 10, rows['L'] + 15, 110, 22, '<i>Retained</i> \\(W_{\\mathcal B}, b_{\\mathcal B}\\)', 14)
edge([(e, rows['L']), (BUS - 5, rows['L'])])

# Row T: text anchors, novel anchor appended
box(16, rows['T'] - 22, 108, 44, '<i>A photo of<br>a {class}.</i>', '#FFFFFF', '#7F7F7F', 15)
edge([(126, rows['T']), (MX - 2, rows['T'])])
module(MX, rows['T'] - 17, MW, 34, 'Textual Encoder'); snowflake(MX + MW - 12, rows['T'] - 9)
edge([(MX + MW + 2, rows['T']), (BX - 2, rows['T'])])
e = blocks(BX, rows['T'] - 16, 3, 1)
text(BX - 10, rows['T'] + 15, 110, 22, '\\(t_b\\), <span style="color:#2E75B6">\\(t_n\\)</span>', 14)
edge([(e, rows['T']), (BUS - 5, rows['T'])])

# Row V: query
image('query_pet_sq.png', 46, rows['V'] - 27, 54)
text(30, rows['V'] + 26, 86, 20, '<b><i>Query</i></b> \\(x\\)', 15)
edge([(102, rows['V']), (MX - 2, rows['V'])])
module(MX, rows['V'] - 17, MW, 34, 'Visual Encoder'); snowflake(MX + MW - 12, rows['V'] - 9)
edge([(MX + MW + 2, rows['V']), (BX - 2, rows['V'])])
box(BX, rows['V'] - 16, 72, 32, '\\(v=f(x)\\)', '#E2F0D9', '#70AD47', 15)
edge([(BX + 72, rows['V']), (BUS, rows['V'])], arrow='none')

# Row P: few-shot prototypes, novel prototype appended
for k, f in enumerate(['support_pet_0_sq.png', 'support_pet_1_sq.png', 'support_pet_2_sq.png']):
    image(f, 18 + 26 * k, rows['P'] - 26 + 4 * k - 4, 44)
text(8, rows['P'] + 26, 124, 20, '<i>Base / \\(k\\)-shot images</i>', 14)
edge([(126, rows['P']), (MX - 2, rows['P'])])
module(MX, rows['P'] - 17, MW, 34, 'Visual Encoder'); snowflake(MX + MW - 12, rows['P'] - 9)
edge([(MX + MW + 2, rows['P']), (BX - 2, rows['P'])])
text(MX + MW - 2, rows['P'] - 22, 28, 16, '<i>Mean</i>', 12)
text(BX, rows['P'] - 50, 140, 20, '<b><i>Prototypes</i></b>', 15)
box(BX, rows['P'] - 28, 140, 56, '', PROTO, 'none')
for k, f in enumerate(['input_0_circ.png', 'input_1_circ.png', 'input_2_circ.png']):
    image(f, BX + 6 + 32 * k, rows['P'] - 20, 30)
image('support_pet_2_circ_new.png', BX + 104, rows['P'] - 21, 32)
text(BX + 70, rows['P'] + 10, 40, 16, '<b>··</b>', 14)
edge([(BX + 140, rows['P']), (BUS - 5, rows['P'])])

# shared encoders (as in the reference)
edge([(MX + 60, rows['V'] + 18), (MX + 60, rows['P'] - 18)], color=INK, sw=1.4, arrow='none', dashed=True)
text(MX + 66, rows['V'] + 30, 70, 22, '<b><i>Shared</i></b>', 15, 'left')

# shared query-feature bus: v scores every evidence source
edge([(BUS, rows['L']), (BUS, rows['P'])], arrow='none', sw=1.8)
for r in ('L', 'T', 'P'):
    dot(BUS, rows[r])
    edge([(BUS + 4, rows[r]), (QX, rows[r])])
text(QX, rows['L'] - 15, 82, 30, '\\(q_L(x,b)\\)', 16)
text(QX, rows['T'] - 15, 82, 30, '\\(q_T(x,c)\\)', 16)
text(QX, rows['P'] - 15, 82, 30, '\\(q_P(x,c)\\)', 16)

# fixed base references -> reference-anchored alignment -> closed-form admission
RX, RW = 588, 132
text(RX, 6, RW, 20, '<b><i>Fixed base references</i></b>', 14)
box(RX + 6, 28, RW - 12, 40, '\\(\\mathcal R_e:\\ T_{\\mathcal B},\\,W_{\\mathcal B},\\,P_{\\mathcal B}\\)', '#E2F0D9', 'none', 14)
snowflake(RX + RW - 10, 30, r=7)
edge([(RX + RW / 2, 69), (RX + RW / 2, 104)])
for r in ('L', 'T', 'P'):
    edge([(QX + 82, rows[r]), (574, rows[r])], arrow='none')
edge([(574, rows['L']), (574, rows['P'])], arrow='none')
edge([(574, 150), (RX - 2, 150)])
box(RX, 106, RW, 92, '<b><i>Reference-anchored<br>alignment</i></b><br>'
    '\\(\\hat q_e=(q_e-\\tilde\\mu_e)/\\tilde s_e\\)', GRAY, 'none', 14)
eqtag(RX + 31, 199, 'Eq. 4')
edge([(RX + RW / 2, 220), (RX + RW / 2, 238)])
box(RX, 240, RW, 76, '<b><i>Closed-form<br>admission</i></b><br>\\(M(x)+d>0\\)', GRAY, 'none', 14)
eqtag(RX + 31, 317, 'Eq. 9')

text(10, 336, 712, 26, '<b><i>(a) Asymmetric Evidence Alignment</i></b>', 19, 'center', RED)

# ====================================================================== separator
edge([(733, 10), (733, 330)], color='#BFBFBF', sw=1.6, arrow='none')

# ====================================================================== panel (b)
# acceptance bar chart (real result: Supplement V.3)
text(748, 2, 170, 20, '<b><i>Retention-test acceptance</i></b>', 14)
edge([(760, 112), (902, 112)], color=INK, sw=1.4, arrow='none')
box(782, 108.5, 34, 3.5, '', '#C55A11', 'none', extra='rounded=0;')
box(846, 112 - 68, 34, 68, '', '#C55A11', 'none', extra='rounded=0;gradientColor=#FBE5D6;gradientDirection=north;')
text(770, 86, 58, 20, '<b>2.8%</b>', 14); text(834, 24, 58, 20, '<b>90.6%</b>', 14)
text(776, 113, 46, 20, '\\(G>0\\)', 14); text(840, 113, 46, 20, '\\(G=0\\)', 14)

# wealth curves (schematic)
ox, oy, aw, ah = 1010, 104, 112, 74
edge([(ox, oy), (ox + aw, oy)], color=INK, sw=1.4)
edge([(ox, oy), (ox, oy - ah)], color=INK, sw=1.4)
edge([(ox, oy - 50), (ox + aw, oy - 50)], color=RED, sw=1.2, arrow='none', dashed=True)
g0 = [(ox, oy)] + [(ox + t, oy - 0.0058 * t * t - 0.12 * t) for t in range(15, 106, 18)]
gp = [(ox, oy)] + [(ox + t, oy - 0.26 * t) for t in range(20, 111, 30)]
edge(g0, color='#1F4E79', sw=2.4, arrow='none', curved=True)
edge(gp, color='#A6A6A6', sw=2.4, arrow='none', curved=True)
text(ox - 4, 8, 70, 20, '\\(\\log E_\\lambda\\)', 13, 'left')
text(ox + aw - 62, oy + 2, 66, 18, '<i>calib. size</i>', 12, 'right')
gradient_arrow([(1000, 72), (956, 46), (912, 60)])
text(936, 106, 78, 20, '<b><i>Betting test</i></b>', 14)
eqtag(940, 124, 'Eq. 11')

# explanation panel: paired outcomes
hollow_arrow([(864, 166), (864, 142)])
text(738, 143, 106, 20, '<b><i>Zero recovery</i></b>', 14, 'right', RED)
box(742, 168, 250, 158, '', PANEL, PANEL_S, extra='arcSize=8;')
text(748, 170, 238, 22, '<b><i>Paired outcomes on base queries</i></b>', 15)
image('input_0_circ.png', 752, 196, 22)
text(778, 195, 210, 24, 'correct <i>before</i>, wrong <i>after</i>:', 14, 'left')
box(752, 222, 230, 34, 'lost prediction \\(D\\), the only cost<br>counted against retention', '#FFFFFF', '#8C8C8C', 13, extra='align=left;spacingLeft=6;rounded=0;')
image('input_2_circ.png', 752, 262, 22)
text(778, 261, 210, 24, 'wrong <i>before</i>, correct <i>after</i>:', 14, 'left')
box(752, 288, 230, 34, 'recovered \\(G=0\\): impossible<br>with invariant scores', '#FFFFFF', RED, 13, sw=1.6,
    extra='align=left;spacingLeft=6;rounded=0;')
eqtag(922, 303, 'Cor. 1', 60)
hollow_arrow([(996, 248), (1022, 248)])
module(1026, 196, 108, 46, 'Certified \\(\\hat d\\)', 16)
module(1026, 256, 108, 58, 'Max. test<br>growth rate', 15)
eqtag(1046, 314, 'Prop. 3')

text(736, 336, 404, 26, '<b><i>(b) Certifiability by Design</i></b>', 19, 'center', RED)

E.ElementTree(mx).write(OUT, encoding='utf-8', xml_declaration=True)
print('wrote', OUT.name, 'cells', len(root))
