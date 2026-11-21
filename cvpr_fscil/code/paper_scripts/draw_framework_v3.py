"""Hand-drawn (sketch) draw.io main figure, v3.
Visual style follows the LatentEvolve framework figure supplied by the authors;
all content (modules, symbols, results) describes this paper.
Output: framework_v3.drawio (open in draw.io, File > Export as > PDF, Crop).
"""
from pathlib import Path
import base64, math, sys, xml.etree.ElementTree as E

H = Path(__file__).resolve().parent
W, HGT = 1320, 540
NO_IMAGES = '--no-images' in sys.argv          # lighter file for browser preview
OUT = H / ('framework_v3_preview.drawio' if NO_IMAGES else 'framework_v3.drawio')

mx = E.Element('mxfile', host='app.diagrams.net', version='29.0.0')
dg = E.SubElement(mx, 'diagram', id='framework_v3', name='Framework')
gm = E.SubElement(dg, 'mxGraphModel', dx=str(W), dy=str(HGT), grid='0', page='1', pageScale='1',
                  pageWidth=str(W), pageHeight=str(HGT), background='#ffffff', math='1', shadow='0')
root = E.SubElement(gm, 'root')
E.SubElement(root, 'mxCell', id='0'); E.SubElement(root, 'mxCell', id='1', parent='0')

# ---------------------------------------------------------------- palette
INK = '#262626'
SK = 'sketch=1;curveFitting=1;jiggle=2;fillStyle=solid;'
FONT = 'fontFamily=Times New Roman;fontColor=' + INK + ';html=1;whiteSpace=wrap;'
WARM_BG, WARM_TITLE = '#FDF3DC', '#B8651B'
COOL_BG, COOL_TITLE = '#E4EEF8', '#1F5FAD'
TEAL, TEAL_S = '#8FC1B5', '#2F5F56'
TOKEN, TOKEN_S = '#DCE7F3', '#34506E'
NEW, NEW_S = '#C8E3A6', '#4E7F2A'
CUBE, CUBE_S = '#D7EA8C', '#5F7F1F'
ENC, ENC_S = '#A9D3A0', '#3F6E3A'
POOL, POOL_S = '#A9CBD9', '#2E5A6B'
PINK, PINK_S = '#F5CDD2', '#A8394B'
PURPLE, PURPLE_S = '#DCCDEE', '#5E4691'
GRAYF, GRAY_S = '#E3E3E3', '#595959'
RED = '#C0392B'
GREEN = '#3E8E3E'

n = 0
def nid(p):
    global n; n += 1; return f'{p}{n}'

def cell(value, style, x, y, w, h, name=None):
    c = E.SubElement(root, 'mxCell', id=name or nid('v'), value=value, style=style, vertex='1', parent='1')
    E.SubElement(c, 'mxGeometry', x=f'{x:.1f}', y=f'{y:.1f}', width=f'{w:.1f}', height=f'{h:.1f}', attrib={'as': 'geometry'})
    return c

def box(x, y, w, h, fill, stroke, value='', sz=20, extra='', sketch=True, rounded=True, sw=2):
    st = (SK if sketch else '') + FONT + f'fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontSize={sz};'
    st += ('rounded=1;arcSize=12;' if rounded else '') + extra
    return cell(value, st, x, y, w, h)

def text(x, y, w, h, value, sz=20, align='center', color=INK, extra=''):
    st = FONT + f'text;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;fontSize={sz};fontColor={color};' + extra
    return cell(value, st, x, y, w, h)

def edge(pts, color=INK, sw=2, arrow='classic', dashed=False, curved=False, sketch=True, extra=''):
    st = (SK if sketch else '') + f'html=1;edgeStyle=none;rounded=0;endArrow={arrow};endFill=1;endSize=8;strokeColor={color};strokeWidth={sw};'
    st += ('dashed=1;dashPattern=6 5;' if dashed else '') + ('curved=1;' if curved else '') + extra
    c = E.SubElement(root, 'mxCell', id=nid('e'), style=st, edge='1', parent='1')
    g = E.SubElement(c, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
    E.SubElement(g, 'mxPoint', x=str(pts[0][0]), y=str(pts[0][1]), attrib={'as': 'sourcePoint'})
    E.SubElement(g, 'mxPoint', x=str(pts[-1][0]), y=str(pts[-1][1]), attrib={'as': 'targetPoint'})
    if len(pts) > 2:
        a = E.SubElement(g, 'Array', attrib={'as': 'points'})
        for x, y in pts[1:-1]:
            E.SubElement(a, 'mxPoint', x=str(x), y=str(y))
    return c

def block_arrow(pts, fill='#C9C9C9', stroke='#7F7F7F', width=14, curved=False):
    """Thick hand-drawn gray arrow, as in the reference figure."""
    extra = f'shape=flexArrow;width={width};endWidth={width+12};endSize=7;fillColor={fill};'
    return edge(pts, color=stroke, sw=1.5, arrow='block', curved=curved, extra=extra)

def image(path, x, y, s):
    if NO_IMAGES:
        return box(x, y, s, s, '#C9A27E', '#464646', extra='ellipse;', sw=2)
    data = base64.b64encode(path.read_bytes()).decode()
    return cell('', 'shape=image;imageAspect=1;aspect=fixed;html=1;image=data:image/png,' + data + ';', x, y, s, s)

def snowflake(cx, cy, r=17, color='#3C78C8'):
    for k in range(3):
        a = k * math.pi / 3; dx, dy = r * math.cos(a), r * math.sin(a)
        edge([(cx - dx, cy - dy), (cx + dx, cy + dy)], color=color, sw=3, arrow='none', sketch=False)
        for sgn in (-1, 1):
            px, py = cx + sgn * dx * .6, cy + sgn * dy * .6
            for side in (-1, 1):
                b = a + (0 if sgn > 0 else math.pi) + side * 2.4
                edge([(px, py), (px + 7 * math.cos(b), py + 7 * math.sin(b))], color=color, sw=2.2, arrow='none', sketch=False)
    box(cx - 4, cy - 4, 8, 8, '#FFFFFF', color, extra='ellipse;', sketch=False, sw=1.5)

def tokens(x, y, nbase, nnew, size=26, gap=7, dots_after=None):
    """Row of square tokens; base tokens first, then appended novel tokens."""
    xs = x
    for i in range(nbase):
        box(xs, y, size, size, TOKEN, TOKEN_S, sw=1.6, extra='arcSize=18;')
        xs += size + gap
        if dots_after is not None and i == dots_after:
            text(xs - 2, y, 30, size, '⋯', 22); xs += 30
    for _ in range(nnew):
        box(xs, y, size, size, NEW, NEW_S, sw=1.8, extra='arcSize=18;')
        xs += size + gap
    return xs

def cube(x, y, w, h, label, fill=CUBE, stroke=CUBE_S):
    box(x, y, w, h, fill, stroke, extra='shape=cube;size=9;darkOpacity=0.08;darkOpacity2=0.2;', rounded=False, sw=1.8)
    text(x - 20, y + h + 2, w + 40, 24, label, 19)

# ================================================================ regions
box(0, 0, 868, HGT, WARM_BG, '#E9D9B5', rounded=False, sw=1, sketch=False)
box(868, 0, W - 868, HGT, COOL_BG, '#C6D7EA', rounded=False, sw=1, sketch=False)
text(14, 4, 520, 40, '<b><i>Asymmetric Evidence Alignment</i></b>', 31, 'left', WARM_TITLE)
text(884, 4, 430, 40, '<b><i>Certifiability by Design</i></b>', 31, 'left', COOL_TITLE)

# ================================================================ problem banner (retention-constrained expansion)
image(H / 'figure_assets_v3/query_pet_circ.png', 30, 52, 66)
text(14, 118, 100, 22, '<i>query</i> \\(x\\)', 18)
box(124, 50, 720, 66, '#F7F7F7', '#5A5A5A', sw=1.8,
    value='<div style="text-align:left;padding-left:12px;line-height:1.25">'
          '<b>Arrival</b> \\(\\mathcal{N}_t\\)<b>:</b> '
          '<span style="font-family:Courier New">name “Ragdoll” + k = 5 images</span><br>'
          '<b>Requirement:</b> '
          '<span style="font-family:Courier New">admit it, keep </span>\\(A_{\\mathcal B}\\ge r\\,A^{\\mathrm{closed}}_{\\mathcal B,\\mathrm{ref}}\\)</div>',
    sz=19, extra='align=left;')
for i, f in enumerate(['support_pet_0', 'support_pet_1', 'support_pet_2']):
    image(H / f'figure_assets_v3/{f}_circ.png', 712 + i * 40, 58, 50)

# ================================================================ frozen encoder
block_arrow([(63, 140), (63, 160)], fill='#BFE0A8', stroke='#5E8F4A', width=16)
box(18, 166, 112, 200, ENC, ENC_S, '<b>CLIP<br>Encoder<br><span style="font-size:17px">(frozen)</span></b>', 22)
snowflake(38, 172)

# ================================================================ asymmetric evidence (token rows)
text(160, 126, 400, 28, '<i>Asymmetric evidence on a shared feature \\(v=f(x)\\)</i>', 19, 'left')
box(156, 156, 420, 186, TEAL, TEAL_S, sw=2)
rows = [('\\(q_L(x,b)\\)', 0), ('\\(q_T(x,c)\\)', 2), ('\\(q_P(x,c)\\)', 2)]
for k, (lab, nnew) in enumerate(rows):
    y = 172 + k * 52
    text(164, y - 2, 92, 30, lab, 19)
    tokens(262, y, 5, nnew, dots_after=3)
text(262, 310, 190, 24, '<i>base</i> \\(b\\in\\mathcal B\\)', 17, 'left')
text(440, 310, 134, 24, '<i>appended</i> \\(n\\in\\mathcal N_t\\)', 17, 'left', '#3E6F1E')
block_arrow([(132, 250), (154, 250)], width=14)

# ================================================================ fixed base references (cubes)
text(600, 126, 260, 28, '<i>Fixed base references</i> \\(\\mathcal R_e\\)', 19, 'left')
box(596, 156, 250, 186, '#CFE3DC', TEAL_S, sw=2)
snowflake(822, 178, r=14)
cube(618, 190, 48, 46, '\\(T_{\\mathcal B}\\)')
cube(696, 190, 48, 46, '\\(W_{\\mathcal B}\\)')
cube(774, 190, 48, 46, '\\(P_{\\mathcal B}\\)')
text(604, 276, 236, 50, '<i>frozen across arrivals;<br>no new class enters</i>', 17)

# ================================================================ alignment pool (dashed)
text(26, 372, 520, 26, '<i>Reference-anchored alignment, per query</i>', 19, 'left')
box(18, 398, 528, 132, POOL, POOL_S, extra='dashed=1;dashPattern=8 6;', sw=2.2)
box(34, 412, 236, 104, '#F4F9FB', POOL_S, sw=1.6,
    value='query moments \\(\\mu_e(x),\\,s_e(x)\\)<br>over \\(\\mathcal R_e\\); '
          '\\(\\eta_{\\mathcal B}=\\min(1,B/30)\\)', sz=18)
block_arrow([(274, 464), (296, 464)], width=10)
box(302, 412, 230, 104, '#F4F9FB', POOL_S, sw=1.6,
    value='\\(\\hat q_e=\\dfrac{q_e-\\tilde\\mu_e(x)}{\\tilde s_e(x)}\\)', sz=21)
# evidence -> pool, references -> pool
edge([(366, 344), (366, 396)], sw=2.2)
text(372, 350, 150, 24, '<i>align</i>', 18, 'left')
block_arrow([(660, 344), (604, 368), (548, 406)], curved=True, width=12)

# ================================================================ closed-form admission
text(566, 372, 230, 26, '<i>Closed-form admission</i>', 19, 'left')
box(562, 398, 222, 132, GRAYF, GRAY_S, sw=2,
    value='\\(s(x,b)=\\hat q_T+a\\,\\hat q_L\\)<br>\\(s(x,n)=\\hat q_T+u\\,\\hat q_P+d\\)'
          '<br><span style="font-size:16px"><i>ridge: \\(\\mu_L=1/B\\) ⟹ no router</i></span>', sz=18)
edge([(546, 464), (560, 464)], sw=2.2)

# ================================================================ bridge: zero recovery
box(790, 378, 156, 150, PINK, PINK_S, extra='ellipse;', sw=2.6,
    value='<b>Invariance</b><br>⟹ <b>\\(G=0\\)</b><br><span style="font-size:15px"><i>no base error<br>is ever repaired</i></span>', sz=21)
edge([(784, 464), (792, 464)], sw=2.2)

# ================================================================ right: paired outcomes (speech bubble)
box(884, 50, 420, 206, '#F6D9DD', PINK_S, sw=2,
    extra='shape=callout;perimeter=calloutPerimeter;position2=0.08;base=24;size=26;position=0.02;direction=south;flipV=1;')
text(896, 56, 400, 26, '<b>Paired outcomes on base queries</b>', 20)
cx, cy, cw, ch = 1010, 108, 130, 40
text(cx, cy - 24, cw, 22, '\\(C_d=1\\)', 18); text(cx + cw, cy - 24, cw, 22, '\\(C_d=0\\)', 18)
text(cx - 96, cy, 92, ch, '\\(R=1\\)', 18); text(cx - 96, cy + ch, 92, ch, '\\(R=0\\)', 18)
box(cx, cy, cw, ch, '#FFFFFF', '#6B6B6B', '<i>kept</i>', 18, sw=1.5, rounded=False)
box(cx + cw, cy, cw, ch, '#FFFFFF', '#6B6B6B', '<i>lost</i> \\(D\\)', 18, sw=1.5, rounded=False)
box(cx, cy + ch, cw, ch, '#FBE3E1', RED, '<b><i>recovered</i> \\(G=0\\)</b>', 16, sw=2.2, rounded=False,
    extra='dashed=1;dashPattern=5 4;fontColor=' + RED + ';')
box(cx + cw, cy + ch, cw, ch, '#FFFFFF', '#6B6B6B', '<i>error</i>', 18, sw=1.5, rounded=False)
text(898, 200, 396, 46, '\\(\\mathrm{Var}(C_d-rR)\\) <i>minimal at fixed accuracies</i><br>'
     '<span style="font-size:16px">(Corollary 1)</span>', 18)
edge([(902, 388), (912, 310), (936, 262)], color=PINK_S, sw=2.4, curved=True)

# ================================================================ calibration set (cylinder)
box(958, 292, 118, 150, PURPLE, PURPLE_S, extra='shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=14;', rounded=False, sw=2,
    value='<b>Calibration</b><br>base queries<br>\\(\\langle R_i, C_{d,i}\\rangle\\)', sz=17)
edge([(944, 430), (958, 420)], color=PINK_S, sw=2.2)
edge([(1078, 366), (1096, 366)], sw=2.2)
text(1072, 336, 30, 24, '\\(E_\\lambda\\)', 16)

# ================================================================ wealth curves
box(1098, 278, 206, 172, '#FFFFFF', '#6B6B6B', sw=1.6)
ox, oy, aw, ah = 1120, 424, 172, 124          # axes origin / size
edge([(ox, oy), (ox + aw, oy)], sw=1.8, sketch=False)
edge([(ox, oy), (ox, oy - ah)], sw=1.8, sketch=False)
text(ox + aw - 120, oy + 2, 122, 22, '<i>calibration size</i> \\(n\\)', 14, 'right')
text(ox + 4, 280, 80, 22, '\\(\\log E_\\lambda\\)', 15, 'left')
thr = oy - 86
edge([(ox, thr), (ox + aw, thr)], color=RED, sw=1.8, arrow='none', dashed=True, sketch=False)
text(ox + aw - 54, thr - 24, 56, 22, '\\(\\log\\frac{T}{\\delta}\\)', 14, 'right', RED)
g0 = [(ox, oy)] + [(ox + t, oy - 0.0052 * t * t - 0.10 * t) for t in range(18, 155, 27)]
gp = [(ox, oy)] + [(ox + t, oy - 0.16 * t - 0.0002 * t * t) for t in range(20, 171, 30)]
edge(g0, color=GREEN, sw=3, curved=True, arrow='none')
edge(gp, color='#8C8C8C', sw=3, curved=True, arrow='none')
text(ox + 8, oy - 112, 96, 22, '\\(G=0\\): <b>90.6%</b>', 14, 'left', GREEN)
text(ox + 96, oy - 52, 80, 22, '\\(G>0\\): <b>2.8%</b>', 14, 'left', '#6B6B6B')

# ================================================================ certified output (stacked panels)
for k in range(3):
    box(1098 + 7 * k, 476 - 6 * k, 192, 54, '#A9CBC3' if k < 2 else '#CFE6E0', TEAL_S, sw=1.6, extra=f'opacity={45 + 25 * k};')
text(1112, 464, 192, 54, '<b>certified</b> \\(\\hat d\\): \\(A_{\\mathcal B}\\ge rA_{\\mathcal B,\\mathrm{ref}}\\)<br>'
     '<span style="font-size:14px"><i>w.p.</i> \\(\\ge1-\\delta\\) (Prop. 3)</span>', 16)
edge([(1200, 450), (1200, 462)], sw=2.2)

# admission candidates -> calibration (dashed feedback arrow, as in the reference)
edge([(672, 530), (800, 537), (1017, 537), (1017, 444)], sw=2.2, dashed=True, curved=True)
text(1024, 470, 74, 40, '<i>candidate</i><br><i>offsets</i> \\(d\\)', 14, 'left')

E.ElementTree(mx).write(OUT, encoding='utf-8', xml_declaration=True)
print('wrote', OUT.name, 'cells', len(root))
