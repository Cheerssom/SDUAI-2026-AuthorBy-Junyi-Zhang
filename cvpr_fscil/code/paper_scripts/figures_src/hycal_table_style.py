"""Restyle the standard-FSCIL tables in HyCal's layout.

Gray title band, Times text subscripts, bold best / underlined second best per column
within each block (Average / Last / Certified Acc.), heavy rule above each block, and
gray unranked ablation rows. Rewrites, in place, every `lcccc` standard table found in
the given files. Usage: python hycal_table_style.py file.tex [file.tex ...]
"""
import re, sys

NUM = re.compile(r'^(?:\\textbf\{)?(-?\d+\.\d+)\}?(?:\$_\{\\pm(\d+\.\d+)\}\$)?$')
ABL = ('w/o centering', 'fixed moments', 'SGLC', 'w/o covariance source', 'unanchored covariance')  # gray, unranked rows


def parse(cell):
    c = cell.strip()
    m = NUM.match(c)
    return (float(m.group(1)), m.group(1), m.group(2)) if m else None


def fmt(p, rank, gray=False):
    _, v, sd = p
    sub = f'\\textsubscript{{\\textpm{sd}}}' if sd is not None else ''
    if rank == 1:
        return f'\\textbf{{{v}}}' + (f'\\textsubscript{{\\textbf{{\\textpm{sd}}}}}' if sd else '')
    if rank == 2:
        return f'\\underline{{{v}{sub}}}'
    return v + sub


def ranks(vals):
    u = sorted({v for v in vals if v is not None}, reverse=True)
    return [None if v is None else (1 if v == u[0] else 2 if len(u) > 1 and v == u[1] else 0) for v in vals]


def restyle(tab):
    lines = tab.split('\n')
    out, block = [], []

    def flush():
        rows = [[c for c in l.rstrip().rstrip('\\').split('&')] for l in block]
        abl = [any(a in r[0] for a in ABL) for r in rows]
        cols = len(rows[0]) if rows else 0
        R = [[0] * cols for _ in rows]
        for j in range(1, cols):
            ps = [None if a else parse(r[j]) for r, a in zip(rows, abl)]
            rk = ranks([p[0] if p else None for p in ps])
            for i in range(len(rows)):
                R[i][j] = rk[i]
        for i, r in enumerate(rows):
            cells = [r[0].strip()]
            for j in range(1, cols):
                p = parse(r[j])
                cells.append(r[j].strip() if p is None else fmt(p, 0 if abl[i] else R[i][j]))
            line = ' & '.join(cells) + ' \\\\'
            if abl[i]:
                line = ' & '.join('\\textcolor{gray}{' + c + '}' for c in cells) + ' \\\\'
            out.append(line)
        block.clear()

    for l in lines:
        s = l.strip()
        if s.startswith('\\multicolumn{5}{c}{\\textbf{'):
            out.append('\\rowcolor{gray!15}' + s)
        elif s.startswith('\\hdashline\\noalign'):
            out.append('\\midrule')
        elif s.startswith(' & CIFAR-100') or s.startswith('& CIFAR-100'):
            out.append(re.sub(r'& (Average|平均) \\\\', r'& \\textbf{\1} \\\\', l))
        elif s.startswith('\\multicolumn{5}{@{}l}{\\textbf{'):
            if block: flush()
            if out and out[-1].strip() == '\\midrule':
                out[-1] = '\\midrule[\\heavyrulewidth]'
            out.append(l)
        elif s.startswith('\\midrule') or s.startswith('\\bottomrule'):
            if block: flush()
            out.append(l)
        elif block is not None and out and any(o.startswith('\\multicolumn{5}{@{}l}') for o in out[-40:]) and '&' in s and not s.startswith('CLIP zero-shot'):
            block.append(l)
        else:
            if block: flush()
            out.append(l)
    if block: flush()
    return '\n'.join(out)


def main(files):
    for f in files:
        restyle_file(f)


TAB = re.compile(r'\\begin\{tabular\}\{@\{\}lcccc@\{\}\}.*?\\end\{tabular\}', re.S)
def restyle_file(f):
    s = open(f, encoding='utf-8').read()
    n = [0]

    def sub(m):
        if 'Average Acc.' not in m.group(0) and '平均准确率' not in m.group(0):
            return m.group(0)
        n[0] += 1
        return restyle(m.group(0))
    s = TAB.sub(sub, s)
    open(f, 'w', encoding='utf-8').write(s)
    print(f, 'restyled', n[0])


if __name__ == '__main__':
    main(sys.argv[1:])
