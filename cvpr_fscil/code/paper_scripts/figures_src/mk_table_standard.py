import json,glob,statistics as st,sys,os
# usage: python mk_table_standard.py <encoder> [main|abl|desc]
# READOUTS=<dir>[:<dir>...] lists readout roots, each containing <ds>/<enc>/<track>/seedN/session*.json
ROOTS=os.environ.get('READOUTS','readouts').split(':')
# CERT_READOUTS: roots for the certified track (deployed predictor = four-source AnchorCE closed head); default ROOTS
CROOTS=os.environ['CERT_READOUTS'].split(':') if os.environ.get('CERT_READOUTS') else ROOTS
# SGLC (CVPR 2026) trains prompts and an adapter; reported Aavg (its Table 1, ViT-B/16 CLIP), gray upper reference
SGLC={'cifar100':84.57,'mini':95.12,'cub200':80.18}
S={'cifar100':9,'mini':9,'cub200':11};DS=list(S)
enc=sys.argv[1];kind=sys.argv[2] if len(sys.argv)>2 else 'main';encname={'b16':'ViT-B/16','l14':'ViT-L/14','b32':'ViT-B/32'}[enc]
M=[('SimpleCIL_transfer','SimpleCIL~\\cite{zhou2024simplecil}'),('TEEN_transfer','TEEN~\\cite{wang2023teen}'),('FeCAM_transfer','FeCAM~\\cite{goswami2023fecam}'),('RanPAC_frozen_transfer','RanPAC~\\cite{mcdonnell2023ranpac}'),('GDA_transfer','GDA~\\cite{wang2024gda}'),('KLDA_transfer','KLDA~\\cite{momeni2025klda}'),('HyCal_transfer','HyCal~\\cite{lee2026hycal}'),('BiMC_transfer','BiMC~\\cite{chen2025bimc}'),('dense','Dense')]
if kind=='desc': M=[m for m in M if m[0] not in ('GDA_transfer','KLDA_transfer')]  # class-name text only
if kind=='abl': M+=[('full','\\textsc{AnchorCE} w/o covariance source'),('full_gda_rawg','\\textsc{AnchorCE} unanchored covariance'),('full_gda_none','\\textsc{AnchorCE} w/o centering'),('full_gda_fixed','\\textsc{AnchorCE} fixed moments')]
M+=[('full_gda','\\textbf{\\textsc{AnchorCE} (Ours)}')]
def rows(ds,track,seed):
 return [r for root in (CROOTS if track=='certified' else ROOTS) for f in sorted(glob.glob(f'{root}/{ds}/{enc}/{track}/seed{seed}/session*.json')) for r in json.load(open(f))['rows']]
have={r['method'] for r in rows('mini','standard',0)}
M=[(m,l) for m,l in M if m in have]  # drop methods without results
o={}
for ds in DS:
 stt=[rows(ds,'standard',s) for s in range(3)];ce=[rows(ds,'certified',s) for s in range(3)]
 for m in [x for x,_ in M]+['ZeroShot_CLIP']:
  acc=[[100*next(r['unconstrained_top1'] for r in x if r['method']==m and r['session']==t) for t in range(S[ds])] for x in stt]
  fin=[next((r for r in x if r['method']==m and r['session']==S[ds]-1),{}).get('common') for x in ce]
  fin=[f for f in fin if f is not None]
  o[(ds,m)]=dict(Avg=[sum(a)/len(a) for a in acc],Last=[a[-1] for a in acc],Cert=[100*f['novel'] for f in fin],fb=bool(fin) and all(f['offset'] is None for f in fin),nocert=not fin)
def cell(v,fb=False):
 if fb:return r'0.00$^\dagger$'  # fallback on every split: deployed predictor kept, certified accuracy 0
 return f"{st.mean(v):.2f}$_{{\\pm{st.stdev(v):.2f}}}$"
L=["\\begin{tabular}{@{}lcccc@{}}","\\toprule",
   "\\multicolumn{5}{c}{\\textbf{Standard FSCIL benchmarks (CLIP "+encname+")}} \\\\","\\hdashline\\noalign{\\vskip 2pt}",
   " & CIFAR-100 & miniImageNet & CUB-200 & Average \\\\","\\midrule"]
z=[st.mean(o[(ds,'ZeroShot_CLIP')]['Last']) for ds in DS]
L.append("CLIP zero-shot~\\cite{radford2021clip} & "+" & ".join(f"{v:.2f}" for v in z)+f" & {st.mean(z):.2f} \\\\")
for key,name in [('Avg','Average Acc.'),('Last','Last Acc.'),('Cert','Certified Acc.')]:
 L+=["\\midrule","\\multicolumn{5}{@{}l}{\\textbf{"+name+"}} \\\\"]
 if key=='Avg' and enc=='b16' and kind!='desc':L+=["SGLC~\\cite{yan2026semanticprompt} (trained) & "+" & ".join(f"{SGLC[ds]:.2f}" for ds in DS)+f" & {st.mean(SGLC.values()):.2f} \\\\","\\hdashline"]
 for m,lab in M:
  if key=='Cert' and any(o[(ds,m)]['nocert'] for ds in DS):continue
  fbs=[key=='Cert' and o[(ds,m)]['fb'] for ds in DS]
  cells=[cell(o[(ds,m)][key],fbs[i]) for i,ds in enumerate(DS)]
  av='0.00' if all(fbs) else f"{st.mean(st.mean(o[(ds,m)][key]) for ds in DS):.2f}"
  L.append(lab+" & "+" & ".join(cells)+" & "+av+" \\\\")
L+=["\\bottomrule","\\end{tabular}"]
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));from hycal_table_style import restyle
print(restyle("\n".join(L)))
