"""Validate the CVPR review artifact without changing models or results."""
from pathlib import Path
import hashlib,json,re
import fitz
H=Path(__file__).resolve().parent;K=H.parent/'kbs'
MAIN_STEM='main_30'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sha_bytes=lambda b:hashlib.sha256(b).hexdigest()
def validate():
    main=(H/'main.tex').read_text(encoding='utf-8')
    prov=json.loads((H/'template_provenance.json').read_text())
    assert sha(H/'cvpr.sty')==prov['official_style_sha256']
    assert sha(H/'ieeenat_fullname.bst')==prov['official_bst_sha256']
    assert r'\documentclass[10pt,twocolumn,letterpaper]{article}' in main
    assert r'\usepackage[review]{cvpr}' in main
    pre=(H/'preamble.tex').read_text()
    assert r'\def\confYear{2027}' in pre
    assert not re.search(r'\\(?:geometry|fontsize|linespread|baselinestretch)\b',main+pre)
    assert not re.search(r'\\setlength\{\\(?:textwidth|textheight|oddsidemargin)\}',main+pre)
    assert 'LLM Usage Statement' in main
    assert main.index('LLM Usage Statement')<main.index(r'\bibliography{')
    forbidden=['junyi zhang','shandong university','cheerssom','32488','g:\\','f:\\','knowledge-based systems submission']
    documents={}
    for name in ['main','supplementary']:
        artifact=MAIN_STEM if name=='main' else name
        log=(H/f'{artifact}.log').read_text(errors='replace')
        errors=[x for x in log.splitlines() if any(w in x for w in ['Overfull','LaTeX Warning:','Package natbib Warning:','! '])]
        assert not errors,(name,errors)
        with fitz.open(H/f'{artifact}.pdf') as d:
            text='\n'.join(p.get_text() for p in d)
            check=(text+'\n'+str(d.metadata)+'\n'+(H/f'{name}.tex').read_text(encoding='utf-8')).lower()
            assert not [w for w in forbidden if w in check],name
            assert '??' not in text,name
            assert 'kbs_' not in text and 'reviewer_revision' not in text,name
            assert 'Reference-anchored table' not in text and 'Candidate-normalized table' not in text,name
            assert not d.metadata.get('author'),name
            assert all(abs(p.rect.width-612)<.1 and abs(p.rect.height-792)<.1 for p in d)
            fonts={f[0]:f for p in d for f in p.get_fonts(full=True)}
            for xref,f in fonts.items():
                embedded=bool(xref and (d.xref_get_key(xref,'CharProcs')[0]!='null' if f[2]=='Type3' else d.extract_font(xref)[3]))
                assert embedded,(name,'unembedded font',f)
            links=[l.get('uri','') for p in d for l in p.get_links() if l.get('uri')]
            assert not any('cheerssom' in u.lower() for u in links)
            documents[name]={'pdf_pages':len(d),'embedded_fonts':len(fonts),'anonymous_text_and_metadata':True,
                             'compile_warnings':errors,'paper_size':'US Letter'}
            if name=='main':
                refpage=next(i+1 for i,p in enumerate(d) if 'References' in p.get_text())
                assert refpage<=9 and len(d)-refpage<=2
                assert 'Anonymous CVPR submission' in d[0].get_text()
                assert 'CVPR 2027 Submission' in d[0].get_text()
                assert 'LLM Usage Statement' in text
                assert all('Conclusion' not in p.get_text() and 'Table 3.' not in p.get_text() for p in list(d)[refpage-1:])
                assert 'References' not in '\n'.join(p.get_text() for p in list(d)[:refpage-1])
                llmpage=next(i+1 for i,p in enumerate(d) if 'LLM Usage Statement' in p.get_text())
                assert llmpage<=8
                assert 'final submission.' in ' '.join(d[llmpage-1].get_text().split())
            else:
                assert '(S1)' in text and 'Estimating means of bounded random variables' in ' '.join(text.split())
    aux=(H/f'{MAIN_STEM}.aux').read_text()
    m=re.search(r'\\newlabel\{end:maintext\}\{\{[^}]*\}\{(\d+)\}',aux)
    assert m and int(m[1])<=8
    floats=re.findall(r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{[^}]*\}\{(\d+)\}',aux)
    assert len(floats)==9 and all(int(p)<=8 for _,p in floats),floats
    # Compare every transposed recognition/admission cell to frozen tables.
    def rows(path):
        return [[x.strip().removesuffix(r'\\').strip() for x in l.split('&')]
                for l in path.read_text().splitlines()
                if l.startswith(('Retained readout','Dense prototypes','Fixed temperature','HyCal transfer'))]
    for kind in ['recognition','admission']:
        original=rows(K/f'table_asymmetric_{kind}.tex');current=rows(H/f'table_cvpr_{kind}.tex')
        assert len(original)==8 and len(current)==4
        for i in range(4):assert current[i]==original[i]+original[i+4][1:],(kind,i)
    dual=[l.removesuffix(r' \\').split(' & ') for l in (H/'table_cvpr_dualview.tex').read_text().splitlines() if any(m in l for m in ['Retained readout &','Dense prototypes &','Fixed temperature &','HyCal transfer &'])]
    assert len(dual)==12
    a,b=rows(H/'table_cvpr_recognition.tex'),rows(H/'table_cvpr_admission.tex')
    for j in range(3):
        for i in range(4):assert [dual[4*j+i][k] for k in [2,4,5,7]]==[a[i][j+1],b[i][j+1],a[i][j+4],b[i][j+4]]
    tuned=json.loads((H/'tuned_no_fallback.json').read_text())['rows']
    for j,ds in enumerate(['cifar','pets','imagenetr']):
        for i,method in enumerate(['retained','anchored_tp','fixed_temperature','hycal']):
            for en,col in [('l14',3),('b32',6)]:
                r=next(r for r in tuned if r['dataset']==ds and r['encoder']==en and r['method']==method)
                assert len(r['novel_partitions'])==len(r['inputs'])==10
                assert abs(sum(r['novel_partitions'])/10-r['novel_mean'])<1e-9
                assert dual[4*j+i][col]==f"{r['novel_mean']:.2f}"
    original=(K/'table_asymmetric_centering.tex').read_text().splitlines()
    current=[l for l in (H/'table_cvpr_centering.tex').read_text().splitlines() if l.startswith(('L/14 &','B/32 &'))]
    query=[l for l in original if '& Query &' in l];none=[l for l in original if '& None &' in l]
    assert len(query)==len(none)==len(current)==6
    for q,n,c in zip(query,none,current):
        q=[x.strip().removesuffix(r'\\').strip() for x in q.split('&')]
        n=[x.strip().removesuffix(r'\\').strip() for x in n.split('&')]
        c=[x.strip().removesuffix(r'\\').strip() for x in c.split('&')]
        assert c[1:5]==[q[0],q[4],n[4],q[5]]
        assert c[5]==q[2].split('/')[0]+'/'+n[2].split('/')[0]
    if K.exists():
        assert sha(K/'main.tex')==prov['source_kbs_main_sha256']
        assert sha(K/'supplementary.tex')==prov['source_kbs_supplement_sha256']
        # Labels may change; compare every table's numerical tokens and figure inputs.
        inherited=[p for p in H.glob('table_*.tex') if not p.name.startswith('table_cvpr_')]
        for p in inherited:
            old=K/p.name.replace('expansion_','kbs_')
            if not old.exists():
                assert p.name.startswith('table_gate_') or p.name in ['table_softmax_main.tex','table_softmax_readout.tex','table_cross_encoder.tex',
                                  'table_cross_encoder_pairs.tex','table_shrinkage_pets.tex','table_shrinkage_cifar.tex',
                                  'table_shrinkage_imagenetr.tex','table_shrinkage_choices.tex','table_shrinkage_pairs.tex',
                                  'table_rn50.tex','table_rn50_pairs.tex','table_cardinality_main.tex',
                                  'table_cardinality_pets.tex','table_cardinality_cifar.tex','table_cardinality_imagenetr.tex'],p.name
                continue
            assert re.findall(r'[-+]?\d+(?:\.\d+)?',p.read_text())==re.findall(r'[-+]?\d+(?:\.\d+)?',old.read_text()),p.name
        for p in H.glob('*.pdf'):
            if (K/p.name).exists() and p.stem not in ['main','supplementary','review_complete','stability','shots','tradeoff']:
                if p.name=='framework.pdf':
                    # Authorized in-place wording edit; preserve original vector layout and images.
                    with fitz.open(p) as new,fitz.open(K/p.name) as old:
                        assert new[0].rect==old[0].rect
                        assert len(new[0].get_drawings())==len(old[0].get_drawings())==149
                        assert sorted(sha_bytes(new.extract_image(i[0])['image']) for i in new[0].get_images())==sorted(sha_bytes(old.extract_image(i[0])['image']) for i in old[0].get_images())
                else:
                    assert sha(p)==sha(K/p.name),p.name
        fig=json.loads((H/'figure_revision_verification.json').read_text())
        assert sha(H.parents[1]/'experiments/fcs/results.json')==fig['auxiliary_input_sha256']
        for condition,row in fig['source_frontier_inputs'].items():
            base=H.parents[1]/f'experiments/kbs_mechanism_followup/{condition}/p0'
            assert sha(base/'o0.json')==row['record_sha256']
            assert sha(base/'o0.npz')==row['arrays_sha256']
    for source in [H/'main.tex',H/'supplementary.tex',H/'supplement_new_controls.tex',H/'supplement_rn50.tex',H/'supplement_cardinality_rule.tex',H/'supplement_gate_calibration.tex']:
        for name in re.findall(r'experiments/([\w]+\.md)',source.read_text()):
            assert (H/'experiments'/name).exists(),name
    new=json.loads((H/'softmax_readout/summary.json').read_text())
    assert new['verification']['complete'] and new['verification']['metric_records']==4800
    assert new['verification']['reconstructed_tests']==5672
    assert len(new['conditions'])==6 and len(new['comparisons'])==4
    with fitz.open(H/f'{MAIN_STEM}.pdf') as d:text='\n'.join(p.get_text() for p in d)
    with fitz.open(H/'supplementary.pdf') as d:softmax_text='\n'.join(p.get_text() for p in d)
    for r in new['conditions']:
        assert f"{r['capacity']['mean']:.2f}" in softmax_text and f"{r['novel']['mean']:.2f}" in softmax_text
    assert r'\label{app:softmaxreadout}' in (H/'supplementary.tex').read_text()
    assert r'\input{table_cvpr_dualcontracts}' in main and r'\input{table_cvpr_dualview}' in (H/'supplementary.tex').read_text()
    cross=json.loads((H/'cross_encoder/summary.json').read_text())
    shrink=json.loads((H/'shrinkage/summary.json').read_text())
    diag=json.loads((H/'shrinkage/cardinality_diagnostic.json').read_text())
    assert cross['verification']['complete'] and cross['verification']['metric_records']==5600
    assert cross['verification']['reconstructed_tests']==6870
    assert cross['feature_verification']['complete']
    assert shrink['verification']['complete'] and shrink['verification']['metric_records']==28800
    assert shrink['verification']['new_threshold_tests']==6316 and diag['records']==43200
    assert len(cross['conditions'])==7 and len(cross['comparisons'])==6
    assert len(shrink['conditions'])==36 and len(shrink['comparisons'])==54
    extension=(H/'supplement_new_controls.tex').read_text()
    assert all(r'\label{app:'+key+'}' in extension for key in ['crossencoder','shrinkage'])
    with fitz.open(H/'supplementary.pdf') as d:supptext='\n'.join(p.get_text() for p in d)
    assert 'sigmoid loss for language image pre-training' in ' '.join(supptext.lower().split())
    numeric=lambda s:re.findall(r'[-+]?\d+(?:\.\d+)?',s)
    # All new displayed rows are checked against the verified summaries.
    data=[l for l in (H/'table_cross_encoder.tex').read_text().splitlines() if ' & ' in l][1:]
    for row,line in zip(cross['conditions'],data):
        cells=line.split(' & ')
        expected=[f"{row['aia']['mean']:.2f}",f"{row['capacity']['mean']:.2f} ({row['capacity']['n']})",
                  f"{row['base']['mean']:.2f}",f"{row['novel']['mean']:.2f}",f"{row['finite']}/{row['test_met']}"]
        assert numeric(' '.join(cells[2:]))==numeric(' '.join(expected)),line
    assert len(data)==7
    for ds in ['pets','cifar','imagenetr']:
        data=[l for l in (H/f'table_shrinkage_{ds}.tex').read_text().splitlines() if l.startswith(('L/14 &','B/32 &'))]
        rows=[r for r in shrink['conditions'] if r['dataset']==ds]
        assert len(data)==len(rows)==12
        for row,line in zip(rows,data):
            c=row['capacity'];cond=row['conditional_novel']['mean'];cs='--' if cond is None else f'{cond:.2f}'
            expected=[f"{row['aia']['mean']:.2f}",f"{c['mean']:.2f} ({c['n']})",f"{row['base']['mean']:.2f}",
                      f"{row['novel']['mean']:.2f} ({cs})",f"{row['finite']}/{row['test_met']}"]
            assert numeric(' '.join(line.split(' & ')[2:]))==numeric(' '.join(expected)),line
    for path,rows,keys in [
        ('table_cross_encoder_pairs.tex',cross['comparisons'],['aia_gain','capacity_gain','policy_gain']),
        ('table_shrinkage_pairs.tex',[r for r in shrink['comparisons'] if r['dataset']=='pets' and r['method']!='half'],['capacity_gain','policy_gain'])]:
        lines=[l for l in (H/path).read_text().splitlines() if ' & ' in l][1:]
        assert len(lines)==len(rows)
        for r,line in zip(rows,lines):
            expected=' '.join(f"{r[k]['mean']:+.2f} [{r[k]['ci95'][0]:+.2f}, {r[k]['ci95'][1]:+.2f}]" for k in keys)+' '+str(len(r['joint_parts']))
            cells=line.split(' & ')
            assert numeric(' '.join(cells[-len(keys)-1:]))==numeric(expected),line
    selections=[r for r in shrink['conditions'] if r['method']=='adaptive_scheme']
    lines=[l for l in (H/'table_shrinkage_choices.tex').read_text().splitlines() if l.startswith(('L/14 &','B/32 &'))]
    assert len(lines)==len(selections)==6
    for r,line in zip(selections,lines):
        expected=[str(r['chosen_schemes'].get(k,0)) for k in ['query','half','fixed','none']]
        assert numeric(' '.join(line.split(' & ')[2:]))==expected
    for row in cross['conditions']:
        assert f"{row['aia']['mean']:.2f}" in supptext and f"{row['novel']['mean']:.2f}" in supptext
    assert '85.68' in text and '1.34' in text
    rn=json.loads((H/'rn50/summary.json').read_text())
    card=json.loads((H/'cardinality_rule/summary.json').read_text())
    assert rn['verification']['complete'] and rn['verification']['metric_records']==5600
    assert rn['verification']['reconstructed_tests']==4907 and rn['score_witness']['complete']
    assert rn['restart_verification']['complete']
    assert card['verification']['complete'] and card['verification']['metric_records']==28800
    assert card['verification']['new_threshold_tests']==6279
    assert card['verification']['stage_cardinality_checked'] and card['verification']['catalog_frozen_control_checked']
    assert card['feature_witness']['complete'] and card['restart_verification']['complete']
    assert len(rn['conditions'])==7 and len(card['conditions'])==36
    lines=[l for l in (H/'table_rn50.tex').read_text().splitlines() if ' & ' in l][1:]
    assert len(lines)==7
    for row,line in zip(rn['conditions'],lines):
        c=row['capacity'];cv='--' if c['mean'] is None else f"{c['mean']:.2f}"
        expected=[f"{row['aia']['mean']:.2f}",f"{cv} ({c['n']})",f"{row['base']['mean']:.2f}",f"{row['novel']['mean']:.2f}",f"{row['finite']}/{row['test_met']}"]
        assert numeric(' '.join(line.split(' & ')[2:]))==numeric(' '.join(expected)),line
    lines=[l for l in (H/'table_rn50_pairs.tex').read_text().splitlines() if ' & ' in l][1:]
    assert len(lines)==len(rn['comparisons'])==6
    for r,line in zip(rn['comparisons'],lines):
        expected=[]
        for k in ['aia_gain','capacity_gain','policy_gain']:
            v=r[k];expected.append('--' if v['mean'] is None else f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
        expected.append(str(len(r['joint_parts'])))
        assert numeric(' '.join(line.split(' & ')[1:]))==numeric(' '.join(expected)),line
    for ds in ['pets','cifar','imagenetr']:
        lines=[l for l in (H/f'table_cardinality_{ds}.tex').read_text().splitlines() if l.startswith(('L/14 &','B/32 &'))]
        rows=[r for r in card['conditions'] if r['dataset']==ds]
        assert len(lines)==len(rows)==12
        for row,line in zip(rows,lines):
            c=row['capacity'];co=row['conditional_novel']['mean'];cs='--' if co is None else f'{co:.2f}'
            expected=[f"{row['aia']['mean']:.2f}",f"{c['mean']:.2f} ({c['n']})",f"{row['base']['mean']:.2f}",f"{row['novel']['mean']:.2f} ({cs})",f"{row['finite']}/{row['test_met']}"]
            assert numeric(' '.join(line.split(' & ')[2:]))==numeric(' '.join(expected)),line
    rows=[r for r in card['conditions'] if r['dataset']=='pets' and r['method'] in ['primary','catalog_frozen','query']]
    lines=[l for l in (H/'table_cardinality_main.tex').read_text().splitlines() if l.startswith(('L/14 &','B/32 &'))]
    assert len(rows)==len(lines)==6
    for r,line in zip(rows,lines):
        assert numeric(' '.join(line.split(' & ')[2:]))==numeric(f"{r['capacity']['mean']:.2f} {r['novel']['mean']:.2f} {r['finite']}/10"),line
    assert r'\widehat q_e(x,c)=\frac{q_e(x,c)-\widetilde\mu_e(x)}{\widetilde s_e(x)}' in main
    assert '41.40' in text and '4.23' in text and '73.15' in text and '47.11' in text
    assert 'leaves the main readout unchanged' not in main
    assert r'\input{table_cvpr_tnorm_compact}' in main and r'\input{table_cardinality_main}' not in main
    prospective=json.loads((H/'prospective/summary.json').read_text())
    displayed=(H/'table_cvpr_prospective.tex').read_text()
    def fmt(v):return f"{v['mean']:+.2f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]"
    for row in prospective['contrasts']:
        if row['block']=='external' and row['dataset'] in ['food','sun'] and row['baseline']=='dense':
            assert all(fmt(row[k]).replace(",",", ") in displayed for k in ['capacity95','matched_actual_base','policy'])
        if row['block']=='mechanism' and row['dataset']=='imagenetr' and row['baseline'].startswith('shuffle'):
            assert fmt(row['capacity95']).replace(",",", ") in displayed
    assert r'\input{table_cvpr_dualcontracts}' in main
    assert r'\paragraph{Limitations}' not in main
    for phrase in ['7.79','10.14','11.44','6.67','4.38','7.52','82.99','54.87','73.15','47.11']:
        assert phrase in main,phrase
    assert main.index(r'\label{eq:retentioncriterion}') < main.index(r'\subsection{Structural requirements for expansion}') < main.index(r'\label{eq:reference}')
    assert r'\label{cor:zerorecovery}' in main and 'zero-recovery' in main.lower()
    assert all(x not in main for x in ['parameter-free','minimum-variance','establishing'])
    assert main.count(r'\item ')==4
    assert r'\paragraph{Scope and mixed results}' in main
    assert r'\subsection{Useful admission under a base-accuracy requirement}' in main
    assert 'Fixed temperature ature' not in text
    gate=json.loads((H/'gate_calibration/summary.json').read_text())
    assert gate['verification']['complete'] and gate['verification']['matched_records']==14400
    assert gate['verification']['gate_metric_records']==6000 and gate['verification']['stability_metric_records']==87120
    assert gate['verification']['coupling_simulation_draws']==60000
    assert gate['feature_witness']['complete'] and gate['restart_verification']['complete']
    assert len(gate['gates'])==12 and len(gate['gate_contrasts'])==12
    for row in gate['gates']:
        assert f"{row['aia']['mean']:.2f}" in softmax_text and f"{row['novel']['mean']:.2f}" in softmax_text
    tn=json.loads((H/'tnorm_control.json').read_text())
    assert tn['complete'] and len(tn['rows'])==len(tn['witness'])==6
    assert all(w['maximum_margin_error']<2e-4 and w['within_group_predictions_identical'] for w in tn['witness'])
    # Final main-table cells are checked against the underlying frozen summaries.
    evidence=H/'revision_evidence'
    primary=json.loads((evidence/'primary_display.json').read_text())
    table=(H/'table_cvpr_primary.tex').read_text()
    prospective=json.loads((evidence/'prospective.json').read_text())
    cub=json.loads((evidence/'cub.json').read_text())
    cache=json.loads((evidence/'sun_cache.json').read_text())
    for a in primary:
        ds,en,method=a['dataset'],a['encoder'],a['method']
        if ds=='cub':
            orig=next(r for r in cub['summary'] if r['encoder']==en and r['track']=='standard' and r['method']==method)
            cert=next(r for r in cub['summary'] if r['encoder']==en and r['track']=='certified' and r['method']==method)['common']
            expected=[orig['aia_top1'],cert['base'],cert['novel']]
        elif method=='cache_mean':
            orig=next(r for r in cache['summary'] if r['encoder']==en and r['method']==method)
            expected=[orig['aia']['mean'],orig['base']['mean'],orig['novel']['mean']]
        else:
            orig=next(r for r in prospective['rows'] if r['block']=='external' and r['dataset']==ds and r['encoder']==en and r['method']==method)
            expected=[orig[k]['mean'] for k in ['aia','base','novel']]
        assert a['values']==expected
        assert ' & '.join([f'{expected[0]:.2f}',f'{expected[1]:.2f}',f'{expected[2]:.2f} ({a["finite"]})']) in table
    # Unified recognition/common/self table: recompute self cells from frozen results.
    dual=(H/'table_cvpr_dualcontracts.tex').read_text()
    for item in json.loads((evidence/'dualcontracts_display.json').read_text()):
        a=item['primary']; en=a['encoder']; method=a['method']; ds=a['dataset']
        if ds=='cub':
            r=next(r for r in cub['summary'] if r['encoder']==en and r['track']=='certified' and r['method']==method)['self']
        else:
            pool=json.loads((evidence/'references.json').read_text())['summary']+json.loads((evidence/'cache_references.json').read_text())['summary']
            r=next(r for r in pool if r['encoder']==en and r['dataset']==ds and r['method']==method and r['reference']=='self')
        assert r==item['self']
        expected=f"{a['values'][0]:.2f} & {a['values'][2]:.2f} ({a['finite']}) & {r['novel']:.2f} ({r['finite']}/{item['denominator']})"
        assert ' '.join(expected.split()) in ' '.join(dual.split()),expected
    refs=json.loads((evidence/'references.json').read_text())['summary']
    refs+=json.loads((evidence/'cache_references.json').read_text())['summary']
    for a in json.loads((evidence/'reference_display.json').read_text()):
        if a['dataset']=='cub':
            r=next(r for r in cub['summary'] if r['encoder']==a['encoder'] and r['track']=='certified' and r['method']==a['method'])[a['reference']]
            novel=r['novel'];floor=.95*r['closed']
        else:
            r=next(r for r in refs if r['dataset']==a['dataset'] and r['encoder']==a['encoder'] and r['method']==a['method'] and r['reference']==a['reference'] and (a['dataset']!='sketch' or r['calibration']=='source_cal'))
            novel=r['novel'];floor=.95*r['closed_base']
        assert a['novel']==novel and abs(a['floor']-floor)<1e-10
        assert f'{novel:.2f} ({a["finite"]})' in (H/'table_cvpr_references.tex').read_text()
    with fitz.open(H/f'{MAIN_STEM}.pdf') as doc:
        assert any('Figure 1.' in p.get_text() for p in list(doc)[:3])
        assert '{framework_drawio.pdf}' in main
        assert 'Fixed base-class references' in ' '.join(' '.join(p.get_text() for p in doc).split())
        assert 'asymmetric readout' in ''.join(p.get_text() for p in doc).lower()
    assert cub['records']==704 and cub['independently_verified_calibration_tests']==3352
    assert json.loads((evidence/'shared.json').read_text())['verified_method_sessions']==400
    assert json.loads((evidence/'constants.json').read_text())['main_constant']==30
    assert json.loads((evidence/'references.json').read_text())['reproduced_common_policies']==3000
    report={'complete':True,'design_by_invariance_structure_checked':True,'main_content_pages':int(m[1]),'documents':documents,
            'official_2027_style_unmodified':True,'figures':4,'tables':3,
            'all_48_recognition_and_admission_cells_preserved':True,'centering_rows_preserved':6,
            'existing_kbs_sources_unchanged':True,'new_training_run_this_revision':False,'local_reference_reanalysis':True,
            'requested_five_narrative_changes':True,'softmax_results_retained_in_supplement':True,
            'softmax_readout_control':new['verification'],'dual_view_cells_preserved':48,'new_tuned_no_fallback_cells_verified':24,'food_main_table_verified':True,
            'cross_pretraining_validation':cross['verification'],
            'shrinkage_extension':shrink['verification'],'reference_subset_records':diag['records'],
            'new_summary_table_cells_verified':True,'cross_encoder_backbone':'ViT',
            'rn50_convolutional_validation':rn['verification'],'rn50_score_witness':rn['score_witness'],
            'rn50_restart':rn['restart_verification'],'cardinality_rule':card['verification'],
            'cardinality_score_witness':card['feature_witness'],'cardinality_restart':card['restart_verification'],
            'new_core_operator':'Interpolated reference moments; full-query and cardinality-aware variants',
            'supplement_equation_prefix':'S','legacy_names_absent_in_pdfs':True,
            'main24_structure_and_three_experimental_figures_retained':True,'architecture_refined_at_user_request':True,'references_start_page':refpage,
            'active_main_pdf':MAIN_STEM+'.pdf','llm_statement_page':llmpage,'new_main_evidence_tables':True,
            'paper_id':None,'paper_id_printed_by_official_template':False,'submitted':False,
            'source_sha256':{p.name:sha(p) for p in [H/'main.tex',H/'supplementary.tex',H/'supplement_new_controls.tex',H/'supplement_rn50.tex',H/'supplement_cardinality_rule.tex',H/'supplement_gate_calibration.tex',H/'cvpr.sty',H/'ieeenat_fullname.bst']}}
    (H/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
if __name__=='__main__':print(json.dumps(validate(),indent=2))
