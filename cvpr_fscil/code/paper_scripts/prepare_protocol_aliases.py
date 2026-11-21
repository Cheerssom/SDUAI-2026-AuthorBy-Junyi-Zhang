"""Provide descriptive protocol filenames without altering frozen run archives."""
from pathlib import Path
import hashlib,json,re
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
mapping={
 'kbs_calibration_budget_protocol.md':'expansion_calibration_budget_protocol.md',
 'kbs_reviewer_revision_protocol.md':'expansion_controlled_ablation_protocol.md',
 'kbs_reviewer_revision_scope.md':'expansion_controlled_ablation_scope.md',
 'kbs_full_fusion_retuning_protocol.md':'expansion_full_fusion_retuning_protocol.md',
 'kbs_sketch_execution_protocol_v1.md':'expansion_sketch_execution_protocol_v1.md',
 'kbs_sketch_prospective_protocol_DRAFT.md':'expansion_sketch_prospective_protocol_DRAFT.md'}
receipt=[]
for old,new in mapping.items():
 source=ROOT/'experiments'/old;s=source.read_text(encoding='utf-8')
 for a,b in mapping.items():s=s.replace(a,b)
 s=s.replace('kbs_independent_risk','the frozen independent-risk study')
 s=s.replace('under experiments/kbs_reviewer_revision','in the frozen controlled-ablation archives')
 s=s.replace('experiments/kbs_strengthening/PAUSE','the shared runtime PAUSE marker')
 s=s.replace('Locked five-block reviewer revision','Locked five-block controlled ablation')
 s=s.replace('reviewer revision','controlled ablation')
 assert 'kbs_' not in s,(old,[l for l in s.splitlines() if 'kbs_' in l])
 dest=ROOT/'experiments'/new
 if dest.exists():assert dest.read_text(encoding='utf-8')==s,new
 else:dest.write_text(s,encoding='utf-8')
 local=H/'experiments'/new;local.parent.mkdir(exist_ok=True);local.write_text(s,encoding='utf-8')
 receipt.append({'original_file':old,'neutral_file':new,
                 'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                 'neutral_sha256':hashlib.sha256(local.read_bytes()).hexdigest()})
(H.parent/'cvpr2027_revisions/20261027_protocol_alias_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
# Add neutral entry points to the public reproduction checkout, retaining history.
P=ROOT.parent/'Reference-Anchored-Candidate-Tables'
for r in receipt:
 (P/'experiments'/r['neutral_file']).write_bytes((H/'experiments'/r['neutral_file']).read_bytes())
print('Created six neutral protocol entry points; original archives retained.')
