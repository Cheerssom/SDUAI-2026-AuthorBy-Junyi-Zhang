from pathlib import Path
import sys,json,zipfile,torch
R=Path.cwd();sys.path.insert(0,str(R/'src/experiments'));import kbs_reviewer_revision as prev
O=R/'experiments/expansion_constant_sensitivity_remote';torch.set_num_threads(2)
with zipfile.ZipFile(O/'payload.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for name in ['expansion_constant_sensitivity_remote.py','kbs_paired_accuracy.py','kbs_risk_control.py','kbs_runtime.py']:z.write(R/'src/experiments'/name,'src/experiments/'+name)
 for name in ['fcs_expansion.py','fcs_retention_value.py','kbs_full_fusion_retuning.py','kbs_calibration_fairness.py','expansion_prospective_validation.py']:z.write(R/'src/experiments'/name,'frozen_sources/'+name)
 z.write(O/'protocol.md','protocol.md')
 for enc in ['l14','b32']:
  d,_=prev.load_data('imagenetr',enc);d={k:(torch.as_tensor(d[k]) if k.endswith('_y') else d[k]) for k in ['E_fit','E_tune','E_cal','E_test','E_support','T','fit_y','tune_y','cal_y','test_y','support_y']};f=O/f'{enc}_frozen.pt';torch.save(d,f);z.write(f,f'inputs/{enc}/frozen.pt');del d
  src=R/f'experiments/expansion_prospective_validation/cardinality/{enc}/imagenetr'
  for p in src.glob('B*/p*/selection.json'):
   rel=p.relative_to(src);z.write(p,f'inputs/{enc}/tasks/{rel.as_posix()}');z.write(p.parent/'weights.pt',f'inputs/{enc}/tasks/{rel.parent.as_posix()}/weights.pt')
  print('PACKED',enc,flush=True)
print('MB',round((O/'payload.zip').stat().st_size/1e6,1))
