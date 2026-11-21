"""Compact on-demand status for the ongoing four-block extension."""
from pathlib import Path
import json,datetime,re
import psutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_prospective_validation';DATA=Path('E:/expansion_external_data')
def progress(p,total):
    if p.exists():n=p.stat().st_size
    else:n=sum(f.stat().st_size for f in p.with_name(p.name+'.ranges').glob('*') if f.suffix in ['.bin','.partial'])
    return dict(downloaded_gib=round(n/2**30,3),total_gib=round(total/2**30,3),percent=round(100*n/total,1),ready=p.exists())
def main():
    state=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),complete_all=(OUT/'complete_all.json').exists())
    state['blocks']={b:dict(complete=(OUT/b/'complete.json').exists(),finished_tasks=len(list((OUT/b).glob('*/*/*/p*/complete.json'))),expected_tasks=(60 if b=='mechanism' else 260)) for b in ['mechanism','cardinality']}
    state['blocks']['shift']=dict(complete=(OUT/'shift/complete.json').exists(),finished_archives=len([p for p in (OUT/'shift/results').glob('*/p*/*.json') if p.name!='complete.json']),expected_archives=180)
    state['blocks']['external']=dict(finished_tasks=len(list((OUT/'external').glob('*/*/catalog/p*/complete.json'))),expected_tasks=40)
    feature_progress={}
    for ds in ['food','sun']:
        log=ROOT/f'experiments/expansion_prospective_{ds}_features.log'
        feature_counts={}
        if log.exists():
            for enc,_,role,done,total,elapsed in re.findall(r'EXTERNAL FEATURES (l14|b32) (food|sun) (\w+) (\d+) / (\d+) elapsed (\d+)',log.read_text(encoding='utf-8',errors='replace')):
                feature_counts.setdefault(enc,{})[role]=dict(done=int(done),total=int(total),elapsed_seconds=int(elapsed))
        feature_progress[ds]={enc:dict(verified=(OUT/f'external/{ds}/{enc}/verification.json').exists(),roles=feature_counts.get(enc,{}),evaluated_partitions=len(list((OUT/f'external/{enc}/{ds}/catalog').glob('p*/complete.json')))) for enc in ['l14','b32']}
    state['feature_progress']=feature_progress
    state['food']=progress(DATA/'food-101.tar.gz',4996278331)
    mp=DATA/'sun397/download_manifest.json'
    if mp.exists():
        m=json.loads(mp.read_text());parts=[progress(mp.parent/Path(f['path']).name,f['size']) for f in m['files']];total=sum(f['size'] for f in m['files']);n=sum((mp.parent/Path(f['path']).name).stat().st_size if (mp.parent/Path(f['path']).name).exists() else sum(p.stat().st_size for p in (mp.parent/(Path(f['path']).name+'.ranges')).glob('*') if p.suffix in ['.bin','.partial']) for f in m['files'])
        state['sun']=dict(downloaded_gib=round(n/2**30,3),total_gib=round(total/2**30,3),percent=round(100*n/total,1),ready_files=sum(p['ready'] for p in parts),total_files=len(parts))
    active=[]
    for proc in psutil.process_iter(['pid','cmdline','name']):
        try:
            if not (proc.info['name'] or '').lower().startswith('python') or proc.pid==psutil.Process().pid:continue
            argv=proc.info['cmdline'] or [];scripts=[Path(a).name for a in argv if a.endswith('.py') and Path(a).name.startswith('expansion_')]
            if scripts:active.append(dict(pid=proc.pid,script=scripts[0]))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    state['active']=active
    if (OUT/'queue_status.json').exists():state['queue']=json.loads((OUT/'queue_status.json').read_text())
    (OUT/'status.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
if __name__=='__main__':main()
