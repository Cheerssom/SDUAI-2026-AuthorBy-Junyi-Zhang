"""Confirm every pinned SUN shard before releasing the queued GPU stage."""
from pathlib import Path
import json,time
root=Path('E:/expansion_external_data/sun397');manifest=root/'download_manifest.json'
while True:
    if (root/'download_complete.json').exists():break
    if manifest.exists():
        m=json.loads(manifest.read_text());ok=True
        for f in m['files']:
            p=root/Path(f['path']).name;digest=p.with_suffix(p.suffix+'.sha256')
            if not p.exists() or p.stat().st_size!=f['size'] or not digest.exists() or digest.read_text().strip()!=f['lfs']['oid']:
                ok=False;break
        if ok:
            (root/'download_complete.json').write_text(json.dumps(dict(complete=True,repo=m['repo'],revision=m['revision'],files=len(m['files']),verified_range_download_sha256=True)))
            print('SUN READY',flush=True);break
    time.sleep(30)
