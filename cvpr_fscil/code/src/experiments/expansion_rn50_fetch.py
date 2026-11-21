"""Pinned, resumable official CLIP RN50 checkpoint acquisition."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,sys,time
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kbs_runtime import atomic_json,digest_file,check_pause
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/expansion_rn50/model'
SHA='afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'
URL=f'https://openaipublic.azureedge.net/clip/models/{SHA}/RN50.pt';SIZE=255827503

def main():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'RN50.pt'
    if path.exists() and digest_file(path)==SHA:
        manifest=dict(complete=True,model='OpenAI CLIP RN50',url=URL,size=SIZE,sha256=SHA,
                      protocol_sha256=digest_file(ROOT/'experiments/expansion_rn50_protocol.md'))
        mp=OUT/'manifest.json'
        if mp.exists():assert json.loads(mp.read_text())==manifest
        else:atomic_json(manifest,mp)
        return
    parts=OUT/'ranges';parts.mkdir(exist_ok=True);chunk=8*1024*1024
    def acquire(start):
        check_pause();stop=min(SIZE,start+chunk)-1;dest=parts/f'{start:09d}.bin'
        if dest.exists() and dest.stat().st_size==stop-start+1:return dest
        for retry in range(6):
            try:
                r=requests.get(URL,headers={'Range':f'bytes={start}-{stop}'},timeout=(30,90))
                r.raise_for_status();assert r.status_code==206
                assert r.headers['Content-Range'].startswith(f'bytes {start}-{stop}/')
                assert len(r.content)==stop-start+1
                temp=dest.with_suffix('.partial');temp.write_bytes(r.content);temp.replace(dest);return dest
            except Exception:
                if retry==5:raise
                time.sleep(min(2**retry,20))
    starts=list(range(0,SIZE,chunk));done=0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(acquire,s) for s in starts]):
            future.result();done+=1;print('RN50 DOWNLOAD',done,'/',len(starts),flush=True)
    temp=OUT/'RN50.pt.partial'
    with temp.open('wb') as target:
        for start in starts:target.write((parts/f'{start:09d}.bin').read_bytes())
    assert temp.stat().st_size==SIZE and digest_file(temp)==SHA;temp.replace(path)
    atomic_json(dict(complete=True,model='OpenAI CLIP RN50',url=URL,size=SIZE,sha256=SHA,
                     protocol_sha256=digest_file(ROOT/'experiments/expansion_rn50_protocol.md')),OUT/'manifest.json')
    # Remove only this downloader's verified component files after final verification.
    for start in starts:(parts/f'{start:09d}.bin').unlink()
    print('RN50 CHECKPOINT VERIFIED',flush=True)
if __name__=='__main__':main()
