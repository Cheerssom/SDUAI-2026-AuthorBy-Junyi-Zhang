"""Pinned, resumable public model acquisition; no credentials are required."""
from pathlib import Path
import json,hashlib,time
import requests
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[2]
REV='7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'
OUT=ROOT/'experiments/expansion_cross_encoder/model'
FILES=['config.json','model.safetensors','preprocessor_config.json','special_tokens_map.json','spiece.model','tokenizer.json','tokenizer_config.json']
def acquire_weights():
    target=OUT/'model.safetensors';expected='2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8'
    if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==expected;return
    size=812672320;block=8*1024*1024;folder=OUT/'weight_parts';folder.mkdir(exist_ok=True)
    url=f'https://huggingface.co/google/siglip-base-patch16-224/resolve/{REV}/model.safetensors?download=true'
    def one(start):
        end=min(size,start+block)-1;p=folder/f'{start:012d}.part'
        if p.exists():assert p.stat().st_size==end-start+1;return end-start+1
        for attempt in range(5):
            try:
                r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(15,30))
                r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                tmp=p.with_suffix('.partial')
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(65536):f.write(chunk)
                assert tmp.stat().st_size==end-start+1;tmp.replace(p);return end-start+1
            except Exception as exc:
                print('RANGE RETRY',start,attempt+1,type(exc).__name__,flush=True)
                if attempt==4:raise
    done=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for task in as_completed([pool.submit(one,start) for start in range(0,size,block)]):
            done+=task.result();print('MODEL BYTES',done,'/',size,flush=True)
    partial=target.with_suffix('.assembling')
    with partial.open('wb') as f:
        for start in range(0,size,block):f.write((folder/f'{start:012d}.part').read_bytes())
    assert partial.stat().st_size==size and hashlib.sha256(partial.read_bytes()).hexdigest()==expected
    partial.replace(target)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    session=requests.Session()
    for name in FILES:
        if name=='model.safetensors':acquire_weights();continue
        target=OUT/name
        if target.exists():continue
        url=f'https://huggingface.co/google/siglip-base-patch16-224/resolve/{REV}/{name}?download=true'
        partial=target.with_name(name+'.partial')
        for attempt in range(5):
            start=partial.stat().st_size if partial.exists() else 0
            try:
                response=session.get(url,headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(20,60))
                response.raise_for_status()
                if start and response.status_code!=206:start=0
                with partial.open('ab' if start else 'wb') as f:
                    for block in response.iter_content(1024*1024):
                        if block:f.write(block)
                expected=response.headers.get('Content-Length')
                if expected:assert partial.stat().st_size==start+int(expected)
                partial.replace(target)
                print('MODEL FILE',name,target.stat().st_size,flush=True);break
            except Exception as exc:
                print('ACQUISITION RETRY',name,attempt+1,type(exc).__name__,str(exc)[:180],flush=True)
                if attempt==4:raise
                time.sleep(2)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/n for n in FILES]}
    (OUT/'manifest.json').write_text(json.dumps({'model':'google/siglip-base-patch16-224','revision':REV,'file_sha256':hashes},indent=2))
    print('MODEL ACQUISITION COMPLETE',flush=True)
if __name__=='__main__':main()
