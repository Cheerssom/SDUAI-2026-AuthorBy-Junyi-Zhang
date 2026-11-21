"""Pinned SUN397 full-image mirror, parallel resumable verified parquet files."""
from pathlib import Path
import requests,json,concurrent.futures,hashlib
from expansion_range_download import download
ROOT=Path('E:/expansion_external_data/sun397');REPO='dpdl-benchmark/sun397';REV='7e6af6a2499ad708618be868e1471eac0aca1168'
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    files=requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/data?limit=1000',timeout=30).json()
    config=dict(repo=REPO,revision=REV,files=files);(ROOT/'download_manifest.json').write_text(json.dumps(config,indent=2))
    text=requests.get(f'https://huggingface.co/datasets/{REPO}/raw/{REV}/README.md',timeout=30).text;(ROOT/'README_source.md').write_text(text,encoding='utf-8')
    def one(f):
        p=ROOT/Path(f['path']).name;download(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{f["path"]}',p,f['size'],workers=12)
        with p.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
        assert sha==f['lfs']['oid'];print('SUN VERIFIED',p.name,flush=True)
    with concurrent.futures.ThreadPoolExecutor(2) as ex:list(ex.map(one,files))
    (ROOT/'download_complete.json').write_text(json.dumps(dict(complete=True,repo=REPO,revision=REV,files=len(files))))
    print('SUN ALL DOWNLOADED',flush=True)
if __name__=='__main__':main()
