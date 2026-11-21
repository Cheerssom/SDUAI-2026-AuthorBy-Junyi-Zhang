source /etc/network_turbo
/root/autodl-tmp/fscil-env/bin/python - <<'PY'
import requests,concurrent.futures,time
urls=['https://drive.google.com/uc?export=download&id=1_x4o0iFetEv-T3PeIxdSbBPUG4hFfT8U','https://drive.google.com/uc?export=download&id=1EtKb1Bxd9ZtK-WUFk5dIhoOdhLeohA1q','https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz']
def f(url):
 try:
  r=requests.get(url,stream=True,timeout=12);print('PROBE',url,r.status_code,r.headers.get('content-length'),r.headers.get('content-type'),flush=True);r.close()
 except Exception as e:print('FAIL',url,str(e)[:150],flush=True)
with concurrent.futures.ThreadPoolExecutor() as p:list(p.map(f,urls))
PY
