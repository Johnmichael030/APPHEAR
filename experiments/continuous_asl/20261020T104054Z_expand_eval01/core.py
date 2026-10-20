from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,urllib.request,urllib.parse,urllib.error,sys
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
TREE=ROOT/'experiments/continuous_asl'
OLD=TREE/'20261020_continuous_expand06'
FOUR=TREE/'20261020_continuous_expand04'
FIVE=TREE/'20261020_continuous_expand05'
PYTHON=TREE/'20261019T155428Z_cslr01/runtime/venv/Scripts/python.exe'
SCHEMA='apphear.continuous_asl.candidate01_causal378.v1'
sys.path.insert(0,str(ROOT/'tools/continuous_asl'))
sys.path.insert(0,str(ROOT))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def write(path,obj):
 path=Path(path).resolve()
 if RUN not in path.parents:raise ValueError('NEW_CONTINUATION_SCOPE')
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2);f.write('\n')
def fetch(url,name,cap=16777216,target_relative=None):
 path=(RUN/target_relative if target_relative else RUN/'sources'/name).resolve();receipt=Path(str(path)+'.receipt.json')
 if RUN not in path.parents:raise ValueError('DOWNLOAD_OUTPUT_SCOPE')
 if path.exists():
  if not receipt.exists() or sha(path)!=read(receipt)['sha256']:raise ValueError('EXISTING_UNVERIFIED_DOWNLOAD')
  return path
 if shutil.disk_usage(ROOT).free<10737418240+2*cap:raise ValueError('DISK_HEADROOM')
 previous=read(TREE/'20261020_continuous_expand08/execution_summary.json')['acquisition_budget']['measured_local_acquired_payload_bytes_including_prior_How2Sign']
 acquired=sum(read(p)['bytes'] for p in RUN.rglob('*.receipt.json'))
 if previous+acquired+cap>107374182400:raise ValueError('100GIB_CAP')
 path.parent.mkdir(parents=True,exist_ok=True);partial=Path(str(path)+'.partial')
 request=urllib.request.Request(url,headers={'User-Agent':'AppHear-research-intake/1.0'})
 try:
  with urllib.request.urlopen(request,timeout=60) as response:
   if urllib.parse.urlsplit(response.geturl()).hostname not in ('dai.cs.rutgers.edu','www.bu.edu','bu.edu','csr.bu.edu'):raise ValueError('UNVERIFIED_REDIRECT_HOST')
   declared=int(response.headers.get('Content-Length') or 0)
   if declared>cap:raise ValueError('DECLARED_SIZE_CAP')
   count=0
   with partial.open('xb') as f:
    while True:
     block=response.read(min(1048576,cap-count+1))
     if not block:break
     count+=len(block)
     if count>cap:raise ValueError('STREAM_SIZE_CAP')
     f.write(block)
   partial.rename(path)
   write(receipt,{'source_url':url,'bytes':count,'sha256':sha(path),'utc':datetime.now(timezone.utc).isoformat(),'content_type':response.headers.get('Content-Type'),'cap_bytes':cap,'authorization':'User100GiB noncommercial research acquisition; official published destination verified.'})
   print(json.dumps({'downloaded':name,'bytes':count}),flush=True)
   return path
 except Exception as e:
  failure=RUN/'sources'/(name+'.failed.json')
  if not failure.exists():write(failure,{'url':url,'error':str(e),'utc':datetime.now(timezone.utc).isoformat(),'bypass_attempted':False})
  raise
