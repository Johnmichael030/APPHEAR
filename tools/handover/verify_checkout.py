"""Verify materialized handover files. Never trains, downloads data or accesses TEST."""
from pathlib import Path
import argparse,hashlib,json,os
def physical(p):
 p=p.resolve()
 return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
def sha(p):
 h=hashlib.sha256()
 with physical(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def verify(root,source_only=False):
 manifest=json.loads((root/'handover/CONTENT_MANIFEST.json').read_text(encoding='utf-8'));errors=[];checked=0
 for r in manifest['files']:
  if source_only and r['delivery']=='lfs':continue
  p=physical(root/r['path'])
  if not p.is_file():errors.append(dict(path=r['path'],problem='MISSING'))
  elif p.stat().st_size!=r['bytes']:errors.append(dict(path=r['path'],problem='SIZE_MISMATCH_OR_LFS_POINTER; run git lfs pull'))
  elif sha(p)!=r['sha256']:errors.append(dict(path=r['path'],problem='HASH_MISMATCH'))
  else:checked+=1
 result=dict(status='FAIL' if errors else 'PASS_MATERIALIZED_HANDOVER',checked=checked,source_only=source_only,errors=errors,scope='Delivered files only. Excluded datasets, participant data and reserved TEST are not read.')
 print(json.dumps(result,indent=2))
 return not errors
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-only',action='store_true');a=p.parse_args();raise SystemExit(0 if verify(Path(__file__).resolve().parents[2],a.source_only) else 1)
