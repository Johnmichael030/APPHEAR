"""Read-only checksum and frozen-parent verification; no model execution."""
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
count=0
for line in (R/'artifact_hashes.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);p=(R/name).resolve();assert R in p.parents and sha(p)==expected,name;count+=1
for p,expected in json.loads((R/'input_hashes.json').read_text()).items():assert sha(p)==expected,p
for line in (R/'SUBSET_MANIFEST.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);assert sha(R/name)==expected
status=json.loads((R/'pilot497/training_status.json').read_text());assert status['used_cuda'] and not status['TEST_used'] and not status['promoted']
assert sha(R/'pilot497'/status['best_checkpoint']['path'])==status['best_checkpoint']['sha256']
subprocess.run([sys.executable,'-B',str(R.parent/'20261020T123237Z_collapse_diagnosis01/verify.py')],check=True)
print(json.dumps({'status':'PASS_ARTIFACT_AND_FROZEN_PRESERVATION','new_files':count,'pilot_epochs':status['completed_epochs'],'TEST_used':False,'promotion':False}))
