"""Read-only: no model execution, fitting, download or TEST scoring."""
from pathlib import Path
import hashlib,json,subprocess,sys
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];PARENT=RUN.parent/'20261020T104054Z_expand_eval01'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
count=0
for line in (RUN/'artifact_hashes.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);p=(RUN/name).resolve();assert RUN in p.parents and sha(p)==expected, name;count+=1
subprocess.run([sys.executable,'-B',str(PARENT/'verify_completed.py')],check=True)
s=json.loads((RUN/'pilot64/training_status.json').read_text());assert s['used_cuda'] and not s['TEST_used'] and not s['promoted']
assert sha(RUN/'pilot64/manifest.json')==s['manifest_sha256']
assert sha(RUN/'pilot64'/s['best_checkpoint']['path'])==s['best_checkpoint']['sha256']
print(json.dumps({'status':'PASS_INTEGRITY_AND_PRESERVATION','new_files_hashed':count,'pilot_epochs':s['completed_epochs'],'recognition':'PARTIAL','promoted':False}))
