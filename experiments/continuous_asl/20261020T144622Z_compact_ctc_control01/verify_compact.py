"""Verify explicit compact experiment inputs/outputs; no inference or training."""
from pathlib import Path
import hashlib,json
RUN=Path(__file__).resolve().parent
def read(n):return json.loads((RUN/n).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
integ=read('input_integrity.json');assert integ['status']=='PASS' and not integ['changed']
for p,e in integ['before'].items():
 q=Path(p);s=q.stat();assert sha(q)==e['sha256'] and s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'],p
outputs=0
for line in (RUN/'artifact_hashes.sha256').read_text().splitlines():
 digest,rel=line.split('  ',1);p=(RUN/rel).resolve();assert RUN in p.parents and sha(p)==digest,rel;outputs+=1
cfg=read('control_config.json');assert sha(RUN/'control.py')==cfg['control_script_sha256']
assert sha(RUN/'subset_manifest.json')==read('frozen_control_inputs.json')['subset_manifest_sha256']
rows=read('subset_manifest.json');assert len(rows)==len({r['sample_id'] for r in rows})==96
d=read('decision.json');assert d['status']=='PASS' and not d['promote'] and not d['TEST_used']
assert sha(RUN/d['checkpoint']['path'])==d['checkpoint']['sha256']
initial=read('initial_predictions.json');final=read('final_predictions.json')
assert len(final['predictions'])==96 and final['reference_tokens']==756
assert abs((final['substitutions']+final['deletions']+final['insertions'])/756-final['WER'])<1e-12
def dist(a,b):
 v=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  u=[i]
  for j,y in enumerate(b,1):u.append(min(v[j]+1,u[j-1]+1,v[j-1]+(x!=y)))
  v=u
 return v[-1]
assert sum(dist(p['reference'],p['hypothesis']) for p in final['predictions'])==91
assert final['WER']<=cfg['pass_WER'] and final['CTC_loss']<=.5*initial['CTC_loss']
assert all(w<=.5 for w in final['family_WER'].values()) and final['rare_bearing_WER']<=.5 and final['upper_input_quartile_WER']<=.5
print(json.dumps(dict(integrity='PASS',input_files_unchanged=len(integ['before']),output_hashes_verified=outputs,independent_edit_errors=91,reference_tokens=756,compact_gate='PASS',overall='PARTIAL',TEST_used=False,promoted=False),indent=2))

