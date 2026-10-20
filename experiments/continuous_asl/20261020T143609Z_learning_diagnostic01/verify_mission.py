"""Read-only check of explicit mission artifacts/inputs; no model load or TEST discovery."""
from pathlib import Path
import csv,hashlib,json
RUN=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(n):return json.loads((RUN/n).read_text())
it=read('input_integrity.json');assert it['status']=='PASS' and not it['changed']
for p,e in it['before'].items():
 q=Path(p);s=q.stat();assert sha(q)==e['sha256'] and s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'],p
count=0
for line in (RUN/'artifact_hashes.sha256').read_text().splitlines():
 digest,rel=line.split('  ',1);p=(RUN/rel).resolve();assert RUN in p.parents and sha(p)==digest,rel;count+=1
cfg=read('config.json');assert sha(RUN/'run_mission.py')==cfg['script_sha256']
assert read('gate_A.json')['status']=='PASS' and read('gate_B.json')['status']=='PASS'
d=read('decision.json');assert not d['promote'] and not d['TEST_used']
if d['full_candidate_trained']:
 p=RUN/d['selected_checkpoint']['path'];assert sha(p)==d['selected_checkpoint']['sha256']
 score=read('selected_primary80.json');assert score['reference_tokens']==485 and score['utterances']==80
 assert abs(score['WER']-(score['substitutions']+score['deletions']+score['insertions'])/485)<1e-12
 assert read('selected_strong27.json')['utterances']==27
 assert read('selected_train1500.json')['utterances']==1500
print(json.dumps(dict(integrity='PASS',input_files_unchanged=len(it['before']),new_artifact_hashes_pass=count,Gate_A='PASS',Gate_B='PASS',full_candidate_trained=d['full_candidate_trained'],status=d['status'],TEST_used=False,promoted=False),indent=2))

