"""Read-only verification of completed nested192 control and synthetic phone evidence."""
from pathlib import Path
import csv,hashlib,json

RUN=Path(__file__).resolve().parent
def read(name):return json.loads((RUN/name).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  new=[i]
  for j,y in enumerate(b,1):new.append(min(row[j]+1,new[-1]+1,row[j-1]+(x!=y)))
  row=new
 return row[-1]

cfg=read('control_config.json');d=read('decision.json');rows=read('subset_manifest.json');fold=read('fold_manifest.json')
assert len(rows)==len({r['sample_id'] for r in rows})==192
assert {r['sample_id'] for r in rows}<=set(fold['train_sample_ids'])
assert not {r['sample_id'] for r in rows}&set(fold['internal_eval_sample_ids'])
assert d['complete_epochs']==30 and d['optimizer_steps']==720 and not d['promote'] and not d['TEST_used']
assert sha(RUN/d['checkpoint']['path'])==d['checkpoint']['sha256']
assert sha(RUN/'control192.py')==cfg['control_script_sha256']
assert sha(RUN/'fold_manifest.json')==cfg['frozen_fold_sha256']
assert sha(RUN/'vocabulary.json')==cfg['vocabulary_sha256']
assert sha(RUN/'normalization.npz')==cfg['normalization_sha256']
before=read('control_input_hashes_before.json')
for p,e in before.items():
 q=Path(p);s=q.stat();assert sha(q)==e['sha256'] and s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'],p
errors={}
for name,count in [('final_predictions.json',192),('heldout80_predictions.json',80)]:
 r=read(name);pp=r['predictions'];assert len(pp)==count
 n=sum(len(p['reference']) for p in pp)
 edits=sum(distance(p['reference'],p['hypothesis']) for p in pp)
 assert n==r['reference_tokens'] and edits==r['substitutions']+r['deletions']+r['insertions']
 assert abs(edits/n-r['WER'])<1e-12
 errors[name]=dict(edits=edits,reference_tokens=n,WER=edits/n)
assert {p['sample_id'] for p in read('heldout80_predictions.json')['predictions']}==set(fold['compatible_eval_sample_ids'])
post=read('postfit_evaluation_integrity.json')
for p,v in post['heldout_feature_hashes'].items():assert sha(p)==v
assert post['initial_tensors_identical'] and post['no_reserved_TEST_access']
assert len(list(csv.DictReader((RUN/'epoch_timing.csv').open())))==30
phone=read('PHONE_PARITY_RESULTS.json')
assert phone['status']=='PASS' and phone['apphear_package_unchanged'] and not phone['camera_used']
assert 'OK (2 tests)' in (RUN/'phone_instrumentation.log').read_text()
hashes=0
if (RUN/'artifact_hashes.sha256').exists():
 for line in (RUN/'artifact_hashes.sha256').read_text().splitlines():
  digest,rel=line.split('  ',1);p=RUN/rel;assert sha(p)==digest,rel;hashes+=1
print(json.dumps(dict(integrity='PASS',input_paths_unchanged=len(before),heldout_feature_hashes_verified=len(post['heldout_feature_hashes']),output_hashes_verified=hashes,metrics=errors,phone_synthetic_tests='2/2PASS',overall='PARTIAL',TEST_used=False,promoted=False),indent=2))
