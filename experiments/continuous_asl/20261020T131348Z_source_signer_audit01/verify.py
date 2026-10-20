"""Read-only design and hash verifier. No model packages, inference, training or networking."""
from pathlib import Path
import hashlib,json,subprocess,sys,re,csv
R=Path(__file__).resolve().parent;P=R.parent/'20261020T104054Z_expand_eval01';PREV=R.parent/'20261020T124813Z_label_support01'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
count=0
for line in (R/'artifact_hashes.sha256').read_text().splitlines():
 expected,name=line.split('  ',1);p=(R/name).resolve();assert R in p.parents and sha(p)==expected,name;count+=1
for filename in ['input_hashes.json','supplement_input_hashes.json','finish_input_hashes.json']:
 for p,expected in read(R/filename).items():assert sha(p)==expected,p
rows=[r for r in read(P/'data/processed/train_dev/manifest.json') if r['split']=='train'];lookup={r['sample_id']:r for r in rows};assert len(lookup)==1706
def family(r):
 m=re.match(r'^(\d{4}-\d{2}-\d{2})_',r['source_group']);return r['signer_id']+'::'+(m[1] if m else 'UNRESOLVED_ALL_UNDATED_MASTERS')
checked=0
for path in (R/'folds').glob('A_guarded*.json'):
 f=read(path);train=[lookup[i] for i in f['train_sample_ids']];ev=[lookup[i] for i in f['internal_eval_sample_ids']]
 assert not ({family(r) for r in train}&{family(r) for r in ev})
 for key in ['sample_id','source_group','collection','raw_sha256','decoded_pixel_sha256']:assert not ({r.get(key) for r in train if r.get(key)}&{r.get(key) for r in ev if r.get(key)})
 labels={t for r in train for t in r['glosses']};expected={r['sample_id'] for r in ev if set(r['glosses'])<=labels};assert expected==set(f['compatible_eval_sample_ids'])
 assert f['training_authorized'] is False;checked+=1
decision=read(R/'guarded_decision.json');assert decision['training_authorized'] is False
assert sha(R/'folds'/f"{decision['candidate']['fold']}.json")==decision['candidate']['manifest_sha256']
assert not list(R.rglob('*.pt')) and not list(R.rglob('*.apk'))
subprocess.run([sys.executable,'-B',str(PREV/'verify.py')],check=True)
print(json.dumps({'status':'PASS_READ_ONLY_SOURCE_SIGNER_AUDIT_VERIFICATION','new_artifacts_hashed':count,'guarded_TRAIN_only_folds':checked,'training_performed':False,'TEST_scored':False,'crossed_identifiability':'BLOCKED'}))
