from pathlib import Path
import hashlib,json
RUN=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(RUN/'control_config.json');rows=read(RUN/'subset_manifest.json')
for name,key in [('vocabulary.json','vocabulary_sha256'),('normalization.npz','normalization_sha256'),('fold_manifest.json','frozen_fold_sha256')]:
 assert sha(RUN/name)==cfg[key],name
fold=read(RUN/'fold_manifest.json');vocab=read(RUN/'vocabulary.json')['tokens']
assert len(rows)==len({r['sample_id'] for r in rows})==96
assert {r['sample_id'] for r in rows}<=set(fold['train_sample_ids'])
assert cfg['classes']==len(vocab)==2289 and vocab[:2]==['<BLANK>','<UNK>']
previous=RUN.parent/'20261020T143609Z_learning_diagnostic01'
original={r['sample_id']:r for r in read(previous/'resolved_manifest.json') if r['internal_role']=='TRAIN'}
assert all({k:v for k,v in r.items() if k!='compact_selection'}==original[r['sample_id']] for r in rows)
assert sha(RUN/'control.py')==cfg['control_script_sha256']
assert read(RUN/'diagnosis_summary.json')['pipeline_valid_for_compact_control']
files=['control.py','control_config.json','subset_manifest.json','subset_coverage.json','vocabulary.json','normalization.npz','fold_manifest.json','PHASE1_FINDINGS.md','diagnosis_summary.json']
snap={}
for name in files:
 p=RUN/name;s=p.stat();snap[str(p)]=dict(sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
with (RUN/'frozen_control_inputs.json').open('x',encoding='utf-8') as f:json.dump(dict(status='PASS',before=snap,subset_manifest_sha256=sha(RUN/'subset_manifest.json')),f,indent=2)
print(json.dumps(dict(status='PASS',frozen_files=len(snap),compact_ids=96,subset_manifest_sha256=sha(RUN/'subset_manifest.json'))))

