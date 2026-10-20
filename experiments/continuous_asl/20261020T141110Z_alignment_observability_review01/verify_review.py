"""Read-only explicit-input verification. No workspace or reserved TEST discovery."""
from pathlib import Path
import csv,hashlib,json
RUN=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(n):return json.loads((RUN/n).read_text(encoding='utf-8-sig'))
before=read('input_files_before.json');assert before==read('input_files_after.json')
for p,e in before.items():
 p=Path(p);s=p.stat()
 assert s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'] and sha(p)==e['sha256'],str(p)
artifacts=0
for line in (RUN/'artifact_hashes.sha256').read_text().splitlines():
 digest,name=line.split('  ',1);assert Path(name).name==name
 assert sha(RUN/name)==digest,name;artifacts+=1
with (RUN/'review_manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==12 and len({r['sample_id'] for r in rows})==12
assert all(sum(r['group']==g for r in rows)==4 for g in 'ABC')
assert all(r['review_status']=='NOT_REVIEWED_NO_QUALIFIED_REVIEWER' and not r['visual_flags'] for r in rows)
integrity=read('integrity_results.json');assert integrity['status']=='PASS' and not integrity['changed_files']
pilot=RUN.parent/'20261020T133148Z_source_transfer_pilot01';audit=RUN.parent/'20261020T135119Z_read_only_failure_audit01'
expected_pilot={p for p in before if Path(p).is_relative_to(pilot)}
expected_audit={p for p in before if Path(p).is_relative_to(audit)}
assert {str(p) for p in pilot.rglob('*') if p.is_file()}==expected_pilot
assert {str(p) for p in audit.iterdir() if p.is_file()}==expected_audit
verdict='PARTIAL — computational evidence improved, but qualified visual/annotation review remains unresolved'
assert (RUN/'ALIGNMENT_OBSERVABILITY_REPORT.md').read_text(encoding='utf-8').rstrip().endswith(verdict)
print(json.dumps(dict(integrity='PASS',existing_files_unchanged=len(before),new_artifacts_verified=artifacts,review_rows=12,reserved_TEST_access=False,training_or_model_inference=False),indent=2))

