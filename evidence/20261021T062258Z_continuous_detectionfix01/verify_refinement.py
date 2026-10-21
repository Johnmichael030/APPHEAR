"""Read-only validation of the completed refinement evidence and protected model/assets."""
import hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()

v=json.loads((HERE/'VALIDATION.json').read_text())
assert sha(v['apk'])==v['apk_sha256'],'APK_HASH'
assert v['host_tests']=={'tests':301,'failures':0,'errors':0,'skipped':0}
assert v['training_runs']==v['new_checkpoints']==v['recognition_classes_added']==0
assert not v['model_promoted'] and not v['reserved_TEST_evaluated']
for item in json.loads((HERE/'preservation_before.json').read_text(encoding='utf-8-sig')):
 assert sha(item['path'])==item['sha256'],('PROTECTED_FILE',item['path'])
checked=0
for line in (HERE/'artifact_hashes.sha256').read_text(encoding='utf-8-sig').splitlines():
 expected,name=line.split('  ',1); assert sha(name)==expected,('ARTIFACT',name);checked+=1
m=json.loads((HERE/'continuity_replay/safety_before_after/RESULTS.json').read_text())['metrics']
assert m['OLD_flush']['WER']==m['NEW_discard']['WER']
assert m['OLD_flush']['nohand_token_publications']==58 and m['NEW_discard']['nohand_token_publications']==0
print(json.dumps({'status':'PASS','artifact_hashes':checked,'protected_files':v['frozen_source_files'],'host_tests':301,'APK_sha256':v['apk_sha256'],'quality':'PARTIAL_UNCHANGED_HELD_WER','training_runs':0,'TEST_used':False,'installed':False},indent=2))
