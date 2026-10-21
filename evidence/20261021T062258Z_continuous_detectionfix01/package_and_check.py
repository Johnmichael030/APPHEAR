import difflib, hashlib, json, pathlib, shutil, zipfile, xml.etree.ElementTree as ET
ROOT=pathlib.Path.cwd();OUT=pathlib.Path('evidence/current_continuous_detection_task.txt').read_text().strip();OUT=pathlib.Path(OUT)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()
old=OUT/'before/AppHear-previous.apk'; built=ROOT/'android/app/build/outputs/apk/debug/app-debug.apk'; target=OUT/'AppHear-continuous-refined-20261021.apk'
if target.exists():raise RuntimeError('DELIVERY_ALREADY_EXISTS')
checks=[]
for item in json.loads((OUT/'preservation_before.json').read_text(encoding='utf-8-sig')):
 actual=sha(item['path']);checks.append({'path':item['path'],'sha256':actual,'unchanged':actual==item['sha256']})
assert all(c['unchanged'] for c in checks),'FROZEN_FILE_CHANGED'
with zipfile.ZipFile(old) as previous,zipfile.ZipFile(built) as current:
 a={i.filename:i for i in previous.infolist() if i.filename.startswith('assets/') and not i.is_dir()}
 b={i.filename:i for i in current.infolist() if i.filename.startswith('assets/') and not i.is_dir()}
 assert a.keys()==b.keys(),'ASSET_ADDITION_OR_REMOVAL'
 assert all(a[n].file_size==b[n].file_size and a[n].CRC==b[n].CRC for n in a),'OLD_ASSET_CHANGED'
 for n in a:
  if n.startswith('assets/continuous_asl/') or n.startswith('assets/continuous_asl_output/'):
   assert previous.read(n)==current.read(n),'CONTINUOUS_ASSET_BYTES_CHANGED'
counts=dict(tests=0,failures=0,errors=0,skipped=0)
for p in (ROOT/'android/app/build/test-results/testDebugUnitTest').glob('TEST-*.xml'):
 s=ET.parse(p).getroot()
 for k in counts:counts[k]+=int(s.get(k,0))
assert counts['tests']==301 and counts['failures']==counts['errors']==counts['skipped']==0
shutil.copy2(built,target);assert sha(target)==sha(built)
changed=['android/app/src/main/java/com/example/apphear/continuousasl/ContinuousAslCore.kt','android/app/src/main/java/com/example/apphear/RecognitionActivity.kt','android/app/src/test/java/com/example/apphear/continuousasl/ContinuousAslCoreTest.kt','android/app/build.gradle.kts']
diff=[]
for name in changed:
 source=ROOT/name; after=OUT/'after'/name;after.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,after)
 before=OUT/'before'/name
 diff.extend(difflib.unified_diff(before.read_text(encoding='utf-8-sig').splitlines(True),source.read_text(encoding='utf-8-sig').splitlines(True),fromfile='before/'+name,tofile='after/'+name))
(OUT/'source_changes.diff').write_text(''.join(diff),encoding='utf-8')
result={'status':'PARTIAL: software bugs fixed; recognition quality not solved','apk':str(target),'apk_bytes':target.stat().st_size,'apk_MiB':target.stat().st_size/1048576,'apk_sha256':sha(target),'previous_apk_sha256':sha(old),'version':'1.0-continuous-refined-20261021','application_id':'com.example.apphear','host_tests':counts,'frozen_source_files':len(checks),'previous_APK_asset_entries_preserved_size_and_CRC':len(a),'exact_continuous_asset_bytes_unchanged':True,'source_files_changed':changed,'training_runs':0,'new_checkpoints':0,'model_promoted':False,'recognition_classes_added':0,'reserved_TEST_evaluated':False,'hardware_check':'BLOCKED: ADB device list empty; no install/camera use','development_replay':'Same known signers; family-held-out80; already development-exposed; WER95.6701% before and after. No live/unseen-signer accuracy claim.'}
(OUT/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'preservation_after.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(result,indent=2))
