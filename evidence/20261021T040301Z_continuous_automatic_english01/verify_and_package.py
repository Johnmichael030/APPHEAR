import hashlib, json, pathlib, shutil, zipfile, xml.etree.ElementTree as ET
root=pathlib.Path.cwd(); task=pathlib.Path('evidence/current_continuous_automatic_task.txt').read_text().strip(); out=pathlib.Path(task)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
checks=[]
for item in json.loads((out/'preservation_before.json').read_text(encoding='utf-8-sig')):
    path=pathlib.Path(item['path']); actual=sha(path)
    checks.append({'path':str(path),'unchanged':actual==item['sha256'],'sha256':actual})
assert all(c['unchanged'] for c in checks),'Frozen artifact changed'
old=out/'before/AppHear-previous.apk'; new=root/'android/app/build/outputs/apk/debug/app-debug.apk'
with zipfile.ZipFile(old) as a, zipfile.ZipFile(new) as b:
    previous={i.filename:i for i in a.infolist() if i.filename.startswith('assets/') and not i.is_dir()}
    current={i.filename:i for i in b.infolist() if i.filename.startswith('assets/') and not i.is_dir()}
    removed=sorted(previous.keys()-current.keys()); added=sorted(current.keys()-previous.keys())
    changed=[name for name in previous.keys()&current.keys() if previous[name].file_size!=current[name].file_size or previous[name].CRC!=current[name].CRC]
    assert not removed and not changed, (removed,changed)
    # Verify exact frozen model bytes, not just ZIP CRCs.
    for name in ['causal_gloss_ctc_fp32.onnx','vocabulary.json','manifest.json']:
        assert b.read('assets/continuous_asl/'+name)==(root/'packages/continuous_asl_research_01/android_assets/continuous_asl'/name).read_bytes()
    for name in ['english.json','policy.json']:
        assert b.read('assets/continuous_asl_output/'+name)==(root/'android/app/src/continuousAsl/assets/continuous_asl_output'/name).read_bytes()
    output_bytes=sum(current[n].file_size for n in added)
    model_bytes=sum(i.file_size for n,i in current.items() if n.startswith('assets/continuous_asl/'))
counts={'tests':0,'failures':0,'errors':0,'skipped':0}
for file in (root/'android/app/build/test-results/testDebugUnitTest').glob('TEST-*.xml'):
    test=ET.parse(file).getroot()
    for k in counts:counts[k]+=int(test.get(k,0))
assert counts['failures']==counts['errors']==0, counts
apk=out/'AppHear-continuous-auto-English-20261021.apk'
if not apk.exists(): shutil.copy2(new,apk)
assert sha(apk)==sha(new)
result={'status':'PARTIAL: engineering checks pass; limited English rules and recognition quality remain experimental','apk':str(apk),'apk_bytes':apk.stat().st_size,'apk_MiB':apk.stat().st_size/1048576,'apk_sha256':sha(apk),'previous_apk_sha256':sha(old),'host_tests':counts,'frozen_source_files_verified':len(checks),'previous_apk_asset_entries_preserved_by_size_and_CRC':len(previous),'changed_existing_assets':changed,'removed_assets':removed,'new_assets':added,'new_output_assets_bytes':output_bytes,'continuous_model_assets_bytes':model_bytes,'device_check':'BLOCKED: adb devices empty on 2026-10-21; no installation or camera capture performed','recognition_accuracy':'UNCHANGED: frozen epoch12 held-out WER95.67%; no new accuracy evaluation or fit','english_coverage':json.loads((out/'english_coverage.json').read_text())}
(out/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'preservation_after.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(result,indent=2))
