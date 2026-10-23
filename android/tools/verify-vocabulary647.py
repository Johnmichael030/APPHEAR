"""Read-only integrity check for the 647-entry ASL preview. Standard library only."""
import importlib.util
import json
import math
from pathlib import Path

spec=importlib.util.spec_from_file_location('v',Path(__file__).with_name('verify-web-asl-vocabulary.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
ANDROID=Path(__file__).resolve().parents[1]
root=ANDROID/'app/src/aslVocabulary647/assets'
metadata=(root/'asl_animations/vocabulary_647/manifest.json').read_bytes();manifest=json.loads(metadata)
payload=(root/manifest['asset']).read_bytes()
contract=(ANDROID/'app/src/main/java/com/example/apphear/domain/texttoasl/AslVocabulary647.kt').read_text()
assert f'const val SHA256 = "{v.sha(payload)}"' in contract
assert f'const val MANIFEST_SHA256 = "{v.sha(metadata)}"' in contract
assert v.sha(payload)==manifest['sha256'] and manifest['totalLexicalEntries']==647 and manifest['newLexicalEntries']==215
assert manifest['qualifiedAslValidations']==0 and manifest['newAliases']==51 and manifest['language']=='ASL'
base_path=ANDROID/'app/src/aslWordMotionFixes/assets/asl_animations/word_motion_fixes/avatar.glb'
baseline=base_path.read_bytes();assert v.sha(baseline)==manifest['baselineSha256']=='244ba8daf6371124dac2da6a985fff0dfaead8ee92e672fcfe93ebf5d5e24616'
old,old_binary=v.glb(baseline);g,binary=v.glb(payload)
assert len(g['animations'])==678 and g['animations'][:463]==old['animations']
assert binary[:old['buffers'][0]['byteLength']]==old_binary[:old['buffers'][0]['byteLength']]
for key in ['nodes','skins','meshes','materials','textures','images','scenes']:assert g[key]==old[key]
entries=manifest['entries'];assert len(entries)==215
for field in ['id','sourceSha256','motionSha256','animationName']:assert len({e[field] for e in entries})==215
assert sum(len(e['english']) for e in manifest['aliases'])==51
actions={a['name']:a for a in g['animations']};keys=0;worst=0
for entry in entries:
    assert entry['sourceLanguage']=='ASL' and entry['license']=='CC-BY-4.0' and entry['linguisticValidationStatus']=='UNREVIEWED'
    assert entry['english']==[entry['id']]
    action=actions[entry['animationName']]
    for channel in action['channels']:
        sample=action['samplers'][channel['sampler']]
        assert channel['target']['path']=='rotation' and sample['interpolation']=='LINEAR'
        times=[r[0] for r in v.values(g,binary,sample['input'])];rotations=v.values(g,binary,sample['output'])
        assert len(times)==len(rotations) and times[0]==0 and all(b>a for a,b in zip(times,times[1:]))
        assert abs(times[-1]*1000-entry['durationMs'])<=1
        assert all(all(math.isfinite(x) for x in q) and abs(sum(x*x for x in q)-1)<2e-6 for q in rotations)
        for a,b in zip(rotations[1:-2],rotations[2:-1]):
            angle=math.degrees(2*math.acos(min(1,abs(sum(x*y for x,y in zip(a,b))))))
            assert angle<90;worst=max(worst,angle)
        keys+=len(times)
print(json.dumps(dict(status='PASS_OFFLINE_INTEGRITY',lexicalEntries=647,newEntries=215,newAliases=51,
    preservedActions=463,totalActions=678,checkedNewRotationKeys=keys,maxInnerStepDegrees=worst,
    fiveWordFixesPreserved=True,bundleBytes=len(payload),bundleSha256=v.sha(payload),
    qualifiedAslValidations=0,phoneRendering='NOT_RUN')))
