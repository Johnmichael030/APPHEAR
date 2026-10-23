"""Read-only integrity check for the five-word ASL preview substitutions."""
import importlib.util
import json
import math
from pathlib import Path

spec=importlib.util.spec_from_file_location('verify',Path(__file__).with_name('verify-web-asl-vocabulary.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
ANDROID=Path(__file__).resolve().parents[1]
root=ANDROID/'app/src/aslWordMotionFixes/assets'
metadata=(root/'asl_animations/word_motion_fixes/manifest.json').read_bytes()
manifest=json.loads(metadata);payload=(root/manifest['asset']).read_bytes()
contract=(ANDROID/'app/src/main/java/com/example/apphear/domain/texttoasl/AslWordMotionFixes.kt').read_text()
assert f'const val SHA256 = "{v.sha(payload)}"' in contract
assert f'const val MANIFEST_SHA256 = "{v.sha(metadata)}"' in contract
assert v.sha(payload)==manifest['sha256']
assert manifest['newLexicalEntries']==0 and manifest['totalLexicalEntries']==432
assert manifest['replacementMotions']==5 and manifest['qualifiedAslValidations']==0
old_payload=(ANDROID/'app/src/aslWebVocabulary/assets'/v.PATH).read_bytes()
assert v.sha(old_payload)==manifest['baselineSha256']=='6b4dbc4fce99ff6d396e3bc8bf2c792adef4f1fe1deb7bfa709306096084e9cb'
old,old_binary=v.glb(old_payload);g,binary=v.glb(payload)
assert len(g['animations'])==463 and g['animations'][:458]==old['animations']
for key in ['nodes','skins','meshes','materials','textures','images','scenes']:assert g[key]==old[key]
assert binary[:old['buffers'][0]['byteLength']]==old_binary[:old['buffers'][0]['byteLength']]
assert {e['replacesId'] for e in manifest['overrides']}=={'why','where','you','your_yours','today'}
actions={a['name']:a for a in g['animations']};keys=0;worst=0
for entry in manifest['overrides']:
    assert entry['sourceLanguage']=='ASL' and entry['license']=='CC-BY-4.0' and entry['linguisticValidationStatus']=='UNREVIEWED'
    for channel in actions[entry['animationName']]['channels']:
        sample=actions[entry['animationName']]['samplers'][channel['sampler']]
        assert channel['target']['path']=='rotation' and sample['interpolation']=='LINEAR'
        times=[x[0] for x in v.values(g,binary,sample['input'])];rotations=v.values(g,binary,sample['output'])
        assert len(times)==len(rotations) and times[0]==0 and all(b>a for a,b in zip(times,times[1:]))
        assert abs(times[-1]*1000-entry['durationMs'])<=1
        assert all(all(math.isfinite(x) for x in q) and abs(sum(x*x for x in q)-1)<2e-6 for q in rotations)
        for a,b in zip(rotations[1:-2],rotations[2:-1]):
            angle=math.degrees(2*math.acos(min(1,abs(sum(x*y for x,y in zip(a,b))))))
            assert angle<90;worst=max(worst,angle)
        keys+=len(times)
print(json.dumps(dict(status='PASS_OFFLINE_INTEGRITY',lexicalEntries=432,replacedMotions=5,
    preservedActions=458,totalActions=463,checkedNewRotationKeys=keys,maxInnerStepDegrees=worst,
    bundleBytes=len(payload),bundleSha256=v.sha(payload),phoneRendering='NOT_RUN',qualifiedAslValidations=0)))
