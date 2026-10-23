"""Read-only packaged vocabulary integrity check, using Python's standard library.

Engineering integrity is not linguistic validation or Android rendering evidence.
Run from any directory: python -B tools/verify-web-asl-vocabulary.py
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ANDROID = Path(__file__).resolve().parents[1]
ASSETS = ANDROID/'app/src/aslWebVocabulary/assets'
ORIGINAL = ANDROID/'app/src/aslOriginalWords/assets'
PATH = 'asl_animations/web_vocabulary_200/avatar.glb'


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def glb(payload: bytes) -> tuple[dict, bytes]:
    assert struct.unpack_from('<III', payload) == (0x46546c67, 2, len(payload))
    size, kind = struct.unpack_from('<II', payload, 12)
    assert kind == 0x4e4f534a
    count, kind = struct.unpack_from('<II', payload, 20+size)
    assert kind == 0x004e4942
    return json.loads(payload[20:20+size]), payload[28+size:28+size+count]


def values(g: dict, binary: bytes, index: int) -> list[tuple[float, ...]]:
    item = g['accessors'][index]; view = g['bufferViews'][item['bufferView']]
    assert item['componentType'] == 5126 and 'sparse' not in item
    width = {'SCALAR': 1, 'VEC4': 4}[item['type']]
    offset = view.get('byteOffset', 0)+item.get('byteOffset', 0)
    stride = view.get('byteStride', width*4)
    return [struct.unpack_from('<'+'f'*width, binary, offset+i*stride) for i in range(item['count'])]


def main() -> None:
    metadata = (ASSETS/'asl_animations/web_vocabulary_200/manifest.json').read_bytes()
    manifest = json.loads(metadata)
    payload = (ASSETS/PATH).read_bytes()
    contract = (ANDROID/'app/src/main/java/com/example/apphear/domain/texttoasl/AslWebVocabularyExpansion.kt').read_text()
    assert f'const val SHA256 = "{sha(payload)}"' in contract
    assert f'const val MANIFEST_SHA256 = "{sha(metadata)}"' in contract
    assert manifest['sha256'] == sha(payload)
    assert manifest['language'] == 'ASL' and manifest['qualifiedAslValidations'] == 0
    assert manifest['newLexicalEntries'] == 200 and manifest['totalLexicalEntries'] == 432
    baseline = (ORIGINAL/'asl_animations/original_words_v1/fingerspelling-avatar.glb').read_bytes()
    assert sha(baseline) == manifest['baselineSha256'] == 'e9079308b6780e180528e4b893e64bdb77e46b33b6ad2cd99355bec03df57b18'
    g, binary = glb(payload); old, old_binary = glb(baseline)
    for key in ['nodes', 'skins', 'meshes', 'materials', 'textures', 'images', 'scenes']:
        assert g[key] == old[key], key
    assert len(g['animations']) == 458
    assert g['animations'][:258] == old['animations']
    assert binary[:old['buffers'][0]['byteLength']] == old_binary[:old['buffers'][0]['byteLength']]
    entries = manifest['entries']; assert len(entries) == 200
    for field in ['id', 'animationName', 'sourceSha256', 'motionSha256']:
        assert len({e[field] for e in entries}) == 200
    actions = {a['name']: a for a in g['animations']}
    channels = 0; keys = 0; worst = 0.
    for entry in entries:
        assert re.fullmatch('[a-z]+', entry['id'])
        assert entry['sourceLanguage'] == 'ASL' and entry['license'] == 'CC-BY-4.0'
        assert entry['linguisticValidationStatus'] == 'UNREVIEWED'
        assert entry['english'] == [entry['id']]
        assert entry['animationName'] == f"SIGN_{entry['id'].upper()}_3DLEX_ASL_PREVIEW"
        assert entry['sourceArchiveMember'] == f"ANIM_GLB/{entry['id']}.glb"
        for field in ['sourceSha256', 'motionSha256']:
            assert re.fullmatch('[a-f0-9]{64}', entry[field])
        action = actions[entry['animationName']]
        for channel in action['channels']:
            sampler = action['samplers'][channel['sampler']]
            assert channel['target']['path'] == 'rotation' and sampler['interpolation'] == 'LINEAR'
            times = [x[0] for x in values(g, binary, sampler['input'])]
            rotations = values(g, binary, sampler['output'])
            assert len(times) == len(rotations) and times[0] == 0
            assert all(math.isfinite(x) for x in times) and all(b > a for a, b in zip(times, times[1:]))
            assert abs(times[-1]*1000-entry['durationMs']) <= 1
            assert all(all(math.isfinite(x) for x in q) and abs(sum(x*x for x in q)-1) < 2e-6 for q in rotations)
            for a, b in zip(rotations[1:-2], rotations[2:-1]):
                angle = math.degrees(2*math.acos(min(1, abs(sum(x*y for x, y in zip(a, b))))))
                worst = max(worst, angle)
                assert angle < 90, (entry['id'], angle)
            channels += 1; keys += len(rotations)
    print(json.dumps(dict(status='PASS_OFFLINE_INTEGRITY',lexicalEntries=432,newEntries=200,actions=458,
        preservedActions=258,checkedChannels=channels,checkedRotationKeys=keys,
        maximumInnerStepDegrees=worst,bundleBytes=len(payload),bundleSha256=sha(payload),
        qualifiedAslValidations=0,phoneRendering='NOT_VERIFIED')))


if __name__ == '__main__':
    main()
