"""Read-only integrity/package checks; writes a new evidence report, never trains."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evidence/continuous_asl_android_20261021'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    before = json.loads((OUT/'preservation_before.json').read_text())
    for name, expected in before['existing_assets'].items():
        assert sha(ROOT/name) == expected, name
    package = ROOT/'packages/continuous_asl_research_01'
    parity = json.loads((package/'export_parity.json').read_text())
    assert parity['status'] == 'PASS' and parity['synthetic_only']
    checkpoint = ROOT/'experiments/continuous_asl/20261020T143609Z_learning_diagnostic01/candidate_checkpoints/epoch_012.pt'
    assert sha(checkpoint) == parity['checkpoint_sha256']
    frozen = ROOT/'packages/fingerspelling_continuous_v2_1_01'
    inventory = json.loads((frozen/'package_inventory.json').read_text())
    for f in inventory['files']:
        assert sha(frozen/f['path']) == f['sha256'], f['path']
    suites = [ET.parse(p).getroot() for p in (ROOT/'android/app/build/test-results/testDebugUnitTest').glob('*.xml')]
    tests = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    assert tests['tests'] >= 284 and tests['failures'] == tests['errors'] == 0
    specific = [s for s in suites if s.attrib['name'].endswith('ContinuousAslCoreTest')]
    assert len(specific) == 1 and specific[0].attrib['tests'] == '9'
    log = (OUT/'build_final.log').read_text(encoding='utf-8-sig', errors='replace')
    assert 'BUILD SUCCESSFUL' in log
    apk = ROOT/'android/app/build/outputs/apk/debug/app-debug.apk'
    test_apk = ROOT/'android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk'
    assert apk.is_file() and test_apk.is_file()
    names = ['causal_gloss_ctc_fp32.onnx', 'manifest.json', 'vocabulary.json']
    entries = []
    with zipfile.ZipFile(apk) as z:
        found = sorted(i.filename for i in z.infolist() if i.filename.startswith('assets/continuous_asl/'))
        assert found == sorted('assets/continuous_asl/'+n for n in names)
        for name in names:
            member = 'assets/continuous_asl/'+name
            info = z.getinfo(member)
            source = package/'android_assets/continuous_asl'/name
            assert hashlib.sha256(z.read(member)).hexdigest() == sha(source)
            entries.append({'name': member, 'bytes': info.file_size, 'apk_payload_bytes': info.compress_size})
        assert not any('synthetic_features' in n or 'synthetic_logits' in n for n in z.namelist())
    report = {
        'status': 'PASS', 'integration': 'EXPERIMENTAL_OPT_IN_DEBUG_ONLY',
        'unit_tests': tests, 'new_continuous_tests': 9,
        'debug_and_instrumentation_apks_built': True,
        'device_tests': 'NOT_RUN', 'installed_to_device': False,
        'recognition_quality': 'NOT_ACCEPTED; saved heldout WER 95.6701%',
        'export_parity': parity,
        'preserved_existing_source_asset_files': len(before['existing_assets']),
        'preserved_fingerspelling_package_files': len(inventory['files']),
        'preserved_source_checkpoint_sha256': sha(checkpoint),
        'runtime_entries': entries,
        'runtime_asset_bytes': sum(r['bytes'] for r in entries),
        'runtime_apk_payload_bytes': sum(r['apk_payload_bytes'] for r in entries),
        'apk': str(apk), 'apk_bytes': apk.stat().st_size, 'apk_sha256': sha(apk),
        'test_apk': str(test_apk), 'test_apk_sha256': sha(test_apk),
        'new_training': False, 'TEST_used': False, 'production_promoted': False,
    }
    (OUT/'VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    md = f'''# Continuous-ASL Android integration result

**PASS host build, tests, export parity and asset integrity; PARTIAL device readiness; BLOCKED recognition-quality/production acceptance.**

The user-requested **CONTINUOUS WORD · experimental** mode is present in Settings in the opt-in debug APK. It uses the selected full-task checkpoint through an independent small FP32 inference export. Candidate01 observations and the existing 378-feature builder feed bounded causal CTC drafts. Explicit review/edit is required before text is used or spoken. WORD/FINGERSPELL remain separate; default/release flags are off.

- JVM tests: **{tests['tests']}**, failures **{tests['failures']}**, errors **{tests['errors']}**; includes nine new continuous tests and existing feature/owner regressions.
- Debug and Android-test APK builds: **PASS**. Final log: `build_final.log`.
- Synthetic desktop PyTorch/ONNX and causal-prefix parity: **PASS**. This is numerical evidence, not model evaluation.
- Android native parity test: compiled, **not run**. No phone installation, application-data deletion, camera recording or participant processing was performed.
- Existing source asset files unchanged: **{len(before['existing_assets'])}**. Frozen fingerspelling package entries unchanged: **{len(inventory['files'])}**. Selected continuous source checkpoint SHA unchanged.
- No fitting, data acquisition, reserved TEST use or avatar/model replacement.

## Size

Only model/vocabulary/manifest are packaged for this feature: **{report['runtime_asset_bytes']:,} bytes ({report['runtime_asset_bytes']/1024**2:.3f} MiB)**. Their compressed/stored APK payload is **{report['runtime_apk_payload_bytes']:,} bytes ({report['runtime_apk_payload_bytes']/1024**2:.3f} MiB)**, excluding ZIP directory/alignment/code overhead. Synthetic fixtures are only in the test APK.

Training checkpoint size is {checkpoint.stat().st_size:,} bytes; the runtime asset package is {100*(1-report['runtime_asset_bytes']/checkpoint.stat().st_size):.2f}% smaller. Optimizer/RNG/data/logs are not shipped. No new inference dependency is added. Extra quantization was unnecessary and was not performed.

Complete debug APK: **{report['apk_bytes']:,} bytes ({report['apk_bytes']/1024**2:.2f} MiB)**. Most existing payload is avatar animation assets; no avatar compression or removal occurred. The 0.776-MiB figure describes this recognition addition, not the whole application.

APK: `{apk}`

APK SHA256: `{report['apk_sha256']}`

## Review

| Layer | Result | Remaining limitation |
|---|---|---|
| Requested Settings + recognition integration | PASS | Available in this built opt-in debug APK; not yet installed. |
| Exclusive owner, review, lifecycle and input guards | PASS host tests/source review | Physical-device switching/backpressure/TTS smoke pending. |
| Model export and contract | PASS numerical/host evidence | Native Android synthetic parity test awaits a device/emulator. |
| Product accuracy, no-sign behavior, cadence/latency | BLOCKED | Full-task held-out WER remains 95.67%; no accepted streaming/unseen-signer evidence. |

No outstanding critical host-build issue was found. The explicit review gate does not make the recognizer accurate. Recommendation: **local experimental engineering smoke only; do not promote**. Device parity and F22 behavior/timing must be checked before wider use. No permission to upload new participant data or to publish restricted models/data is inferred.

Implementation/design/size details: `docs/continuous-asl-android-research-20261021.md`.
Rollback: disable `continuousAsl` when building, then choose WORD. A stale continuous preference resolves to WORD in a build without this flag. No existing model or GLB needs restoration.

One pre-existing Windows-1252 ellipsis byte in RecognitionActivity was normalized to UTF-8 for source editing; its original bytes are in `before/`. An initial unsigned PowerShell-wrapper invocation was blocked by the machine's script policy. The equivalent direct Gradle command succeeded without changing persistent execution policy.
'''
    (OUT/'INTEGRATION_REPORT.md').write_text(md, encoding='utf-8')
    files = [OUT/'VALIDATION.json', OUT/'INTEGRATION_REPORT.md', OUT/'build_final.log',
             ROOT/'docs/continuous-asl-android-research-20261021.md',
             ROOT/'android/app/src/main/java/com/example/apphear/RecognitionActivity.kt']
    files += sorted((ROOT/'android/app/src/main/java/com/example/apphear/continuousasl').glob('*.kt'))
    (OUT/'artifact_hashes.sha256').write_text(''.join(f'{sha(p)}  {p}\n' for p in files), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['export_parity','runtime_entries']}, indent=2))


if __name__ == '__main__':
    main()
