"""Read-only continuation audit. Writes only to a new --out child; never trains."""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[2]
PREVIOUS = RUN.parent / '20261019T155428Z_cslr01'
sys.path.insert(0, str(ROOT / 'tools/continuous_asl'))
from archive_safety import inspect_archive

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')

def command(args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, errors='replace')
    return {'command': args, 'returncode': p.returncode,
            'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}

def npz_metadata(path):
    result = {'path': str(path), 'exists': path.is_file(), 'inspection': 'NPY headers and scalar schema only; no feature/test payload loaded'}
    if not path.is_file():
        return result
    result['bytes'] = path.stat().st_size
    headers = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.npy'):
                continue
            with z.open(name) as f:
                magic = f.read(8)
                if magic[:6] != b'\x93NUMPY':
                    raise ValueError('INVALID_NPY_HEADER')
                nbytes = 2 if magic[6] == 1 else 4
                size = int.from_bytes(f.read(nbytes), 'little')
                if size > 65536:
                    raise ValueError('NPY_HEADER_BUDGET')
                header = ast.literal_eval(f.read(size).decode('latin1'))
                headers[name] = header
                if name in ('schema_version.npy', 'feature_schema.npy'):
                    raw = f.read(1024)
                    dtype = str(header['descr'])
                    if 'U' in dtype:
                        headers[name]['scalar_value'] = raw.decode('utf-32-le' if dtype.startswith('<') else 'utf-32-be').rstrip('\0')
                    elif 'S' in dtype:
                        headers[name]['scalar_value'] = raw.decode('utf-8').rstrip('\0')
    result['members'] = headers
    return result

def classify(text):
    if re.search(r"No module named ['\"]mediapipe", text):
        return 'missing_mediapipe', 'explicit missing dependency'
    if 'FileNotFoundError' in text or '[Errno 2] No such file' in text:
        return 'missing_file_or_fixture', 'explicit missing path; not proof failure predates research'
    if 'research sandbox' in text.lower() or 'sandbox denied' in text.lower() or 'REGRESSION_SANDBOX' in text or 'network forbidden' in text.lower():
        return 'sandbox_restriction', 'explicit protected-copy restriction'
    if 'PermissionError' in text:
        return 'permission_or_sandbox', 'access denied; cause needs targeted inspection'
    if re.search(r'WinError (123|206)|\[Errno 22\]|invalid argument', text, re.I):
        return 'path_or_platform_error', 'explicit OS/path error; not automatic product exemption'
    if 'ModuleNotFoundError' in text or 'ImportError' in text:
        return 'other_import_dependency', 'explicit import failure'
    if 'TypeError' in text or 'AttributeError' in text:
        return 'api_or_type_mismatch', 'traceback symptom; implementation/environment root cause unresolved'
    return 'unresolved_assertion_or_validation', 'needs isolated investigation; not claimed pre-existing'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != HERE or out.exists():
        raise ValueError('OUTPUT_MUST_BE_NEW_DIRECT_CHILD_OF_CONTINUATION')
    out.mkdir()
    now = datetime.now(timezone.utc).isoformat()
    runtime = {
        'utc': now, 'repository': str(ROOT), 'run': str(RUN),
        'git': command(['git', 'status', '--short']),
        'gpu': command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader']),
        'processes': command(['powershell', '-NoProfile', '-Command', "Get-Process | Where-Object {$_.ProcessName -match 'python|ffmpeg|curl|wget'} | Select-Object Id,ProcessName,CPU,Path | ConvertTo-Json"]),
        'disk_free_bytes': shutil.disk_usage(ROOT).free,
        'cim_initial_attempt': 'Get-CimInstance Win32_Process: Access to a CIM resource was not available to the client; Get-Process fallback used',
        'actions': 'No training, download, test-set evaluation, production or Android action.'}
    write(out / 'workspace.json', runtime)
    receipt = read(RUN / 'data/raw/how2sign/val_rgb_front_clips.zip.receipt.json')
    plan = read(RUN / 'how2sign_currentpage_plan.json')['items'][0]
    archive = RUN / 'data/raw/how2sign/val_rgb_front_clips.zip'
    before = archive.stat()
    digest = sha(archive)
    cap = 2147483648
    verified_id = '1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO'
    if plan['max_bytes'] != cap or before.st_size > cap or verified_id not in plan['url'] or verified_id not in receipt['source_url']:
        raise ValueError('SOURCE_OR_DOWNLOAD_CAP_MISMATCH')
    if digest != receipt['sha256'] or before.st_size != receipt['bytes']:
        raise ValueError('ARCHIVE_RECEIPT_MISMATCH')
    rows = inspect_archive(archive, max_bytes=3221225472, max_entries=10000)
    assert archive.stat().st_mtime_ns == before.st_mtime_ns
    source = {
        'status': 'PASS', 'verified_utc': now,
        'authority': 'https://how2sign.github.io/#download',
        'official_link_context': 'Green Screen RGB clips* (frontal view): Validation (1.7G)',
        'verification_method': 'Opened official page and clicked its actual numbered link 10 in this continuation; Drive destination titled val_rgb_front_clips.zip.',
        'destination': 'https://drive.google.com/file/d/' + verified_id + '/view?usp=sharing',
        'file_id': verified_id, 'download_url': receipt['source_url'],
        'archive': str(archive), 'bytes': before.st_size, 'download_cap_bytes': cap,
        'sha256': digest, 'official_published_hash_available': False,
        'zip_crc_all_members': 'PASS', 'safe_paths_links_expansion_checks': 'PASS',
        'archive_members': len(rows), 'video_members': sum(r['path'].lower().endswith('.mp4') for r in rows),
        'expanded_bytes': sum(r['bytes'] for r in rows),
        'decision': 'REUSE_VERIFIED_ARCHIVE; zero bytes newly downloaded or extracted',
        'previous_failed_attempt': read(RUN / 'logs/download_how2sign_val_frontal_pilot_archive_attempt1.json'),
        'keypoint_fallback': {
            'official_destination_verified_by_click': 'https://drive.google.com/file/d/1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w/view?usp=sharing',
            'title': 'val_2D_keypoints.tar.gz',
            'status': 'NOT_REQUIRED_RGB_SUCCEEDED',
            'payload_access_and_schema': 'NOT_TESTED; no keypoint archive acquired; link resolution is not payload-access proof or compatibility proof'},
        'annotation_audit': read(RUN / 'how2sign_annotation_details.json'),
        'scope': 'Fresh full ZIP CRC/hash check, not full archive video decoding. Existing eight-clip decode evidence reused.',
        'unresolved_download_limitations': 'No local RGB acquisition blocker. Existing archive success does not guarantee future remote availability. No publisher checksum found.'}
    write(out / 'how2sign_verification.json', source)
    print(json.dumps({'source': source['status'], 'bytes': before.st_size, 'zip_members': len(rows), 'video_members': source['video_members']}), flush=True)

    failures = []
    counts = Counter()
    xml_path = RUN / 'logs/full_regression_final.xml'
    for case in ET.parse(xml_path).getroot().iter('testcase'):
        outcome = 'passed'
        node = case.find('failure')
        if node is not None:
            outcome = 'failed'
        else:
            node = case.find('error')
            if node is not None:
                outcome = 'error'
            elif case.find('skipped') is not None:
                outcome = 'skipped'
        counts[outcome] += 1
        if outcome in ('failed', 'error'):
            text = node.text or ''
            category, confidence = classify(text)
            lines = [line for line in text.splitlines() if line.strip()]
            failures.append({'test': case.get('classname', '') + '::' + case.get('name', ''),
                             'outcome': outcome, 'category': category, 'interpretation': confidence,
                             'message': node.get('message', ''), 'traceback_tail': '\n'.join(lines[-7:])[-1800:]})
    write(out / 'regression_classification.json', {
        'status': 'PARTIAL_NOT_GREEN', 'source': str(xml_path), 'source_sha256': sha(xml_path),
        'counts': dict(counts), 'failure_categories': dict(Counter(f['category'] for f in failures)),
        'classification_method': 'Conservative traceback signatures, not causal proof; no full suite rerun.',
        'failures': failures})
    print(json.dumps({'regressions': dict(counts), 'categories': dict(Counter(f['category'] for f in failures))}), flush=True)

    base = 'artifacts/recognition_upgrade/perception_vnext/recognition_utility_236_01/'
    required = [
        base + 's2_r2_phone_temporal_supervision_01/p1_temporal_review_01/p1_reviewed_intervals.csv',
        base + 's2_r2_phone_temporal_supervision_01/signer_governance_audit.json',
        base + 's2_r3_negative_raw_01',
        base + 'phone_repeated_recognition_integration_01/device_capture/session_20260920T145710Z/observations.jsonl',
        base + 'phone_new_sign_admission_and_active_acquisition_validation_01/device_capture_final/session_20260921T102840Z/observations.jsonl',
        'scratchpad/e6_product_core_collection/raw/nosign/signs',
        'scratchpad/e2e_c0_capture/raw/p1',
        'evidence/recognition_v3_stage2_device_20261015/human_boundaries.analysis.json',
        'android/app/src/test/resources/recognition_v3_stage2/recorded_transition.tsv']
    paths = []
    for root in (ROOT, Path('D:/AppHear-claude-s2-r3-eval-suite')):
        for name in required:
            p = root / name
            item = {'path': str(p), 'exists': p.exists()}
            if p.is_file():
                item.update(bytes=p.stat().st_size, sha256=sha(p))
            paths.append(item)
    features = [npz_metadata(Path('D:/derived/derived/apphear3') / name) for name in (
        'features_streaming_conversational_fps/train.npz',
        'features_streaming_recorded_local/recorded.npz')]
    audit_csv = Path('D:/classmate_supplementary_dataset_package/reports/audit_manifest.csv')
    with audit_csv.open(encoding='utf-8-sig', newline='') as f:
        supplementary = list(csv.DictReader(f))
    available = sum(Path(row['video']).is_file() for row in supplementary)
    labels = Counter(row['schema'] for row in supplementary)
    write(out / 'data_inventory.json', {
        'status': 'PARTIAL', 'paths': paths, 'legacy_feature_headers': features,
        'supplementary_manifest': str(audit_csv), 'manifest_sha256': sha(audit_csv),
        'supplementary_rows': len(supplementary), 'referenced_video_paths_existing': available,
        'supplementary_schema_counts': dict(labels),
        'supplementary_findings': 'Clip-level ASL labels and signer IDs; not natural sequence gloss/boundary annotations. Presence does not establish consent for new streaming evaluation.',
        'scope': 'Current repository, duplicate transfer copy, D:/derived and existing supplementary package. Unrelated recordings only directory/filename inventoried; no media read or assumed consent. No exhaustive disk-wide absence claim.'})
    print(json.dumps({'metadata_inventory': 'complete', 'supplementary_videos_existing': available}), flush=True)

    # Verify frozen outputs and protected originals into NEW evidence, without calling mutating legacy verifiers.
    records = read(RUN / 'protected_sources.json') + read(RUN / 'external_inputs_protected.json')
    bad, mtimes = [], []
    for item in records:
        p = Path(item['path'])
        if not p.is_file() or sha(p) != item['sha256']:
            bad.append(str(p))
        elif p.stat().st_mtime_ns != item['mtime_ns']:
            mtimes.append(str(p))
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    historical = []
    original = read(PREVIOUS / 'preservation_baseline.json')['files']
    for item in original:
        whole, fraction = item['last_write_utc'].rstrip('Z').split('.')
        delta = datetime.fromisoformat(whole).replace(tzinfo=timezone.utc) - epoch
        expected = (delta.days * 86400 + delta.seconds) * 1000000000 + int(fraction.ljust(9, '0'))
        if (ROOT / item['path']).stat().st_mtime_ns != expected:
            historical.append(item['path'])
    manifest = read(RUN / 'artifact_manifest.final.json')
    frozen_bad = []
    for item in manifest['artifacts']:
        p = (ROOT if item['scope'] == 'new_tools' else RUN) / item['path']
        if not p.is_file() or sha(p) != item['sha256']:
            frozen_bad.append(str(p))
    result = {'status': 'PASS' if not (bad or mtimes or historical or frozen_bad) else 'FAIL',
              'checked_protected_records': len(records), 'original_inventory_files': len(original),
              'frozen_artifacts_checked': len(manifest['artifacts']), 'hash_mismatches': bad,
              'mtime_changes': mtimes, 'historical_mtime_changes': historical, 'frozen_artifact_mismatches': frozen_bad,
              'original_manifest_sha256': sha(RUN / 'artifact_manifest.final.json'),
              'new_deliverables_location': str(HERE),
              'test_payload_reopened_for_evaluation': False, 'training_started': False,
              'production_android_changed': False}
    write(out / 'preservation.json', result)
    print(json.dumps(result), flush=True)
    if result['status'] != 'PASS':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
