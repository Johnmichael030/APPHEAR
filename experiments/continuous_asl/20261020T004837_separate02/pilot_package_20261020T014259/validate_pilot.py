"""Strict, read-only streaming-pilot admission; no capture, inference or training."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any
import zipfile

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from pipeline.perception_vnext.recognition_schema import RAW_SCHEMA, load_raw_sequence
from pipeline.perception_vnext.recognition_schema_v2 import FEATURE_SCHEMA_V2, temporal_view_v2
from pipeline.perception_vnext.segmentation.learned_features import (
    SEGMENTATION_FEATURE_SCHEMA, derive_causal_segmentation_features,
)

MANIFEST_SCHEMA = 'apphear.streaming_pilot.manifest.v1'
ANNOTATION_SCHEMA = 'apphear.streaming_pilot.annotations.v1'
COVERAGE = {'current_word_training', 'current_segmentation_training',
            'legacy_isolated_frozen', 'prior_phone_design'}
ACTIONS = {'RECORD', 'LOCAL_PROCESS', 'ANNOTATE', 'EVALUATE'}
CONTRACT_FILES = ('pipeline/perception_vnext/recognition_schema.py',
                  'pipeline/perception_vnext/recognition_schema_v2.py',
                  'pipeline/perception_vnext/segmentation/learned_features.py')

class Rejected(ValueError):
    """Machine-readable rejection; do not print participant paths or labels."""

def require(condition: bool, code: str) -> None:
    if not condition:
        raise Rejected(code)

def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        require(key not in result, 'DUPLICATE_JSON_KEY')
        result[key] = value
    return result

def read_json(path: Path) -> dict:
    require(path.stat().st_size <= 8 * 1024 * 1024, 'JSON_SIZE_LIMIT')
    def reject_constant(value: str) -> None:
        raise Rejected('NONFINITE_JSON_NUMBER')
    with path.open(encoding='utf-8-sig') as stream:
        result = json.load(stream, object_pairs_hook=_pairs, parse_constant=reject_constant)
    require(isinstance(result, dict), 'JSON_OBJECT_REQUIRED')
    return result

def text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())

def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)

def identity(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value) is not None

def date(value: Any) -> datetime:
    require(text(value), 'DATE_REQUIRED')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'DATE_TIMEZONE_REQUIRED')
    return result

def permissions(manifest: dict, synthetic: bool, stage: str) -> dict:
    """Complete deidentified permission/identity gate BEFORE artifact resolution/read."""
    require(manifest.get('schema') == MANIFEST_SCHEMA, 'MANIFEST_SCHEMA')
    require(manifest.get('mode') == ('SYNTHETIC' if synthetic else 'REAL'), 'MODE_MISMATCH')
    require(manifest.get('purpose') == 'EVALUATION_ONLY' and manifest.get('training_allowed') is False,
            'EVALUATION_ONLY_REQUIRED')
    people = manifest.get('participants')
    require(isinstance(people, list) and len(people) == (1 if stage == 'first' else 2), 'STAGE_PARTICIPANT_COUNT')
    status = 'SYNTHETIC_NOT_CONSENT' if synthetic else 'HUMAN_VERIFIED'
    approval = manifest.get('permission_review', {})
    require(approval.get('status') == status and text(approval.get('reviewer_id'))
            and text(approval.get('reference')), 'PERMISSION_NOT_ESTABLISHED')
    history = manifest.get('signer_history', {})
    require(history.get('status') == status and text(history.get('reference'))
            and text(history.get('reviewer_id'))
            and set(history.get('covered_sources', [])) >= COVERAGE, 'IDENTITY_HISTORY_NOT_VERIFIED')
    registry = history.get('registry')
    require(isinstance(registry, list) and bool(registry), 'HISTORICAL_REGISTRY_REQUIRED')
    old_keys, old_ids = set(), set()
    for row in registry:
        require(identity(row.get('identity_key')) and identity(row.get('signer_id'))
                and text(row.get('role')), 'REGISTRY_ROW_INVALID')
        old_keys.add(row['identity_key']); old_ids.add(row['signer_id'])
    people_by_id, keys = {}, set()
    for index, person in enumerate(people, 1):
        pid, key = person.get('signer_id'), person.get('identity_key')
        require(identity(pid) and identity(key), 'PSEUDONYMOUS_ID_REQUIRED')
        require(pid not in people_by_id and key not in keys and pid not in old_ids and key not in old_keys,
                'SIGNER_OVERLAP')
        require(person.get('role') == f'PILOT_{index}_EVAL', 'PILOT_ROLE_ORDER')
        qualification = person.get('qualification', {})
        require(qualification.get('status') == status and text(qualification.get('reference'))
                and text(qualification.get('reviewer_id')), 'ASL_QUALIFICATION_NOT_VERIFIED')
        consent = person.get('consent', {})
        require(consent.get('status') == status and consent.get('withdrawn') is False
                and text(consent.get('reference')) and text(consent.get('reviewer_id'))
                and set(consent.get('allowed_actions', [])) >= ACTIONS, 'CONSENT_NOT_ESTABLISHED')
        if not synthetic:
            now = datetime.now(timezone.utc)
            require(date(consent.get('established_utc')) <= now < date(consent.get('retention_until_utc')),
                    'CONSENT_DATE_OR_RETENTION_INVALID')
        keys.add(key); people_by_id[pid] = person
    return people_by_id

def artifact(root: Path, item: dict, *, max_bytes: int = 512 * 1024 * 1024) -> Path:
    name = item.get('path')
    require(text(name) and '\\' not in name and ':' not in name, 'ARTIFACT_RELATIVE_PATH')
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'ARTIFACT_PATH_ESCAPE')
    candidate = root / relative
    require(root in candidate.resolve().parents, 'ARTIFACT_PATH_ESCAPE')
    for parent in (candidate, *candidate.parents):
        require(not (parent.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
                if hasattr(parent.lstat(), 'st_file_attributes') else not parent.is_symlink(),
                'ARTIFACT_LINK_FORBIDDEN')
        if parent == root:
            break
    require(candidate.is_file() and 0 < candidate.stat().st_size <= max_bytes, 'ARTIFACT_MISSING_OR_SIZE')
    require(isinstance(item.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', item['sha256']) is not None,
            'ARTIFACT_HASH_REQUIRED')
    require(sha(candidate) == item['sha256'], 'ARTIFACT_HASH_MISMATCH')
    return candidate

def bounded_npz(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist()
        require(len(rows) <= 32 and len({r.filename for r in rows}) == len(rows)
                and sum(r.file_size for r in rows) <= 128 * 1024 * 1024,
                'NPZ_EXPANSION_OR_DUPLICATE_MEMBER')
        require(all(not r.flag_bits & 1 for r in rows), 'NPZ_ENCRYPTED')

def times(value: np.ndarray, expected: int, code: str) -> None:
    require(value.dtype == np.float64 and value.shape == (expected,)
            and np.all(np.isfinite(value)) and np.all(np.diff(value) > 0), code)

def feature(path: Path, schema: str, width: int) -> dict:
    bounded_npz(path)
    with np.load(path, allow_pickle=False) as data:
        # Check schema BEFORE loading the feature vector.
        require(str(data['schema_version'].item()) == schema, 'FEATURE_SCHEMA_MISMATCH')
        expected = {'schema_version', 'X', 'timestamp_ms', 'frame_index'}
        if width == 350:
            expected.add('frame_mask')
        require(set(data.files) == expected, 'FEATURE_KEYS')
        result = {name: np.asarray(data[name]) for name in expected}
    x = result['X']
    require(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == width
            and len(x) > 0 and np.all(np.isfinite(x)), 'FEATURE_SHAPE_DTYPE_OR_FINITE')
    return result

def validate(manifest: dict, data_root: Path, *, synthetic: bool, stage: str) -> dict:
    people = permissions(manifest, synthetic, stage)
    root = data_root.resolve()
    require(root.is_dir(), 'DATA_ROOT_MISSING')
    if synthetic:
        require(PACKAGE in root.parents, 'SYNTHETIC_ROOT_MUST_BE_IN_PACKAGE')
    else:
        require(root != ROOT and ROOT not in root.parents and root not in ROOT.parents,
                'REAL_DATA_MUST_BE_OUTSIDE_REPOSITORY')
    expected = {name: sha(ROOT / name) for name in CONTRACT_FILES}
    require(manifest.get('contract_hashes') == expected, 'PRODUCER_CONTRACT_HASH_MISMATCH')
    records = manifest.get('recordings')
    require(isinstance(records, list) and bool(records), 'NO_RECORDINGS')
    seen_ids, seen_hashes, sequence_ids, interval_ids = set(), set(), set(), set()
    counts = {pid: {'recordings': 0, 'sequences': 0, 'reviewed_no_sign_ms': 0.0} for pid in people}
    for rec in records:
        rid, pid = rec.get('recording_id'), rec.get('signer_id')
        require(identity(rid) and rid not in seen_ids and pid in people, 'RECORDING_ID_OR_SIGNER_JOIN')
        require(identity(rec.get('session_id')) and text(rec.get('prompt_id')), 'SESSION_AND_PROMPT_REQUIRED')
        seen_ids.add(rid)
        require(rec.get('source_clock') == 'SOURCE_MONOTONIC_MS'
                and rec.get('image_semantics') == 'UPRIGHT_UNMIRRORED_SOURCE_RGB', 'SOURCE_CLOCK_OR_ORIENTATION')
        require(number(rec.get('start_ms')) and number(rec.get('end_ms'))
                and rec['start_ms'] < rec['end_ms'], 'RECORDING_BOUNDARIES')
        require(text(rec.get('device_id')) and number(rec.get('nominal_fps'))
                and rec['nominal_fps'] > 0 and text(rec.get('producer_reference')), 'CAPTURE_METADATA_REQUIRED')
        media = artifact(root, rec['media'])
        require(rec['media']['sha256'] not in seen_hashes, 'DUPLICATE_RAW_RECORDING')
        seen_hashes.add(rec['media']['sha256'])
        if synthetic:
            require(media.read_bytes().startswith(b'APPHEAR_SYNTHETIC_NOT_VIDEO\n'), 'SYNTHETIC_MEDIA_MARKER')
        index = read_json(artifact(root, rec['capture_index'], max_bytes=8 * 1024 * 1024))
        require(index.get('schema') == 'apphear.streaming_pilot.capture_index.v1'
                and index.get('recording_id') == rid, 'CAPTURE_INDEX_JOIN')
        frames = index.get('frames', [])
        require(2 <= len(frames) <= 10000, 'CAPTURE_FRAME_COUNT')
        require(all(number(row.get('source_timestamp_ms')) for row in frames), 'CAPTURE_TIMESTAMP_MISSING')
        require(all(isinstance(row.get('frame_index'), int) and not isinstance(row['frame_index'], bool) for row in frames),
                'CAPTURE_FRAME_INDEX_TYPE')
        frame_times = np.asarray([row['source_timestamp_ms'] for row in frames], dtype=np.float64)
        times(frame_times, len(frames), 'CAPTURE_TIMESTAMPS_MISSING_DUPLICATE_OR_ORDER')
        require([row['frame_index'] for row in frames] == list(range(len(frames))), 'CAPTURE_FRAME_INDEX_MISSING_OR_DUPLICATE')
        require(frame_times[0] >= rec['start_ms'] and frame_times[-1] < rec['end_ms'], 'CAPTURE_OUTSIDE_RECORDING')
        raw_path = artifact(root, rec['raw_observations'])
        bounded_npz(raw_path)
        with np.load(raw_path, allow_pickle=False) as data:
            require(json.loads(str(data['metadata_json'].item())).get('schema') == RAW_SCHEMA, 'RAW_SCHEMA_MISMATCH')
        raw = load_raw_sequence(raw_path)
        require(raw.metadata.get('schema') == RAW_SCHEMA and raw.metadata.get('recording_id') == rid
                and raw.metadata.get('signer_id') == pid
                and raw.metadata.get('synthetic') is synthetic, 'RAW_METADATA_JOIN')
        times(raw.timestamp_ms, len(raw.frame_index), 'RAW_TIMESTAMPS_MISSING_DUPLICATE_OR_ORDER')
        require(all(getattr(raw, name).dtype == np.bool_ for name in
                    ('hand_joint_valid', 'hand_observed', 'hand_proposal_present', 'body_valid', 'body_available')),
                'RAW_VALIDITY_FLAGS_MUST_BE_BOOLEAN')
        require(np.issubdtype(raw.frame_index.dtype, np.integer) and np.all(raw.frame_index >= 0)
                and np.all(raw.frame_index < len(frames)), 'RAW_FRAME_INDEX')
        require(np.array_equal(raw.timestamp_ms, frame_times[raw.frame_index]), 'RAW_CAPTURE_TIMESTAMP_JOIN')
        require(np.all(np.isfinite(raw.source_size)) and np.all(raw.source_size > 0), 'RAW_IMAGE_SIZE')
        seg = feature(artifact(root, rec['segmentation_features']), SEGMENTATION_FEATURE_SCHEMA, 378)
        times(seg['timestamp_ms'], len(raw.timestamp_ms), 'SEGMENTATION_TIMESTAMPS')
        require(np.array_equal(seg['timestamp_ms'], raw.timestamp_ms)
                and np.array_equal(seg['frame_index'], raw.frame_index), 'SEGMENTATION_SOURCE_JOIN')
        require(np.allclose(seg['X'], derive_causal_segmentation_features(raw), atol=1e-6, rtol=0), 'SEGMENTATION_RECIPE_MISMATCH')
        ann = read_json(artifact(root, rec['annotations'], max_bytes=8 * 1024 * 1024))
        require(ann.get('schema') == ANNOTATION_SCHEMA and ann.get('recording_id') == rid
                and ann.get('signer_id') == pid, 'ANNOTATION_RECORDING_JOIN')
        provenance = ann.get('provenance', {})
        reviewers = provenance.get('reviewer_ids', [])
        require(provenance.get('kind') == ('SYNTHETIC' if synthetic else 'QUALIFIED_ASL_REVIEW')
                and len(reviewers) >= 2 and all(text(v) for v in reviewers)
                and len(set(reviewers)) == len(reviewers)
                and provenance.get('prediction_blinded') is True
                and provenance.get('adjudication') == 'COMPLETE' and text(provenance.get('reference')),
                'ANNOTATION_PROVENANCE')
        intervals, previous_end = {}, rec['start_ms']
        for row in ann.get('intervals', []):
            iid = row.get('interval_id')
            require(identity(iid) and iid not in interval_ids, 'DUPLICATE_INTERVAL_ID')
            interval_ids.add(iid)
            require(number(row.get('start_ms')) and number(row.get('end_ms'))
                    and previous_end <= row['start_ms'] < row['end_ms'] <= rec['end_ms'],
                    'INTERVAL_BOUNDARY_OVERLAP_OR_ORDER')
            previous_end = row['end_ms']
            kind = row.get('type')
            require(kind in {'SIGN', 'NO_SIGN', 'OOV_SIGN', 'UNCERTAIN'}, 'INTERVAL_LABEL_TYPE')
            if kind == 'SIGN':
                require(text(row.get('gloss')) and identity(row.get('sequence_id')), 'AUTHENTIC_SIGN_LABEL_REQUIRED')
            elif kind == 'NO_SIGN':
                require(row.get('gloss') is None and row.get('sequence_id') is None
                        and row.get('negative_subtype') in {'NO_VISIBLE_HANDS', 'VISIBLE_IDLE', 'FACE_ADJACENT', 'ACCIDENTAL_MOTION'},
                        'NO_SIGN_LABEL_INVALID')
                counts[pid]['reviewed_no_sign_ms'] += row['end_ms'] - row['start_ms']
            else:
                require(row.get('sequence_id') is None, 'UNSUPPORTED_LABEL_NOT_GLOSS_SEQUENCE')
            intervals[iid] = row
        used, last_sequence_end = set(), rec['start_ms']
        for seq in rec.get('sequences', []):
            sid = seq.get('sequence_id')
            require(identity(sid) and sid not in sequence_ids, 'DUPLICATE_SEQUENCE_ID')
            sequence_ids.add(sid)
            require(number(seq.get('start_ms')) and number(seq.get('end_ms'))
                    and last_sequence_end <= seq['start_ms'] < seq['end_ms'] <= rec['end_ms'], 'SEQUENCE_BOUNDARIES')
            last_sequence_end = seq['end_ms']
            require(seq.get('label_type') == ('SYNTHETIC_TOKENS' if synthetic else 'AUTHENTIC_ASL_GLOSS_SEQUENCE'),
                    'ENGLISH_OR_UNVERIFIED_LABELS_NOT_GLOSSES')
            ids = seq.get('interval_ids', [])
            require(bool(ids) and len(set(ids)) == len(ids) and not (set(ids) & used)
                    and all(i in intervals for i in ids), 'ANNOTATION_SEQUENCE_JOIN')
            rows = [intervals[i] for i in ids]
            require(all(row['type'] == 'SIGN' and row.get('sequence_id') == sid
                        and seq['start_ms'] <= row['start_ms'] < row['end_ms'] <= seq['end_ms'] for row in rows),
                    'SIGN_SEQUENCE_BOUNDARY_JOIN')
            require([row['start_ms'] for row in rows] == sorted(row['start_ms'] for row in rows)
                    and seq.get('glosses') == [row['gloss'] for row in rows], 'GLOSS_ORDER_OR_SEQUENCE_MISMATCH')
            used.update(ids)
            counts[pid]['sequences'] += 1
        require(used == {i for i, row in intervals.items() if row['type'] == 'SIGN'}, 'ORPHAN_SIGN_ANNOTATION')
        # WORD recognizes one isolated event per350 window, never an entire phrase.
        # Reference windows test feature plumbing only and cannot prove event selection.
        windows_used = set()
        for window in rec.get('recognition_windows', []):
            iid = window.get('interval_id')
            require(iid in used and iid not in windows_used, 'WINDOW_ANNOTATION_JOIN')
            label = intervals[iid]
            require(number(window.get('start_ms')) and number(window.get('end_ms'))
                    and window.get('basis') == 'REFERENCE_DIAGNOSTIC_ONLY'
                    and window.get('start_ms') == label['start_ms'] and window.get('end_ms') == label['end_ms'],
                    'REFERENCE_WINDOW_BOUNDARIES')
            windows_used.add(iid)
            chosen = np.flatnonzero((raw.timestamp_ms >= window['start_ms']) & (raw.timestamp_ms < window['end_ms']))
            require(len(chosen) > 0, 'EMPTY_SEQUENCE_OBSERVATIONS')
            subset = replace(raw, **{name: getattr(raw, name)[chosen] for name in raw.__dataclass_fields__ if name != 'metadata'})
            view = temporal_view_v2(subset, condition='dense', phone_completed_fps=15.0)
            rec_features = feature(artifact(root, window['features']), FEATURE_SCHEMA_V2, 350)
            real_length = int(view['real_length'])
            times(rec_features['timestamp_ms'], real_length, 'RECOGNITION_TIMESTAMPS')
            require(rec_features['X'].shape == (96, 350) and rec_features['frame_mask'].dtype == np.bool_
                    and np.array_equal(rec_features['frame_mask'], view['mask']), 'RECOGNITION_MASK_OR_LENGTH')
            require(np.array_equal(rec_features['timestamp_ms'], view['source_timestamps_ms'])
                    and np.array_equal(rec_features['frame_index'], subset.frame_index[view['source_indices']]),
                    'RECOGNITION_SOURCE_JOIN')
            require(np.allclose(rec_features['X'], view['features'], atol=1e-6, rtol=0), 'RECOGNITION_RECIPE_MISMATCH')
        require(windows_used == used, 'MISSING_REFERENCE_WINDOW_FEATURES')
        counts[pid]['recordings'] += 1
    require(all(row['recordings'] > 0 and row['sequences'] > 0 and row['reviewed_no_sign_ms'] > 0 for row in counts.values()),
            'EACH_SIGNER_NEEDS_POSITIVE_AND_NO_SIGN_SMOKE')
    return {'status': 'PASS_SYNTHETIC' if synthetic else 'PARTIAL_REAL_COMPATIBILITY_ONLY',
            'stage': stage, 'synthetic': synthetic, 'signers': len(people), 'recordings': len(records),
            'feature_and_annotation_checks': 'PASS', 'per_signer': counts,
            'collection_volume_targets_met': all(row['sequences'] >= 10 and row['reviewed_no_sign_ms'] >= 900000 for row in counts.values()),
            'permission_evidence': 'SYNTHETIC_NOT_CONSENT' if synthetic else 'HUMAN_ATTESTATIONS_CHECKED_NOT_AUTHENTICATED_BY_SOFTWARE',
            'real_first_signer_validated': not synthetic, 'model_evaluation_evidence': False,
            'android_parity_and_end_to_end_latency': 'NOT_MEASURED',
            'runtime_event_selection': 'NOT_MEASURED_REFERENCE_WINDOWS_ONLY', 'training_allowed': False}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--stage', choices=('first', 'second'), required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        result = validate(read_json(args.manifest), args.data_root, synthetic=args.synthetic, stage=args.stage)
        code = 0
    except (Rejected, ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile, IndexError, OverflowError) as error:
        result = {'status': 'BLOCKED_OR_REJECTED', 'reason': str(error) if isinstance(error, Rejected) else type(error).__name__,
                  'synthetic': args.synthetic, 'training_allowed': False, 'model_evaluation_evidence': False,
                  'real_first_signer_validated': False}
        code = 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
