"""Targeted permission/compatibility rejection checks; no real recordings."""
import copy
import json
from pathlib import Path
import uuid
import numpy as np
import pytest
import synthetic_fixture as fixture
import validate_pilot as v

@pytest.fixture
def tmp_path():
    # Short names avoid Windows MAX_PATH effects masking validator behavior.
    path = v.PACKAGE / 't' / uuid.uuid4().hex[:8]
    path.mkdir(parents=True)
    return path

@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'bundle'
    return root, fixture.build(root)

def validate(root, manifest):
    return v.validate(manifest, root, synthetic=True, stage='second')

def rewrite(root, item, edit, npz=False):
    path = root / item['path']
    if npz:
        with np.load(path, allow_pickle=False) as data:
            obj = {name: np.asarray(data[name]) for name in data.files}
        edit(obj)
        np.savez_compressed(path, **obj)
    else:
        obj = json.loads(path.read_text())
        edit(obj)
        path.write_text(json.dumps(obj), encoding='utf-8')
    item['sha256'] = v.sha(path)

def test_valid_second_synthetic_is_never_model_evidence(bundle):
    result = validate(*bundle)
    assert result['status'] == 'PASS_SYNTHETIC' and not result['training_allowed']
    assert not result['model_evaluation_evidence'] and not result['real_first_signer_validated']

def test_valid_first_synthetic(tmp_path):
    root = tmp_path / 'first'
    result = v.validate(fixture.build(root, 'first'), root, synthetic=True, stage='first')
    assert result['status'] == 'PASS_SYNTHETIC' and result['signers'] == 1

@pytest.mark.parametrize('mutation,code', [
    ('schema', 'MANIFEST_SCHEMA'), ('mode', 'MODE_MISMATCH'), ('training', 'EVALUATION_ONLY'),
    ('permission', 'PERMISSION'), ('withdrawn', 'CONSENT'), ('scope', 'CONSENT'),
    ('history', 'IDENTITY_HISTORY'), ('identity_alias', 'SIGNER_OVERLAP'),
    ('historical_alias', 'SIGNER_OVERLAP'), ('contract', 'PRODUCER_CONTRACT'),
    ('recording_join', 'RECORDING_ID'), ('media_duplicate', 'DUPLICATE_RAW'),
    ('path_escape', 'ARTIFACT_PATH_ESCAPE'), ('hash', 'ARTIFACT_HASH'),
    ('english_labels', 'ENGLISH_OR_UNVERIFIED'), ('interval_join', 'ANNOTATION_SEQUENCE_JOIN'),
    ('sequence_bounds', 'SEQUENCE_BOUNDARIES'), ('gloss_order', 'GLOSS_ORDER'),
])
def test_manifest_rejections(bundle, mutation, code):
    root, m = bundle
    rec = m['recordings'][0]; seq = rec['sequences'][0]
    if mutation == 'schema': m['schema'] = 'wrong'
    elif mutation == 'mode': m['mode'] = 'REAL'
    elif mutation == 'training': m['training_allowed'] = True
    elif mutation == 'permission': m['permission_review']['status'] = 'PENDING'
    elif mutation == 'withdrawn': m['participants'][0]['consent']['withdrawn'] = True
    elif mutation == 'scope': m['participants'][0]['consent']['allowed_actions'] = ['RECORD']
    elif mutation == 'history': m['signer_history']['covered_sources'] = []
    elif mutation == 'identity_alias': m['participants'][1]['identity_key'] = m['participants'][0]['identity_key']
    elif mutation == 'historical_alias': m['participants'][0]['identity_key'] = 'SYN_OLD_KEY'
    elif mutation == 'contract': m['contract_hashes'] = {}
    elif mutation == 'recording_join': rec['signer_id'] = 'UNKNOWN'
    elif mutation == 'media_duplicate': m['recordings'][1]['media'] = copy.deepcopy(rec['media'])
    elif mutation == 'path_escape': rec['media']['path'] = '../outside.bin'
    elif mutation == 'hash': rec['media']['sha256'] = '0' * 64
    elif mutation == 'english_labels': seq['label_type'] = 'ENGLISH_TRANSLATION'
    elif mutation == 'interval_join': seq['interval_ids'] = ['ABSENT']
    elif mutation == 'sequence_bounds': seq['end_ms'] = 9000.0
    elif mutation == 'gloss_order': seq['glosses'].reverse()
    with pytest.raises(v.Rejected, match=code): validate(root, m)

@pytest.mark.parametrize('mutation,code', [
    ('missing_timestamp', 'CAPTURE_TIMESTAMP_MISSING'), ('duplicate_timestamp', 'CAPTURE_TIMESTAMPS'),
    ('decreasing_timestamp', 'CAPTURE_TIMESTAMPS'), ('duplicate_frame', 'CAPTURE_FRAME_INDEX'),
    ('annotation_join', 'ANNOTATION_RECORDING_JOIN'), ('annotation_overlap', 'INTERVAL_BOUNDARY'),
    ('annotation_duplicate', 'DUPLICATE_INTERVAL_ID'), ('reviewers', 'ANNOTATION_PROVENANCE'),
])
def test_json_artifact_rejections(bundle, mutation, code):
    root, m = bundle; rec = m['recordings'][0]
    item = rec['capture_index'] if 'timestamp' in mutation or mutation == 'duplicate_frame' else rec['annotations']
    def edit(obj):
        if mutation == 'missing_timestamp': del obj['frames'][1]['source_timestamp_ms']
        elif mutation == 'duplicate_timestamp': obj['frames'][1]['source_timestamp_ms'] = 0.0
        elif mutation == 'decreasing_timestamp': obj['frames'][1]['source_timestamp_ms'] = -1.0
        elif mutation == 'duplicate_frame': obj['frames'][1]['frame_index'] = 0
        elif mutation == 'annotation_join': obj['signer_id'] = 'OTHER'
        elif mutation == 'annotation_overlap': obj['intervals'][1]['start_ms'] = 70.0
        elif mutation == 'annotation_duplicate': obj['intervals'][1]['interval_id'] = obj['intervals'][0]['interval_id']
        elif mutation == 'reviewers': obj['provenance']['reviewer_ids'] = ['SAME', 'SAME']
    rewrite(root, item, edit)
    with pytest.raises(v.Rejected, match=code): validate(root, m)

@pytest.mark.parametrize('mutation,code', [
    ('wrong_schema', 'FEATURE_SCHEMA'), ('wrong_width', 'FEATURE_SHAPE'), ('nan', 'FEATURE_SHAPE'),
    ('duplicate_feature_time', 'SEGMENTATION_TIMESTAMPS'), ('source_join', 'SEGMENTATION_SOURCE_JOIN'),
    ('numerical_skew', 'SEGMENTATION_RECIPE'), ('padding_mask', 'RECOGNITION_MASK'),
    ('recognition_join', 'RECOGNITION_SOURCE_JOIN'), ('raw_schema', 'RAW_SCHEMA'),
])
def test_npz_rejections(bundle, mutation, code):
    root, m = bundle; rec = m['recordings'][0]
    item = rec['raw_observations'] if mutation == 'raw_schema' else (
        rec['recognition_windows'][0]['features'] if mutation in ('padding_mask', 'recognition_join') else rec['segmentation_features'])
    def edit(obj):
        if mutation == 'wrong_schema': obj['schema_version'] = np.asarray('3.0')
        elif mutation == 'wrong_width': obj['X'] = obj['X'][:, :258]
        elif mutation == 'nan': obj['X'][0, 0] = np.nan
        elif mutation == 'duplicate_feature_time': obj['timestamp_ms'][1] = obj['timestamp_ms'][0]
        elif mutation == 'source_join': obj['frame_index'][0] = 99
        elif mutation == 'numerical_skew': obj['X'][0, 0] += 1.0
        elif mutation == 'padding_mask': obj['frame_mask'][8] = True
        elif mutation == 'recognition_join': obj['frame_index'][0] = 99
        elif mutation == 'raw_schema':
            metadata = json.loads(str(obj['metadata_json'].item())); metadata['schema'] = 'wrong'
            obj['metadata_json'] = np.asarray(json.dumps(metadata))
    rewrite(root, item, edit, npz=True)
    with pytest.raises(v.Rejected, match=code): validate(root, m)

def test_permission_failure_precedes_any_participant_artifact_read(bundle, monkeypatch):
    root, m = bundle
    m['mode'] = 'REAL'
    m['permission_review']['status'] = 'PENDING'
    touched = []
    def forbidden(*args, **kwargs):
        touched.append(True)
        raise AssertionError('Participant artifact accessed without permission')
    monkeypatch.setattr(v, 'artifact', forbidden)
    with pytest.raises(v.Rejected, match='PERMISSION_NOT_ESTABLISHED'):
        v.validate(m, root, synthetic=False, stage='second')
    assert not touched

def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(v.Rejected, match='DUPLICATE_JSON_KEY'): v.read_json(path)

def test_nonfinite_json_rejected(tmp_path):
    path = tmp_path / 'nan.json'; path.write_text('{"time":NaN}')
    with pytest.raises(v.Rejected, match='NONFINITE_JSON_NUMBER'): v.read_json(path)

def test_phrase_is_not_one_recognition_window(bundle):
    root, m = bundle
    m['recordings'][0]['recognition_windows'][0]['end_ms'] = 200.0
    with pytest.raises(v.Rejected, match='REFERENCE_WINDOW_BOUNDARIES'): validate(root, m)

def test_missing_window_rejected(bundle):
    root, m = bundle
    m['recordings'][0]['recognition_windows'].pop()
    with pytest.raises(v.Rejected, match='MISSING_REFERENCE_WINDOW'): validate(root, m)

def test_missing_feature_timestamp_array_rejected(bundle):
    root, m = bundle
    rewrite(root, m['recordings'][0]['segmentation_features'], lambda obj: obj.pop('timestamp_ms'), npz=True)
    with pytest.raises(v.Rejected, match='FEATURE_KEYS'): validate(root, m)

def test_raw_nonfinite_timestamp_rejected(bundle):
    root, m = bundle
    rewrite(root, m['recordings'][0]['raw_observations'], lambda obj: obj['timestamp_ms'].__setitem__(1, np.nan), npz=True)
    with pytest.raises(ValueError): validate(root, m)

def test_expired_real_consent_fails_preflight_without_artifacts(bundle):
    root, m = bundle
    m['mode'] = 'REAL'
    m['permission_review']['status'] = m['signer_history']['status'] = 'HUMAN_VERIFIED'
    for person in m['participants']:
        person['qualification']['status'] = person['consent']['status'] = 'HUMAN_VERIFIED'
        person['consent'].update(established_utc='2000-01-01T00:00:00Z', retention_until_utc='2001-01-01T00:00:00Z')
    with pytest.raises(v.Rejected, match='CONSENT_DATE_OR_RETENTION_INVALID'):
        v.permissions(m, synthetic=False, stage='second')

def test_boolean_window_timestamp_rejected(bundle):
    root, m = bundle
    m['recordings'][0]['recognition_windows'][0]['start_ms'] = False
    with pytest.raises(v.Rejected, match='REFERENCE_WINDOW_BOUNDARIES'): validate(root, m)

def test_nonboolean_visibility_flags_rejected(bundle):
    root, m = bundle
    rewrite(root, m['recordings'][0]['raw_observations'], lambda obj: obj.update(hand_observed=obj['hand_observed'].astype(np.int32)), npz=True)
    with pytest.raises(v.Rejected, match='RAW_VALIDITY_FLAGS_MUST_BE_BOOLEAN'): validate(root, m)
