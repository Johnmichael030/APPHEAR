"""Synthetic-only fixture builder; never reads cameras, participants or real media."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import validate_pilot as v
from pipeline.perception_vnext.recognition_schema import save_raw_sequence

def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')

def build(root: Path, stage: str = 'second') -> dict:
    root = root.resolve()
    v.require(v.PACKAGE in root.parents and not root.exists(), 'NEW_PACKAGE_SYNTHETIC_DIRECTORY_REQUIRED')
    root.mkdir(parents=True)
    fixture_path = v.ROOT / 'tests/test_perception_vnext_recognition_schema_v2.py'
    spec = importlib.util.spec_from_file_location('existing_authorized_synthetic_fixture', fixture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    status = 'SYNTHETIC_NOT_CONSENT'
    manifest = {
        'schema': v.MANIFEST_SCHEMA, 'mode': 'SYNTHETIC', 'purpose': 'EVALUATION_ONLY', 'training_allowed': False,
        'fixture_origin': {'path': str(fixture_path.relative_to(v.ROOT)), 'sha256': v.sha(fixture_path),
                           'scope': 'Existing synthetic unit-test geometry, not human landmarks'},
        'contract_hashes': {name: v.sha(v.ROOT / name) for name in v.CONTRACT_FILES},
        'permission_review': {'status': status, 'reviewer_id': 'SYNTHETIC_OPERATOR', 'reference': 'NO_REAL_PERMISSION_ASSERTED'},
        'signer_history': {'status': status, 'reference': 'SYNTHETIC_REGISTRY', 'reviewer_id': 'SYNTHETIC_OPERATOR',
                           'covered_sources': sorted(v.COVERAGE),
                           'registry': [{'signer_id': 'SYN_HISTORICAL', 'identity_key': 'SYN_OLD_KEY', 'role': 'TRAIN'}]},
        'participants': [], 'recordings': []}
    for index in range(1, 2 if stage == 'first' else 3):
        pid, rid, sid = f'SYN_S{index:02}', f'SYN_R{index:02}', f'SYN_SEQ{index:02}'
        manifest['participants'].append({
            'signer_id': pid, 'identity_key': f'SYN_KEY{index:02}', 'role': f'PILOT_{index}_EVAL',
            'qualification': {'status': status, 'reference': 'NO_REAL_SIGNER', 'reviewer_id': 'SYNTHETIC_OPERATOR'},
            'consent': {'status': status, 'withdrawn': False, 'reference': 'NOT_CONSENT',
                        'reviewer_id': 'SYNTHETIC_OPERATOR', 'allowed_actions': sorted(v.ACTIONS)}})
        def item(name: str) -> dict:
            return {'path': name, 'sha256': v.sha(root / name)}
        media_name = f'{rid}.synthetic.bin'
        (root / media_name).write_bytes(b'APPHEAR_SYNTHETIC_NOT_VIDEO\n' + rid.encode('ascii'))
        raw = module._raw()
        raw = replace(raw, metadata={**raw.metadata, 'recording_id': rid, 'signer_id': pid, 'synthetic': True})
        raw_name, seg_name = f'{rid}.raw.npz', f'{rid}.seg378.npz'
        save_raw_sequence(root / raw_name, raw)
        np.savez_compressed(root / seg_name, X=v.derive_causal_segmentation_features(raw),
                            timestamp_ms=raw.timestamp_ms, frame_index=raw.frame_index,
                            schema_version=np.asarray(v.SEGMENTATION_FEATURE_SCHEMA))
        frame_name, ann_name = f'{rid}.capture.json', f'{rid}.annotations.json'
        write_json(root / frame_name, {'schema': 'apphear.streaming_pilot.capture_index.v1', 'recording_id': rid,
                                     'frames': [{'frame_index': int(i), 'source_timestamp_ms': float(t)}
                                                for i, t in zip(raw.frame_index, raw.timestamp_ms)]})
        ids = [f'{rid}_A', f'{rid}_B']
        windows = []
        for interval_id, start, end in ((ids[0], 0.0, 80.0), (ids[1], 80.0, 200.0)):
            chosen = np.flatnonzero((raw.timestamp_ms >= start) & (raw.timestamp_ms < end))
            subset = replace(raw, **{name: getattr(raw, name)[chosen] for name in raw.__dataclass_fields__ if name != 'metadata'})
            view = v.temporal_view_v2(subset, condition='dense', phone_completed_fps=15.0)
            rec_name = f'{interval_id}.350.npz'
            np.savez_compressed(root / rec_name, X=view['features'], frame_mask=view['mask'],
                                timestamp_ms=view['source_timestamps_ms'],
                                frame_index=subset.frame_index[view['source_indices']], schema_version=np.asarray(v.FEATURE_SCHEMA_V2))
            windows.append({'interval_id': interval_id, 'basis': 'REFERENCE_DIAGNOSTIC_ONLY',
                            'start_ms': start, 'end_ms': end, 'features': item(rec_name)})
        write_json(root / ann_name, {
            'schema': v.ANNOTATION_SCHEMA, 'recording_id': rid, 'signer_id': pid,
            'provenance': {'kind': 'SYNTHETIC', 'reviewer_ids': ['SYN_REVIEW_1', 'SYN_REVIEW_2'],
                           'prediction_blinded': True, 'adjudication': 'COMPLETE', 'reference': 'SYNTHETIC_NOT_GROUND_TRUTH'},
            'intervals': [
                {'interval_id': ids[0], 'type': 'SIGN', 'sequence_id': sid, 'start_ms': 0.0, 'end_ms': 80.0, 'gloss': 'SYN_TOKEN_A'},
                {'interval_id': ids[1], 'type': 'SIGN', 'sequence_id': sid, 'start_ms': 80.0, 'end_ms': 200.0, 'gloss': 'SYN_TOKEN_B'},
                {'interval_id': f'{rid}_N', 'type': 'NO_SIGN', 'sequence_id': None, 'start_ms': 200.0,
                 'end_ms': 400.0, 'gloss': None, 'negative_subtype': 'VISIBLE_IDLE'}]})
        manifest['recordings'].append({
            'recording_id': rid, 'session_id': f'SYN_SESSION{index:02}', 'signer_id': pid, 'prompt_id': 'SYNTHETIC_ONLY',
            'start_ms': 0.0, 'end_ms': 400.0, 'source_clock': 'SOURCE_MONOTONIC_MS',
            'image_semantics': 'UPRIGHT_UNMIRRORED_SOURCE_RGB', 'device_id': 'NO_DEVICE_SYNTHETIC', 'nominal_fps': 30.0,
            'producer_reference': 'SYNTHETIC_UNIT_FIXTURE_NOT_A_CAPTURE',
            'media': item(media_name), 'capture_index': item(frame_name), 'raw_observations': item(raw_name),
            'segmentation_features': item(seg_name), 'annotations': item(ann_name),
            'recognition_windows': windows,
            'sequences': [{'sequence_id': sid, 'start_ms': 0.0, 'end_ms': 200.0, 'label_type': 'SYNTHETIC_TOKENS',
                           'glosses': ['SYN_TOKEN_A', 'SYN_TOKEN_B'], 'interval_ids': ids}]})
    write_json(root / 'manifest.json', manifest)
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('first', 'second'), default='second')
    args = parser.parse_args()
    build(args.out, args.stage)
    print(json.dumps({'status': 'SYNTHETIC_FIXTURE_ONLY', 'output': str(args.out), 'real_consent_or_labels': False}))
