"""Read-only verification of the additive catalog and frozen comparison. Never fit."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from core import RUN, OLD, SECOND, SCHEMA, read, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preservation():
    baseline = read(RUN / 'preservation_baseline.json')
    failures = []
    for row in baseline['guards']:
        p = Path(row['path'])
        if not p.is_file() or p.stat().st_mtime_ns != row['mtime_ns'] or sha(p) != row['sha256']:
            failures.append({'path': str(p), 'check': 'GUARD_HASH_AND_MTIME'})
    for row in baseline['metadata_snapshot']:
        p = Path(row['path'])
        if not p.is_file() or p.stat().st_size != row['size'] or p.stat().st_mtime_ns != row['mtime_ns']:
            failures.append({'path': str(p), 'check': 'SIZE_AND_MTIME'})
    return {'status': 'PASS' if not failures else 'FAIL', 'failures': failures,
            'fresh_guard_hashes_and_mtimes': len(baseline['guards']),
            'fresh_protected_metadata_checks': len(baseline['metadata_snapshot']),
            'scope': 'Fresh guard SHA256 and protected size/mtime checks; original full content-hash evidence reused, not rerun.',
            'full_hash_evidence_reused': baseline['full_hash_evidence_reused']}


def verify(artifacts=False):
    freeze = read(RUN / 'model_selection_freeze.json')
    expected = {'dataset_plan.json': 'plan_sha256', 'data/processed/train_dev/manifest.json': 'manifest_sha256',
                'training_plan.v2.json': 'training_plan_sha256', 'tools/model_run.py': 'script_sha256',
                'tools/core.py': 'core_script_sha256'}
    for relative, key in expected.items():
        require(sha(RUN / relative) == freeze[key], 'FROZEN_HASH:' + relative)
    for name, frozen in freeze['models'].items():
        base = RUN / 'models' / name
        require(sha(base / 'config.json') == frozen['config_sha256'], 'CONFIG:' + name)
        require(sha(base / 'training_status.json') == frozen['status_sha256'], 'STATUS:' + name)
        require(read(base / 'checkpoints/best.json') == frozen['pointer'], 'BEST_POINTER:' + name)
        require(sha(base / frozen['pointer']['path']) == frozen['pointer']['sha256'], 'CHECKPOINT:' + name)
        require(read(base / 'config.json')['effective_receptive_field_observations'] == 61, 'RF61')
    require(sha(OLD / freeze['unchanged_legacy_baseline']['path']) == freeze['unchanged_legacy_baseline']['sha256'], 'LEGACY_CHECKPOINT')
    require(sha(OLD / 'vocabulary.json') == freeze['legacy_vocabulary_sha256'], 'LEGACY_VOCABULARY')
    result = read(RUN / 'final_results.json')
    require(sha(RUN / 'model_selection_freeze.json') == result['selection_freeze_sha256'], 'SELECTION_FREEZE')
    require(sha(RUN / 'data/processed/test/manifest.json') == result['test_manifest_sha256'], 'TEST_MANIFEST')
    plan = read(RUN / 'expanded_catalog_v2/plan.json')
    rows = read(RUN / 'expanded_catalog_v2/manifest.json')
    quality = read(RUN / 'expanded_catalog_v2/quality.json')
    require(quality['status'] == 'PASS' and not quality['failures'], 'CATALOG_PREPARATION')
    require(len(rows) == plan['eligible_candidates'] == 223, 'CANDIDATE_COVERAGE')
    index_path = Path(read(RUN / 'dataset_plan.json')['index_path'])
    require(sha(index_path) == plan['index_sha256'], 'SOURCE_INDEX_HASH')
    index = list(csv.DictReader(index_path.open(encoding='utf-8-sig', newline='')))
    require(len(index) == plan['index_rows'], 'INDEX_COUNT')
    fields = ('sample_id', 'source_url', 'raw_sha256', 'decoded_pixel_sha256')
    for key in fields:
        require(len({r[key] for r in rows}) == len(rows), 'DUPLICATE:' + key)
    roles = Counter(r['split'] for r in rows)
    require(dict(roles) == plan['role_counts'], 'ROLE_COUNTS')
    groups = {role: {r['collection'] for r in rows if r['split'] == role} for role in roles}
    for a in groups:
        for b in groups:
            if a != b:
                require(not groups[a] & groups[b], 'COLLECTION_OVERLAP:' + a + ':' + b)
    for row in rows:
        origin = index[row['annotation_provenance']['source_row'] - 1]
        require(origin['XML file'] == row['collection'] and origin['Occurs in utterance id'] == row['utterance_id'], 'ANNOTATION_IDENTITY')
        require(origin['Perspective/Camera id'] == '0' and origin['Video sequence id'] == row['source_group'], 'ANNOTATION_VIEW_GROUP')
        require(origin['main gloss'] == row['gloss_text'] and origin['main gloss'].split() == row['glosses'], 'AUTHENTIC_GLOSS_JOIN')
        require(origin['Compressed MOV file'] == row['source_url_original'], 'SOURCE_URL_JOIN')
        require(row['unique_valid'] and row['signer_id'] is None and row['sign_times'] is None, 'NO_INVENTED_SIGNER_TIMING')
        video = Path(row['video_path'])
        receipt = read(str(video) + '.receipt.json')
        require(video.stat().st_size == row['bytes'] == receipt['bytes'], 'VIDEO_SIZE')
        require(receipt['source_url'] == row['source_url'] and receipt['sha256'] == row['raw_sha256'], 'RECEIPT_JOIN')
        require(sha(row['feature_path']) == row['feature_sha256'], 'FEATURE_HASH')
        with np.load(row['feature_path'], allow_pickle=False) as f:
            x, ts = f['X'], f['timestamp_ms']
            require(str(f['schema_version'].item()) == SCHEMA, 'FEATURE_SCHEMA')
            require(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 378 and np.isfinite(x).all(), 'FEATURE_VALUES')
            require(len(ts) == len(x) == row['observations'] and np.isfinite(ts).all() and np.all(np.diff(ts) > 0), 'TIMESTAMPS')
            repeated = sum(a == b for a, b in zip(row['glosses'], row['glosses'][1:]))
            require(len(x) >= len(row['glosses']) + repeated, 'CTC_FEASIBILITY')
    by_id = {r['sample_id']: r for r in rows}
    fitted = read(RUN / 'data/processed/train_dev/manifest.json')
    require(dict(Counter(r['split'] for r in fitted)) == {'train': 100, 'dev': 9}, 'FITTED_COUNTS')
    for row in fitted:
        catalog = by_id[row['sample_id']]
        require(catalog['split'] == {'train': 'TRAIN_POOL_NOT_FIT', 'dev': 'FROZEN_DEV_REFERENCE'}[row['split']], 'FITTED_CATALOG_ROLE')
        require(row['feature_sha256'] == catalog['feature_sha256'] and row['glosses'] == catalog['glosses'], 'FITTED_CATALOG_JOIN')
    test = read(RUN / 'data/processed/test/manifest.json')
    require(len(test) == 14 and not {r['sample_id'] for r in test} & {r['sample_id'] for r in fitted}, 'TEST_ISOLATION')
    from datetime import datetime
    frozen_at = datetime.fromisoformat(freeze['created_utc'])
    for row in test:
        require(by_id[row['sample_id']]['split'] == 'FROZEN_TEST_EXPOSED_LOCKED', 'TEST_ROLE')
        receipt = read(str(row['video_path']) + '.receipt.json')
        require(datetime.fromisoformat(receipt['completed_utc'].replace('Z', '+00:00')) > frozen_at, 'TEST_DOWNLOADED_BEFORE_FREEZE')
    raw = list((RUN / 'data/raw/ncslgr').rglob('*.mov'))
    total = plan['how2sign_bytes_reused_stat_only'] + plan['prior_ncslgr_raw_bytes'] + sum(p.stat().st_size for p in raw)
    require(total <= plan['conservative_global_raw_cap_with_How2Sign'], 'GLOBAL_RAW_CAP')
    # Reuse completed archive verification; check its stat and preservation guard evidence only.
    require((SECOND / 'data/raw/how2sign/val_rgb_front_clips.zip').stat().st_size == plan['how2sign_bytes_reused_stat_only'], 'HOW2SIGN_SIZE')
    protected = preservation()
    require(protected['status'] == 'PASS', 'PROTECTED_FILES_CHANGED:' + json.dumps(protected['failures']))
    count = 0
    if artifacts:
        manifest = read(RUN / 'artifact_manifest.json')
        for entry in manifest['files']:
            path = (RUN / entry['path']).resolve()
            require(RUN in path.parents, 'ARTIFACT_PATH_ESCAPE')
            require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'ARTIFACT_HASH:' + entry['path'])
            count += 1
    return {'status': 'PASS', 'catalog_sequences': len(rows), 'unique_valid': len(rows), 'role_counts': dict(roles),
            'frozen_training_counts': {'train': 100, 'dev': 9}, 'test': 14, 'receptive_field': 61,
            'signer_independent': False, 'conservative_raw_bytes_including_How2Sign': total,
            'raw_cap': plan['conservative_global_raw_cap_with_How2Sign'], 'artifacts_hashed': count,
            'preservation': protected, 'training_performed_by_verifier': False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.artifacts), indent=2))
