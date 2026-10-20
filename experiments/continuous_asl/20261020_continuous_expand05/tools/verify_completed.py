"""Read-only verification of new artifacts, data contracts, freezes and preserved sources."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from core import RUN, SOURCE, ACQUIRED, SCHEMA, read, sha
from finish_mission import preservation


def main():
    checked = {}
    for root in (ACQUIRED, RUN):
        manifest = read(root / 'artifact_manifest.json')
        for entry in manifest['files']:
            path = (root / entry['path']).resolve()
            if root not in path.parents or not path.is_file() or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
                raise ValueError('ARTIFACT_HASH:' + str(path))
        checked[root.name] = len(manifest['files'])
    protected = preservation()
    plan = read(RUN / 'dataset_plan.json')
    rows = read(RUN / 'data/processed/train_dev/manifest.json')
    reserved = read(RUN / 'data/reserved_test/manifest.json')
    if sha(RUN / 'data/processed/train_dev/manifest.json') != plan['manifest_sha256'] or sha(RUN / 'data/reserved_test/manifest.json') != plan['reserved_test_manifest_sha256']:
        raise ValueError('STAGED_MANIFEST_HASH')
    if dict(Counter(r['split'] for r in rows)) != plan['fit_counts']:
        raise ValueError('FIT_COUNTS')
    for key in ('sample_id', 'source_url', 'raw_sha256', 'decoded_pixel_sha256', 'source_group'):
        if len({r[key] for r in rows + reserved}) != len(rows + reserved):
            raise ValueError('FIT_DEV_TEST_LEAKAGE:' + key)
    for row in rows + reserved:
        if sha(row['feature_path']) != row['feature_sha256'] or ' '.join(row['glosses']) != row['gloss_text']:
            raise ValueError('FEATURE_OR_TARGET_JOIN')
        with np.load(row['feature_path'], allow_pickle=False) as f:
            x, ts = f['X'], f['timestamp_ms']
            if str(f['schema_version'].item()) != SCHEMA or x.dtype != np.float32 or x.shape != (len(ts), 378) or not np.isfinite(x).all():
                raise ValueError('FEATURE_CONTRACT')
            if not np.isfinite(ts).all() or not np.all(np.diff(ts) > 0):
                raise ValueError('TIMESTAMPS')
            if len(x) < len(row['glosses']) + sum(a == b for a, b in zip(row['glosses'], row['glosses'][1:])):
                raise ValueError('CTC_FEASIBILITY')
    old_dev = [r for r in read(SOURCE / 'data/processed/train_dev/manifest.json') if r['split'] == 'dev']
    dev = [r for r in rows if r['split'] == 'dev']
    if [(r['sample_id'], r['glosses'], r['feature_sha256']) for r in dev] != [(r['sample_id'], r['glosses'], r['feature_sha256']) for r in old_dev]:
        raise ValueError('FROZEN9DEV_CHANGED')
    final = read(RUN / 'final_reserved_test.json')
    if final['selection_freeze_sha256'] != sha(RUN / 'model_selection_freeze.json') or final['old_exposed14_evaluated'] or final['test_used_for_selection']:
        raise ValueError('TEST_PROTOCOL')
    known_train = {r['signer_id'] for r in rows if r['split'] == 'train' and r.get('signer_id')}
    known_test = {r['signer_id'] for r in reserved if r.get('signer_id')}
    print(json.dumps({'status': 'PASS', 'artifact_hash_counts': checked, 'preservation': protected,
          'fit_counts': plan['fit_counts'], 'new_test': len(reserved), 'signer_independent': False,
          'known_signer_overlap_ids': sorted(known_train & known_test),
          'test_now_exposed': True, 'training_performed_by_verifier': False, 'production_promoted': False}, indent=2))


if __name__ == '__main__':
    main()
