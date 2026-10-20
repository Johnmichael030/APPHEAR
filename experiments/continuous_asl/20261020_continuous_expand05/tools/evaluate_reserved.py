"""One post-freeze evaluation of fresh reserved recordings; no fitting or selection."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from core import RUN, SOURCE, ACQUIRED, SCHEMA, read, write, sha
from ctc_experiment import CausalCTC, evaluate, seeds


def main():
    target = RUN / 'final_reserved_test.json'
    if target.exists():
        print('RESERVED_TEST_ALREADY_EVALUATED_NO_REPEAT')
        return
    frozen = read(RUN / 'model_selection_freeze.json')
    if sha(RUN / 'data/processed/train_dev/manifest.json') != frozen['manifest_sha256'] or sha(RUN / 'tools/train_expanded.py') != frozen['trainer_sha256']:
        raise ValueError('FROZEN_FITTING_CONTRACT')
    plan = read(RUN / 'dataset_plan.json')
    if sha(RUN / 'data/reserved_test/manifest.json') != plan['reserved_test_manifest_sha256']:
        raise ValueError('RESERVED_MANIFEST_CHANGED')
    rows = read(RUN / 'data/reserved_test/manifest.json')
    train_dev = read(RUN / 'data/processed/train_dev/manifest.json')
    old_test = read(SOURCE / 'data/processed/test/manifest.json')
    for field in ('sample_id', 'source_url', 'raw_sha256', 'decoded_pixel_sha256', 'source_group'):
        other_field = 'source_video_sha256' if field == 'raw_sha256' else field
        old_values = {r[other_field] for r in old_test} if field in ('sample_id', 'source_url', 'raw_sha256', 'source_group') else set()
        if {r[field] for r in rows} & ({r[field] for r in train_dev} | old_values):
            raise ValueError('RESERVED_RECORDING_OVERLAP:' + field)
    data = {}
    for row in rows:
        if sha(row['feature_path']) != row['feature_sha256']:
            raise ValueError('RESERVED_FEATURE_HASH')
        with np.load(row['feature_path'], allow_pickle=False) as f:
            if str(f['schema_version'].item()) != SCHEMA:
                raise ValueError('RESERVED_FEATURE_SCHEMA')
            x = f['X'].copy()
            ts = f['timestamp_ms']
        if x.ndim != 2 or x.shape != (len(ts), 378) or not np.isfinite(x).all() or not np.all(np.diff(ts) > 0):
            raise ValueError('RESERVED_FEATURE_VALUES')
        data[row['sample_id']] = x
    seeds(20261020)
    results = {}
    # Both old pairs stay immutable. All model decisions precede this first test scoring.
    entries = [('expand03_reference48', SOURCE, 'reference48'), ('expand03_compact24', SOURCE, 'compact24'),
               ('expand04_reference48', ACQUIRED, 'reference48'), ('expand04_compact24', ACQUIRED, 'compact24'),
               ('expand05_reference48', RUN, 'reference48'), ('expand05_compact24', RUN, 'compact24')]
    for label, root, name in entries:
        info = read(root / 'model_selection_freeze.json')['models'][name]
        base = root / 'models' / name
        if sha(base / 'training_status.json') != info['status_sha256'] or sha(base / 'config.json') != info['config_sha256']:
            raise ValueError('POST_FREEZE_MODEL_CHANGED:' + label)
        path = base / info['pointer']['path']
        if sha(path) != info['pointer']['sha256']:
            raise ValueError('FROZEN_CHECKPOINT_HASH:' + label)
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = CausalCTC(len(state['vocabulary']), state['config']['channels'], state['model']['mean'].numpy(), state['model']['std'].numpy())
        model.load_state_dict(state['model'])
        result = evaluate(model, rows, data, state['vocabulary'], 'cpu')
        result.update(checkpoint_path=str(path), checkpoint_sha256=info['pointer']['sha256'], vocabulary_size=len(state['vocabulary']),
                      scope='Fresh raw recordings, source-group-disjoint, shared historical collections and incomplete signer IDs; diagnostic only')
        results[label] = result
    write(target, {'status': 'PASS_POST_FREEZE_RESERVED_TEST', 'created_utc': datetime.now(timezone.utc).isoformat(),
          'test_rows': len(rows), 'models': results, 'selection_freeze_sha256': sha(RUN / 'model_selection_freeze.json'),
          'reserved_manifest_sha256': sha(RUN / 'data/reserved_test/manifest.json'),
          'old_exposed14_evaluated': False, 'test_used_for_selection': False, 'signer_independent': False,
          'now_exposed_for_future_work': True,
          'scope_limit': 'Metadata may identify some participants, but original training pool is incompletely mapped; collection and known signer overlap prohibit independence claims.'})
    print(json.dumps({'status': 'PASS_POST_FREEZE_RESERVED_TEST', 'rows': len(rows), 'wer': {name: r['gloss_wer'] for name, r in results.items()}}))


if __name__ == '__main__':
    main()
