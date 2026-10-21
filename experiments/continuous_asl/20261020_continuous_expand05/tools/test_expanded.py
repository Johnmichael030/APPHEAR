from collections import Counter
import numpy as np
from core import RUN, ROOT, SOURCE, ACQUIRED, read, sha
from train_expanded import dataset


def test_exact_literal_targets_including_fs_words():
    rows, _ = dataset()
    source = {r['sample_id']: r for r in read(ACQUIRED / 'full_index_v2/preparation/manifest.json')}
    assert any(any(s.startswith('fs-') for s in r['glosses']) for r in rows if r['split'] == 'train')
    for row in rows:
        assert row['glosses'] == source[row['sample_id']]['glosses']
        assert ' '.join(row['glosses']) == row['gloss_text']


def test_reserved_and_quarantined_samples_never_fit():
    rows, _ = dataset()
    reserved = read(RUN / 'data/reserved_test/manifest.json')
    quarantined = read(RUN / 'data/quarantined.json')
    assert not {r['sample_id'] for r in rows} & {r['sample_id'] for r in reserved + quarantined}
    for key in ('sample_id', 'source_url', 'raw_sha256', 'decoded_pixel_sha256', 'source_group'):
        assert len({r[key] for r in rows + reserved}) == len(rows + reserved)


def test_original9dev_and_collection_separation():
    rows, _ = dataset()
    before = [r for r in read(SOURCE / 'data/processed/train_dev/manifest.json') if r['split'] == 'dev']
    now = [r for r in rows if r['split'] == 'dev']
    assert [(r['sample_id'], r['glosses'], r['feature_sha256']) for r in before] == [(r['sample_id'], r['glosses'], r['feature_sha256']) for r in now]
    assert not {r['collection'] for r in rows if r['split'] == 'train'} & {r['collection'] for r in now}


def test_timestamps_and_ctc_feasibility_of_all_fitted_samples():
    rows, data = dataset()
    assert dict(Counter(r['split'] for r in rows)) == read(RUN / 'dataset_plan.json')['fit_counts']
    for row in rows:
        with np.load(row['feature_path'], allow_pickle=False) as f:
            ts = f['timestamp_ms']
            assert len(ts) == len(data[row['sample_id']]) and np.isfinite(ts).all() and np.all(np.diff(ts) > 0)
        repeats = sum(a == b for a, b in zip(row['glosses'], row['glosses'][1:]))
        assert len(data[row['sample_id']]) >= len(row['glosses']) + repeats


def test_frozen_hyperparameters_and_encoder61():
    original = read(ACQUIRED / 'training_plan.v2.json')
    current = read(RUN / 'training_plan.v2.json')
    for key in ('models', 'seed', 'batch', 'gradient_accumulation', 'lr', 'weight_decay', 'patience', 'mixed_precision', 'schema', 'effective_receptive_field_observations'):
        assert current[key] == original[key]
    assert current['effective_receptive_field_observations'] == 61
    assert sha(ROOT / 'tools/continuous_asl/ctc_experiment.py') == read(RUN / 'code_provenance.json')['model_and_decoder_sha256']
