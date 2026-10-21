from collections import Counter
import numpy as np
import torch
from core import RUN, ROOT, SOURCE, read, sha
from train_expanded import dataset
from ctc_experiment import CausalCTC, edit_counts


def test_expanded_partitions_and_source_joins():
    rows, _ = dataset()
    source = {r['sample_id']: r for r in read(SOURCE / 'expanded_catalog_v2/manifest.json')}
    assert Counter(r['split'] for r in rows) == Counter(train=181, dev=9)
    for key in ('sample_id', 'source_url', 'raw_sha256', 'decoded_pixel_sha256'):
        assert len({r[key] for r in rows}) == len(rows)
    train = {r['collection'] for r in rows if r['split'] == 'train'}
    dev = {r['collection'] for r in rows if r['split'] == 'dev'}
    assert not train & dev
    for row in rows:
        expected = source[row['sample_id']]
        assert expected['split'] in ('TRAIN_POOL_NOT_FIT', 'FROZEN_DEV_REFERENCE')
        assert row['glosses'] == expected['glosses'] and row['gloss_text'] == expected['gloss_text']
        assert row['feature_sha256'] == expected['feature_sha256']


def test_frozen_dev_identical_to_original_comparison():
    rows, _ = dataset()
    before = [r for r in read(SOURCE / 'data/processed/train_dev/manifest.json') if r['split'] == 'dev']
    now = [r for r in rows if r['split'] == 'dev']
    assert [(r['sample_id'], r['glosses'], r['feature_sha256']) for r in now] == [(r['sample_id'], r['glosses'], r['feature_sha256']) for r in before]


def test_timestamps_and_ctc_feasibility():
    rows, data = dataset()
    for row in rows:
        with np.load(row['feature_path'], allow_pickle=False) as f:
            ts = f['timestamp_ms']
            assert np.isfinite(ts).all() and np.all(np.diff(ts) > 0)
        repeated = sum(a == b for a, b in zip(row['glosses'], row['glosses'][1:]))
        assert len(data[row['sample_id']]) >= len(row['glosses']) + repeated


def test_exposed_test_and_prior_dev_quarantine_never_fit():
    rows, _ = dataset()
    catalog = read(SOURCE / 'expanded_catalog_v2/manifest.json')
    excluded = {r['sample_id'] for r in catalog if r['split'] in ('FROZEN_TEST_EXPOSED_LOCKED', 'PRIOR_DEV_QUARANTINE')}
    assert len(excluded) == 33 and not excluded & {r['sample_id'] for r in rows}


def test_same_causal_encoder_retains_all_features_and_rf61():
    torch.set_num_threads(2)
    model = CausalCTC(10, 24, np.zeros(378, np.float32), np.ones(378, np.float32)).eval()
    assert read(RUN / 'training_plan.v2.json')['effective_receptive_field_observations'] == 1 + 2 * 2 * sum((1, 2, 4, 8)) == 61
    x = torch.randn(1, 80, 378)
    with torch.inference_mode():
        assert torch.allclose(model(x)[:, :40], model(x[:, :40]), atol=1e-6, rtol=0)


def test_original_edit_count_protocol_and_code_provenance():
    counts = edit_counts(['A', 'B', 'C'], ['A', 'X'])
    assert counts['reference_tokens'] == 3
    assert sum(counts[k] for k in ('substitutions', 'deletions', 'insertions')) == 2
    provenance = read(RUN / 'code_provenance.json')
    assert sha(ROOT / 'tools/continuous_asl/ctc_experiment.py') == provenance['model_and_decoder_sha256']
    assert sha(RUN / 'tools/train_expanded.py') == provenance['new_trainer_sha256']
