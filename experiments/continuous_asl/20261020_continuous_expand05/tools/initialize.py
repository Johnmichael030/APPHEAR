"""Stage only accepted unique recordings; quarantine all protected data roles."""
from collections import Counter
import json
from core import RUN, ROOT, SOURCE, ACQUIRED, SCHEMA, read, write, sha


def main():
    if (RUN / 'dataset_plan.json').exists():
        plan = read(RUN / 'dataset_plan.json')
        if sha(RUN / 'data/processed/train_dev/manifest.json') != plan['manifest_sha256']:
            raise ValueError('EXISTING_MANIFEST_CHANGED')
        print('STAGED_MANIFEST_EXISTS_VERIFIED_NO_REWRITE')
        return
    catalog = read(ACQUIRED / 'full_index_v2/preparation/manifest.json')
    quality = read(ACQUIRED / 'full_index_v2/preparation/quality.json')
    rows, reserved, excluded = [], [], []
    for original in catalog:
        if not original['unique_valid']:
            excluded.append({'sample_id': original['sample_id'], 'reason': original['exclusion_reason']})
            continue
        row = {**original, 'catalog_role': original['split'], 'schema_version': SCHEMA,
               'source_video_sha256': original['raw_sha256'], 'reuse_feature_cache': True}
        if original['split'] in ('TRAIN_POOL_NEW_FIT', 'TRAIN_POOL_NOT_FIT', 'FROZEN_DEV_REFERENCE'):
            row['split'] = 'dev' if original['split'] == 'FROZEN_DEV_REFERENCE' else 'train'
            rows.append(row)
        elif original['split'] == 'NEW_TEST_RESERVED_UNTOUCHED':
            row['split'] = 'test'
            reserved.append(row)
        else:
            excluded.append({'sample_id': original['sample_id'], 'reason': 'PROTECTED_' + original['split']})
    counts = dict(Counter(r['split'] for r in rows))
    if counts.get('dev') != 9 or counts.get('train', 0) <= 181 or not reserved:
        raise ValueError('INSUFFICIENT_EXPANSION_OR_NO_RESERVED_TEST')
    for key in ('sample_id', 'source_url', 'raw_sha256', 'decoded_pixel_sha256', 'source_group'):
        if len({r[key] for r in rows + reserved}) != len(rows + reserved):
            raise ValueError('FIT_DEV_TEST_OVERLAP:' + key)
    old_dev = [r for r in read(SOURCE / 'data/processed/train_dev/manifest.json') if r['split'] == 'dev']
    now_dev = [r for r in rows if r['split'] == 'dev']
    if [(r['sample_id'], r['glosses'], r['feature_sha256']) for r in old_dev] != [(r['sample_id'], r['glosses'], r['feature_sha256']) for r in now_dev]:
        raise ValueError('ORIGINAL_DEVELOPMENT_SPLIT_CHANGED')
    write(RUN / 'data/processed/train_dev/manifest.json', rows)
    write(RUN / 'data/processed/train_dev/quality.json', {'status': 'PASS', 'source_media_hash_duplicates': 0,
          'exact_decoded_duplicates': [], 'source_quality': str(ACQUIRED / 'full_index_v2/preparation/quality.json'),
          'accepted_subset_only': True, 'source_status': quality['status'], 'rejected': quality['failures']})
    write(RUN / 'data/reserved_test/manifest.json', reserved)
    write(RUN / 'data/quarantined.json', excluded)
    plan = read(ACQUIRED / 'training_plan.v2.json')
    plan['baseline'] = 'Immutable expand03 reference48, same9DEV and decoder; expanded training data and TRAIN-only vocabulary/normalization change. No architecture-only comparison.'
    plan['task'] = 'Literal source-main-gloss-token CTC including authentic fingerspelled-word targets, retained as complete source labels'
    write(RUN / 'training_plan.v2.json', plan)
    write(RUN / 'dataset_plan.json', {
        'schema': 'apphear.continuous_asl.expanded_fit.v2', 'fit_counts': counts, 'reserved_test_count': len(reserved),
        'manifest_sha256': sha(RUN / 'data/processed/train_dev/manifest.json'),
        'reserved_test_manifest_sha256': sha(RUN / 'data/reserved_test/manifest.json'),
        'source_manifest_sha256': sha(ACQUIRED / 'full_index_v2/preparation/manifest.json'),
        'source_plan_sha256': sha(ACQUIRED / 'full_index_v2/plan.json'), 'source_preparation_quality': quality,
        'signer_independent': False,
        'test_scope': 'Fresh raw recordings reserved before this fit; same historical10d/10q collections and incomplete signer metadata, not signer/collection-independent. Old exposed14 never evaluated here.',
        'label_policy': read(ACQUIRED / 'full_index_v2/plan.json')['target_policy'], 'rows': rows})
    source = (ACQUIRED / 'tools/train_expanded.py').read_text(encoding='utf-8')
    source = source.replace('SOURCE, OLD, SCHEMA', 'SOURCE, ACQUIRED, OLD, SCHEMA')
    before = 'RUN in path.parents or SOURCE in path.parents or OLD in path.parents'
    if before not in source:
        raise ValueError('TRAINER_CACHE_PORT_ANCHOR')
    source = source.replace(before, 'RUN in path.parents or SOURCE in path.parents or ACQUIRED in path.parents or OLD in path.parents')
    (RUN / 'tools/train_expanded.py').write_text(source, encoding='utf-8')
    write(RUN / 'code_provenance.json', {'source': str(ACQUIRED / 'tools/train_expanded.py'),
          'source_sha256': sha(ACQUIRED / 'tools/train_expanded.py'), 'new_trainer_sha256': sha(RUN / 'tools/train_expanded.py'),
          'change': 'Allow verified acquisition04 feature paths; unchanged fitting/model/decoding implementation and bounded hyperparameters.',
          'model_and_decoder_sha256': sha(ROOT / 'tools/continuous_asl/ctc_experiment.py')})
    # Same verified wrapper and DEV comparison logic; globals bind this new output.
    for relative in ('Invoke-Research.ps1', 'tools/freeze_compare.py'):
        content = (ACQUIRED / relative).read_text(encoding='utf-8')
        if relative == 'tools/freeze_compare.py':
            content = content.replace('new matched pair shares181TRAIN/vocab/normalization. Historical100TRAIN baseline has117rather than143tokens',
                                      'new matched pair shares its full TRAIN partition/vocab/normalization. Historical100TRAIN baseline has117tokens; each model vocabulary size is recorded')
        (RUN / relative).write_text(content, encoding='utf-8')
    print(json.dumps({'status': 'PASS_STAGED_EXPANDED_FIT', 'fit': counts, 'reserved_test': len(reserved), 'quarantined': len(excluded)}))


if __name__ == '__main__':
    main()
