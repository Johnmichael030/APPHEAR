"""Reuse all verified data and trainer logic; create a new expanded fitting contract."""
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from core import RUN, ROOT, SOURCE, SCHEMA, read, write, sha


def main():
    if (RUN / 'dataset_plan.json').exists():
        plan = read(RUN / 'dataset_plan.json')
        provenance = read(RUN / 'code_provenance.json')
        if sha(RUN / 'data/processed/train_dev/manifest.json') != plan['manifest_sha256'] or sha(RUN / 'tools/train_expanded.py') != provenance['new_trainer_sha256']:
            raise ValueError('EXISTING_INITIALIZATION_HASH_MISMATCH')
        print(json.dumps({'status': 'EXISTING_INITIALIZATION_VERIFIED_NO_REWRITE', 'fit_counts': plan['fit_counts']}))
        return
    catalog = read(SOURCE / 'expanded_catalog_v2/manifest.json')
    quality = read(SOURCE / 'expanded_catalog_v2/quality.json')
    if quality['status'] != 'PASS' or quality['unique_valid'] != 223:
        raise ValueError('SOURCE_CATALOG_QUALITY')
    # Guard every completed source artifact's size/mtime and its signed hash inventory.
    inventory = read(SOURCE / 'artifact_manifest.json')
    protected = [{'path': str(SOURCE / r['path']), 'size': (SOURCE / r['path']).stat().st_size,
                  'mtime_ns': (SOURCE / r['path']).stat().st_mtime_ns} for r in inventory['files']]
    protected += read(SOURCE / 'preservation_baseline.json')['metadata_snapshot']
    write(RUN / 'preservation_baseline.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_artifact_manifest': str(SOURCE / 'artifact_manifest.json'),
        'source_artifact_manifest_sha256': sha(SOURCE / 'artifact_manifest.json'),
        'source_artifact_manifest_mtime_ns': (SOURCE / 'artifact_manifest.json').stat().st_mtime_ns,
        'metadata_snapshot': protected,
        'source_preservation_evidence': str(SOURCE / 'preservation_final.json')})
    rows = []
    for original in catalog:
        if original['split'] not in ('TRAIN_POOL_NOT_FIT', 'FROZEN_DEV_REFERENCE'):
            continue
        row = {**original, 'catalog_role': original['split'],
               'split': 'train' if original['split'] == 'TRAIN_POOL_NOT_FIT' else 'dev',
               'source_video_sha256': original['raw_sha256'], 'schema_version': SCHEMA,
               'reuse_feature_cache': True}
        rows.append(row)
    if Counter(r['split'] for r in rows) != Counter(train=181, dev=9):
        raise ValueError('EXPANDED_PARTITION_COUNTS')
    for partition in ('train', 'dev'):
        for row in (r for r in rows if r['split'] == partition):
            if sha(row['feature_path']) != row['feature_sha256']:
                raise ValueError('FEATURE_HASH')
            with np.load(row['feature_path'], allow_pickle=False) as f:
                x, ts = f['X'], f['timestamp_ms']
                if str(f['schema_version'].item()) != SCHEMA or x.shape != (len(ts), 378) or x.dtype != np.float32 or not np.isfinite(x).all():
                    raise ValueError('FEATURE_CONTRACT')
                if not np.isfinite(ts).all() or not np.all(np.diff(ts) > 0):
                    raise ValueError('TIMESTAMPS')
                if len(x) < len(row['glosses']) + sum(a == b for a, b in zip(row['glosses'], row['glosses'][1:])):
                    raise ValueError('CTC_FEASIBILITY')
    write(RUN / 'data/processed/train_dev/manifest.json', rows)
    write(RUN / 'data/processed/train_dev/quality.json', {
        'status': 'PASS', 'source_media_hash_duplicates': 0, 'exact_decoded_duplicates': [],
        'processed': len(rows), 'new_feature_extraction': False,
        'source_evidence': str(SOURCE / 'expanded_catalog_v2/quality.json')})
    write(RUN / 'dataset_plan.json', {
        'schema': 'apphear.continuous_asl.expanded_fit.v1', 'fit_counts': {'train': 181, 'dev': 9},
        'source_catalog': str(SOURCE / 'expanded_catalog_v2/manifest.json'),
        'source_catalog_sha256': sha(SOURCE / 'expanded_catalog_v2/manifest.json'),
        'source_plan_sha256': sha(SOURCE / 'dataset_plan.json'),
        'manifest_sha256': sha(RUN / 'data/processed/train_dev/manifest.json'),
        'retained_not_fit': {'prior_validation_quarantine': 19, 'previously_exposed_test_locked': 14},
        'untouched_test': 'Reserve any newly eligible independent source before fitting it; do not reuse exposed14 as a fresh final test.',
        'signer_independent': False, 'new_data_acquisition_required_before_initial_fit': False,
        'label_policy': 'Verbatim source main-gloss tokens with case, punctuation, locus and grammar retained.',
        'rows': rows})
    plan = read(SOURCE / 'training_plan.v2.json')
    plan['baseline'] = 'Frozen expand03 reference48 on same9DEV; TRAIN100 versus new181. Vocabulary/normalization refitted on TRAIN only; no architecture-only claim.'
    plan.pop('prior_metadata_correction', None)
    plan['selection'] = 'Fixed9DEV WER then loss; no test-based selection.30epochs/180seconds,patience10.'
    write(RUN / 'training_plan.v2.json', plan)
    original = (SOURCE / 'tools/model_run.py').read_text(encoding='utf-8')
    functions = {node.name: ast.get_source_segment(original, node) for node in ast.parse(original).body if isinstance(node, ast.FunctionDef)}
    dataset = functions['dataset'].replace("if not (RUN in path.parents or OLD in path.parents):", "if not (RUN in path.parents or SOURCE in path.parents or OLD in path.parents):")
    train = functions['train']
    before = "if len(fit)!=100 or len(dev)!=9: raise ValueError('FROZEN_SPLIT_COUNTS')"
    after = "if dict(Counter(r['split'] for r in rows))!=read(RUN/'dataset_plan.json')['fit_counts']: raise ValueError('FROZEN_SPLIT_COUNTS')"
    if before not in train:
        raise ValueError('TRAINER_PORT_ANCHOR_CHANGED')
    train = train.replace(before, after)
    header = '''"""Expanded fitting: port of completed trainer; no frozen-source writes or TEST evaluation."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from core import RUN, ROOT, SOURCE, OLD, SCHEMA, SEED, read, write, sha
from ctc_experiment import CausalCTC, batch, evaluate, seeds
from replay_profile import memory
'''
    tail = '''
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',choices=('smoke','reference48','compact24'),required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    status=RUN/'models'/args.name/'training_status.json'
    if status.exists():
        saved=read(status);pointer=read(RUN/'models'/args.name/'checkpoints/best.json')
        if sha(RUN/'models'/args.name/pointer['path'])!=pointer['sha256']:raise ValueError('COMPLETED_CHECKPOINT_HASH')
        print(json.dumps({'status':'VERIFIED_COMPLETE_NO_TRAINING','name':args.name,'epochs':saved['completed_epochs']}))
    else:train(args.name,args.resume)
'''
    target = RUN / 'tools/train_expanded.py'
    target.write_text(header + '\n\n' + '\n\n'.join([dataset, functions['atomic_json'], functions['checkpoint'], train]) + '\n' + tail, encoding='utf-8')
    write(RUN / 'code_provenance.json', {
        'trainer_source': str(SOURCE / 'tools/model_run.py'), 'source_sha256': sha(SOURCE / 'tools/model_run.py'),
        'new_trainer_sha256': sha(target),
        'reused_functions': ['dataset', 'atomic_json', 'checkpoint', 'train'],
        'explicit_changes': ['Allow verified SOURCE feature cache paths', 'Frozen partition counts are read from new181/9 plan',
                             'CLI verifies completed training status instead of starting duplicate fits', 'Old freeze/TEST functions omitted'],
        'model_and_decoder': str(ROOT / 'tools/continuous_asl/ctc_experiment.py'),
        'model_and_decoder_sha256': sha(ROOT / 'tools/continuous_asl/ctc_experiment.py')})
    print(json.dumps({'status': 'PASS', 'train': 181, 'dev': 9, 'new_raw_this_initialization': 0, 'cached_features_reused': 190}))


if __name__ == '__main__':
    main()
