"""New continuous-data experiment; never write outside this run."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlsplit, urlunsplit

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[2]
OLD = ROOT / 'experiments/continuous_asl/20261019T155428Z_cslr01'
SECOND = ROOT / 'experiments/continuous_asl/20261020T004837_separate02'
sys.path.insert(0, str(ROOT / 'tools/continuous_asl'))
sys.path.insert(0, str(ROOT))
SCHEMA = 'apphear.continuous_asl.candidate01_causal378.v1'
SEED = 20261020

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, obj):
    path = Path(path)
    assert RUN in path.resolve().parents
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2); f.write('\n')

def initialize():
    if (RUN / 'dataset_plan.json').exists():
        raise ValueError('PLAN_ALREADY_FROZEN')
    guards = []
    for p in (SECOND/'artifact_manifest.final.json', SECOND/'continuation_20261020T013509/artifact_hashes.json',
              SECOND/'pilot_package_20261020T014259/artifact_hashes.json', OLD/'reproducibility_manifest.json'):
        guards.append({'path': str(p), 'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns})
    records = read(SECOND/'protected_sources.json') + read(SECOND/'external_inputs_protected.json')
    snapshot = [{'path': r['path'], 'size': Path(r['path']).stat().st_size, 'mtime_ns': Path(r['path']).stat().st_mtime_ns} for r in records]
    write(RUN/'preservation_baseline.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'guards': guards,
        'metadata_snapshot': snapshot, 'full_hash_evidence_reused': str(SECOND/'continuation_20261020T013509/evidence/preservation.json')})
    index = next((OLD/'data/extracted/ncslgr_video_index').rglob('files_by_xml_file.csv'))
    rows = list(csv.DictReader(index.open(encoding='utf-8-sig', newline='')))
    uses = defaultdict(set)
    for r in rows:
        uses[r['Compressed MOV file']].add((r['XML file'], r['Occurs in utterance id'], r['main gloss']))
    lexical = re.compile(r'(?:\(2h\))?[A-Za-z][A-Za-z0-9_:+().-]*\+*$')
    candidates, seen = [], set()
    rejected = Counter()
    for r in rows:
        url = r['Compressed MOV file']; parts = urlsplit(url); tokens = r['main gloss'].split()
        identity = (r['XML file'], r['Occurs in utterance id'])
        if r['Perspective/Camera id'] != '0': rejected['other_view'] += 1; continue
        if len(uses[url]) != 1: rejected['shared_video_without_utterance_bounds'] += 1; continue
        if parts.hostname != 'csr.bu.edu' or not parts.path.endswith('.mov'): rejected['not_public_small_movie'] += 1; continue
        if not 2 <= len(tokens) <= 12 or not all(lexical.fullmatch(t) and not any(s in t.lower() for s in ('cl:', 'fs-', 'fingerspell')) for t in tokens):
            rejected['complex_quoted_or_fingerspelled_labels'] += 1; continue
        if identity in seen: continue
        seen.add(identity)
        sample = 'ncslgr_' + hashlib.sha256(('|'.join(identity)).encode()).hexdigest()[:16]
        candidates.append({'sample_id': sample, 'dataset': 'NCSLGR', 'collection': identity[0], 'utterance_id': identity[1],
            'source_group': r['Video sequence id'], 'signer_id': None, 'original_split': 'NOT_PROVIDED_IN_PUBLIC_INDEX',
            'source_url_original': url, 'source_url': urlunsplit(('https', parts.netloc, parts.path, parts.query, '')),
            'gloss_text': r['main gloss'], 'glosses': tokens, 'annotation_type': 'AUTHENTIC_INDEX_MAIN_GLOSS_TOKEN_SEQUENCE',
            'individual_sign_boundaries_available': False, 'license_status': 'PRIVATE_RESEARCH_TERMS_REVIEWED_NO_REDISTRIBUTION',
            'license_reference': 'https://www.bu.edu/asllrp/dai-terms.html'})
    old = read(OLD/'data/processed/full/manifest.json')
    old_train = [r for r in old if r['split'] == 'train']
    quarantine = {r['collection'] for r in old if r['split'] == 'val'}
    old_by_id = {r['sample_id']: r for r in old_train}
    train_pool = [r for r in candidates if r['collection'] not in quarantine | {'ncslgr10p.xml', 'ncslgr10q.xml'} and r['sample_id'] not in old_by_id]
    train_pool.sort(key=lambda r: hashlib.sha256((str(SEED)+r['sample_id']).encode()).hexdigest())
    selected = [{**r, 'split': 'train'} for r in old_train]
    selected += [{**r, 'split': 'train'} for r in train_pool[:100-len(old_train)]]
    selected += [{**r, 'split': 'dev'} for r in candidates if r['collection'] == 'ncslgr10p.xml']
    selected += [{**r, 'split': 'test'} for r in candidates if r['collection'] == 'ncslgr10q.xml']
    for r in selected:
        if r['sample_id'] in old_by_id:
            original = old_by_id[r['sample_id']]
            r.update(reuse=True, video_path=str(OLD/'data/raw/ncslgr/videos'/f"{r['sample_id']}.mov"),
                     reuse_feature_path=str(OLD/original['feature_path']), reuse_feature_sha256=original['feature_sha256'])
        else:
            r.update(reuse=False, video_path=str(RUN/'data/raw/ncslgr/videos'/f"{r['sample_id']}.mov"))
        r['original_split'] = 'NOT_PROVIDED_IN_PUBLIC_INDEX'
        r['annotation_type'] = 'AUTHENTIC_INDEX_MAIN_GLOSS_TOKEN_SEQUENCE'
        r['license_status'] = 'PRIVATE_RESEARCH_TERMS_REVIEWED_NO_REDISTRIBUTION'
        r['cadence_max_hz'] = 10
    groups = {s: {r['collection'] for r in selected if r['split'] == s} for s in ('train','dev','test')}
    for a, b in (('train','dev'), ('train','test'), ('dev','test')):
        assert not groups[a] & groups[b]
        assert not {r['source_group'] for r in selected if r['split'] == a} & {r['source_group'] for r in selected if r['split'] == b}
    assert len({r['sample_id'] for r in selected}) == len(selected)
    assert all(r['collection'] not in quarantine for r in selected)
    assert len([r for r in selected if r['split'] == 'train']) == 100
    plan = {'schema': 'apphear.continuous_expansion.plan.v1', 'seed': SEED, 'index_path': str(index), 'index_sha256': sha(index),
        'official_source': 'https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html',
        'eligible_candidates': len(candidates), 'filter_rejections': dict(rejected),
        'split_counts': dict(Counter(r['split'] for r in selected)), 'quarantined_prior_dev_collections': sorted(quarantine),
        'custom_split_collections': {s: sorted(g) for s, g in groups.items()}, 'signer_independence': 'UNVERIFIED_IDS_UNAVAILABLE',
        'download_max_bytes_per_file': 8388608, 'download_max_total_bytes': 1073741824, 'total_run_budget_bytes': 4294967296,
        'disk_headroom_bytes': 10737418240, 'disk_free_before': shutil.disk_usage(RUN).free,
        'preprocessing': 'Identical existing causal378/max10Hz policy; full genuine utterances, no synthetic concatenation',
        'target_policy': 'Verbatim source main-gloss whitespace tokens incl. case/locus/grammar notation and HOLD; not normalized lexical-only glosses',
        'rows': selected}
    assert plan['disk_free_before'] > plan['disk_headroom_bytes'] + plan['total_run_budget_bytes']
    assert sum(not r['reuse'] for r in selected) * plan['download_max_bytes_per_file'] < plan['download_max_total_bytes']
    write(RUN/'dataset_plan.json', plan)
    write(RUN/'training_plan.json', {'schema': SCHEMA, 'seed': SEED, 'task': 'source-main-gloss-token CTC diagnostic',
        'models': {'smoke': {'channels':24, 'max_epochs':1, 'wall_seconds':60},
                   'reference48': {'channels':48, 'max_epochs':30, 'wall_seconds':180},
                   'compact24': {'channels':24, 'max_epochs':30, 'wall_seconds':180}},
        'batch':4, 'gradient_accumulation':2, 'lr':0.001, 'weight_decay':0.0001, 'gradient_clip':5., 'patience':10,
        'mixed_precision':False, 'max_vram_allocated_bytes':2684354560, 'minimum_free_vram_bytes':268435456,
        'effective_receptive_field_observations':121, 'prior_metadata_correction':'Eight kernel3 convolutions at dilations1,1,2,2,4,4,8,8 imply RF121, not previously recorded61.',
        'selection':'DEV WER then DEV loss; fixed greedy CTC decoder; TEST used only after checkpoint freeze',
        'baseline':'New matched reference48 trained first and frozen; additionally preserve legacy48 unchanged as historical reference',
        'normalization':'TRAIN mean/std only, same arrays for both model sizes', 'production_enabled':False})
    print(json.dumps({k: plan[k] for k in ('eligible_candidates','split_counts','custom_split_collections','disk_free_before')}))

def prepare(partition):
    import cv2
    import numpy as np
    from prepare_features import make_raw
    from pipeline.perception_vnext.offline_candidate01 import Candidate01Offline, iter_video_frames
    from pipeline.perception_vnext.recognition_schema import save_raw_sequence
    from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features
    if partition == 'test' and not (RUN/'model_selection_freeze.json').is_file(): raise ValueError('HOLDOUT_LOCKED')
    out = RUN/'data/processed'/partition
    if out.exists(): raise ValueError('PREPROCESSING_OUTPUT_EXISTS')
    out.mkdir(parents=True)
    os.environ['KERAS_HOME'] = str(RUN/'cache/keras'); os.environ['TF_CPP_MIN_LOG_LEVEL']='2'; cv2.setNumThreads(2)
    plan = read(RUN/'dataset_plan.json')
    requested = [r for r in plan['rows'] if (r['split']=='test') == (partition=='test')]
    previous_decodes = {r['sample_id']:r for r in read(OLD/'logs/visual_duplicates.json')['records']}
    timings, results, failures, decoded = [], [], [], []
    started = time.perf_counter()
    with Candidate01Offline(ROOT/'android/app/src/main/assets/perception_vnext/dwpose_t_singlepass.onnx',
                            ROOT/'android/app/src/main/assets/perception_vnext/hand_landmarks_detector.tflite') as runtime:
        for row in requested:
            try:
                sample = row['sample_id']; video=Path(row['video_path']); receipt=read(str(video)+'.receipt.json')
                if sha(video) != receipt['sha256'] or receipt['source_url'] != row['source_url']: raise ValueError('MEDIA_IDENTITY')
                records=[]; last=-1e9; pixel=hashlib.sha256(); total=0
                if row['reuse']:
                    cached=previous_decodes[sample]; total=cached['frames']; pixel_digest=cached['decoded_sha256']
                    meta={'native_fps':row['native_fps']}
                else:
                    frames, meta=iter_video_frames(video,None)
                    for idx, ms, source, rgb in frames:
                        total+=1; pixel.update(rgb[...,::-1].tobytes())
                        if ms < last+100-1e-6: continue
                        tick=time.perf_counter(); obs=runtime.process_rgb(rgb); timings.append((time.perf_counter()-tick)*1000)
                        records.append((idx,ms,source,obs)); last=ms
                    pixel_digest=pixel.hexdigest()
                decoded.append({'sample_id':sample,'split':row['split'],'decoded_pixel_sha256':pixel_digest,'frames':total,
                                'method':'all-frame decoded BGR SHA256; prior verified result reused if source media hash matches','reused':row['reuse']})
                if total < 2: raise ValueError('EMPTY_DECODE')
                if row['reuse']:
                    path=Path(row['reuse_feature_path'])
                    if sha(path) != row['reuse_feature_sha256']: raise ValueError('REUSED_FEATURE_HASH')
                else:
                    if len(records)<2: raise ValueError('EMPTY_FEATURES')
                    raw=make_raw({**row, 'source_video_sha256':receipt['sha256']},records,meta); raw.validate()
                    features=derive_causal_segmentation_features(raw)
                    # Existing feature recipe must be causal at the midpoint, without full-sequence lookahead.
                    from dataclasses import replace
                    mid=max(1,len(records)//2)
                    prefix=replace(raw,**{name:getattr(raw,name)[:mid] for name in raw.__dataclass_fields__ if name!='metadata'})
                    if not np.allclose(features[:mid],derive_causal_segmentation_features(prefix),atol=1e-6,rtol=0): raise ValueError('NONCAUSAL_RECIPE')
                    path=out/f'{sample}.features.npz'
                    save_raw_sequence(out/f'{sample}.raw.npz',raw)
                    np.savez_compressed(path,X=features,timestamp_ms=raw.timestamp_ms,schema_version=np.asarray(SCHEMA),
                        metadata_json=np.asarray(json.dumps({**row,**meta,'schema_version':SCHEMA})))
                with np.load(path,allow_pickle=False) as f:
                    if str(f['schema_version'].item()) != SCHEMA: raise ValueError('FEATURE_SCHEMA')
                    x=f['X']; ts=f['timestamp_ms']
                repeats=sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:]))
                if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=378 or not np.isfinite(x).all(): raise ValueError('FEATURE_VALUES')
                if len(ts)!=len(x) or not np.isfinite(ts).all() or not np.all(np.diff(ts)>0): raise ValueError('TIMING')
                if len(x)<len(row['glosses'])+repeats: raise ValueError('CTC_LENGTH')
                results.append({**row,'feature_path':str(path),'feature_sha256':sha(path),'source_video_sha256':receipt['sha256'],
                    'observations':len(x),'decoded_frames':total,'native_fps':meta['native_fps'],
                    'dimensions':row.get('dimensions') if row['reuse'] else np.asarray(records[0][3]['source_size']).tolist(),
                    'preprocessing_status':'PASS','schema_version':SCHEMA})
                print(json.dumps({'sample_id':sample,'split':row['split'],'observations':len(x),'reused':row['reuse']}),flush=True)
            except Exception as error:
                failures.append({'sample_id':row['sample_id'],'error':str(error)})
                break
    groups=defaultdict(list)
    for r in decoded: groups[r['decoded_pixel_sha256']].append(r['sample_id'])
    duplicates=[v for v in groups.values() if len(v)>1]
    if len(results)!=len(requested) or duplicates: failures.append({'error':'INCOMPLETE_OR_DUPLICATE_CORPUS'})
    report={'status':'PASS' if not failures else 'FAIL','requested':len(requested),'processed':len(results),'failures':failures,
        'exact_decoded_duplicates':duplicates,'decoded':decoded,'elapsed_seconds':time.perf_counter()-started,
        'new_observations_processed':len(timings),'extraction_ms_p50':float(np.percentile(timings,50)) if timings else None,
        'extraction_ms_p95':float(np.percentile(timings,95)) if timings else None,'split_counts':dict(Counter(r['split'] for r in results)),
        'source_media_hash_duplicates':len(results)-len({r['source_video_sha256'] for r in results}),
        'timing_limit':'Container PTS/fps-derived timestamps, not independently synchronized boundary ground truth; publisher notes some archive clips may play at half speed.',
        'training_eligible_for':'DIAGNOSTIC_CTC_ONLY','signer_independence':False}
    write(out/'manifest.json',results); write(out/'quality.json',report)
    print(json.dumps({k:report[k] for k in ('status','requested','processed','elapsed_seconds','extraction_ms_p50','extraction_ms_p95','failures')}))
    if failures or report['source_media_hash_duplicates']: raise SystemExit(2)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('plan','prepare'));p.add_argument('--partition',choices=('train_dev','test'),default='train_dev')
    a=p.parse_args()
    if a.phase=='plan': initialize()
    else: prepare(a.partition)
