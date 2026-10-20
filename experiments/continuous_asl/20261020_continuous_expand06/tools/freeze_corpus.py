"""Build the canonical reference manifest after all scheduled feature attempts.

Publisher label conventions define separate eligible pools. No videos copied,
no labels normalized, no invalid spans trimmed, no held-out data selected.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import time
import numpy as np
from core import RUN,FOUR,FIVE,SCHEMA,read,write,sha

def build(watch=False):
    folder=RUN/'canonical'
    if (folder/'quality.json').exists():
        q=read(folder/'quality.json')
        if sha(folder/'manifest.json')!=q['manifest_sha256']:raise ValueError('FROZEN_CANONICAL_MANIFEST_CHANGED')
        print('CANONICAL_ALREADY_FROZEN_NO_REWRITE');return
    plans=read(RUN/'asllrp2025_catalog/media_manifest.json');feature_dir=RUN/'data/processed/asllrp2025'
    start=time.monotonic();last=-1e9
    while True:
        done={p.name[:-len('.row.json')]:p for p in feature_dir.glob('*.row.json')}
        failed={p.name[:-len('.failure.json')]:p for p in feature_dir.glob('*.failure.json')}
        pending=[r['sample_id'] for r in plans if r['sample_id'] not in done and r['sample_id'] not in failed]
        if time.monotonic()-last>=60:
            print(json.dumps({'feature_ready_before_global_exclusions':len(done),'failed_attempts':len(failed),'pending':len(pending),'utc':datetime.now(timezone.utc).isoformat()}),flush=True);last=time.monotonic()
        if not pending:break
        if not watch:raise ValueError('PREPROCESSING_ACTIVE_OR_INCOMPLETE:'+str(len(pending)))
        if time.monotonic()-start>14400:raise TimeoutError('BOUNDED4H_WAIT; completed features retained; resume workers and command')
        time.sleep(5)
    boundary=read(RUN/'asllrp2025_catalog/annotation_boundary_audit.v2.json')
    forbidden=set(boundary['reject_sample_ids']);prior=read(FOUR/'full_index_v2/preparation/manifest.json')
    timed=read(RUN/'timed_preparation/manifest.json')
    old_dev={r['sample_id'] for r in read(FIVE/'data/processed/train_dev/manifest.json') if r['split']=='dev'}
    rows=[];rejections=[];known_ids=set();seen_raw={};seen_pixels={};seen_segments={}
    for row in prior:
        role='eligible_train' if row['split'] in ('TRAIN_POOL_NOT_FIT','TRAIN_POOL_NEW_FIT') else ('fixed_dev' if row['sample_id'] in old_dev else ('reserved_test' if row['split']=='NEW_TEST_RESERVED_UNTOUCHED' else 'quarantine'))
        value={**row,'canonical_role':role,'training_pool':'NCSLGR2_LITERAL_MAIN_GLOSS','source_catalog':'expand04_verified552','schema_version':SCHEMA}
        rows.append(value);known_ids.add(row['sample_id']);seen_raw[row['raw_sha256']]=row['sample_id'];seen_pixels[row['decoded_pixel_sha256']]=row['sample_id']
    for row in timed:
        if row['sample_id'] in known_ids:raise ValueError('TIMED_PRIOR_IDENTITY_OVERLAP')
        if row['raw_sha256'] in seen_raw:raise ValueError('TIMED_MEDIA_EQUALS_PREVIOUS_WHOLE_RECORDING')
        signature=(row['raw_sha256'],row['start_ms'],row['end_ms'])
        if signature in seen_segments:raise ValueError('DUPLICATE_TIMED_UTTERANCE')
        seen_segments[signature]=row['sample_id'];known_ids.add(row['sample_id'])
        rows.append({**row,'canonical_role':'eligible_train','training_pool':'NCSLGR2_LITERAL_MAIN_GLOSS','source_catalog':'expand06_verified_publisher_timed_slices','schema_version':SCHEMA})
    timed_raw={r['raw_sha256'] for r in timed};timed_pixels={r['source_decoded_pixel_sha256'] for r in timed}
    boundaries=defaultdict(list)
    for row in timed:boundaries[row['raw_sha256']].append((row['start_ms'],row['end_ms'],row['sample_id']))
    for spans in boundaries.values():
        spans.sort()
        if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):raise ValueError('TIMED_UTTERANCE_OVERLAP')
    for plan in plans:
        sample=plan['sample_id']
        if sample in failed:
            rejections.append({**read(failed[sample]),'stage':'FEATURE_OR_MEDIA_CONTRACT'});continue
        row=read(done[sample]);reason=None
        if sample in forbidden:reason='PUBLISHER_MASTER_FRAME_UTTERANCE_OVERLAP'
        elif sample in known_ids:reason='STABLE_ID_DUPLICATE'
        elif row['raw_sha256'] in seen_raw or row['raw_sha256'] in timed_raw:reason='RAW_MEDIA_HASH_DUPLICATE'
        elif row['decoded_pixel_sha256'] in seen_pixels or row['decoded_pixel_sha256'] in timed_pixels:reason='EXACT_DECODED_CONTENT_DUPLICATE'
        if reason:
            rejections.append({'sample_id':sample,'stage':'GLOBAL_DUPLICATE_OR_BOUNDARY','reason':reason});continue
        # Cross-source matching is annotation aware: source collection + master
        # video + utterance identity, plus hashes and original inclusive bounds.
        if row['glosses']!=plan['glosses'] or row['events']!=plan['events'] or row['raw_sha256']!=plan['raw_sha256']:raise ValueError('FEATURE_ANNOTATION_MEDIA_JOIN_CHANGED')
        path=Path(row['feature_path'])
        if sha(path)!=row['feature_sha256']:raise ValueError('FEATURE_HASH')
        with np.load(path,allow_pickle=False) as f:
            x=f['X'];ts=f['timestamp_ms']
            if str(f['schema_version'].item())!=SCHEMA or x.dtype!=np.float32 or x.shape!=(len(ts),378) or not np.isfinite(x).all() or not np.all(np.diff(ts)>0):raise ValueError('FEATURE_CONTRACT')
        if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('CTC_SEQUENCE_CONTRACT')
        known_ids.add(sample);seen_raw[row['raw_sha256']]=sample;seen_pixels[row['decoded_pixel_sha256']]=sample
        rows.append({**row,'canonical_role':'eligible_train','training_pool':'ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE','source_catalog':'official_ASLLRP2025_verified_utterance_archive','schema_version':SCHEMA})
    # Distinct source labels are retained; intentionally do not merge these
    # publisher versions into a single training vocabulary or normalization.
    counts=Counter(r['training_pool'] for r in rows if r['canonical_role']=='eligible_train')
    write(folder/'manifest.json',rows);write(folder/'rejections.json',rejections)
    result={'status':'PASS_VERIFIED_CANONICAL_REFERENCE_MANIFEST','created_utc':datetime.now(timezone.utc).isoformat(),
       'unique_records':len(rows),'eligible_train_by_label_pool':dict(counts),'eligible_train_total_across_separate_pools':sum(counts.values()),
       'canonical_role_counts':dict(Counter(r['canonical_role'] for r in rows)),
       'stage_counts':{'prior_verified_utterances':len(prior),'new_ncslgr_annotation_candidates':read(RUN/'depaul_catalog/plan.json')['new_annotation_matched_utterances'],
         'new_ncslgr_downloaded_media_files':len(list((RUN/'data/raw/ncslgr_media').glob('*.mov'))),
         'new_ncslgr_verified_feature_ready_utterances':len(timed),'ASLLRP_annotation_candidates':len(plans),
         'ASLLRP_downloaded_unique_media_files':len({r['raw_sha256'] for r in plans}),
         'ASLLRP_verified_annotation_media_joins':len(plans),'ASLLRP_decoded_with_saved_full_decode_evidence':len(done),
         'ASLLRP_feature_and_CTC_ready_before_global_exclusions':len(done),
         'ASLLRP_training_eligible_after_all_checks':counts['ASLLRP3_LITERAL_DOMINANT_REALIZED_OCCURRENCE']},
       'rejection_counts':dict(Counter(r['reason'] for r in rejections)),
       'manifest_sha256':sha(folder/'manifest.json'),'script_sha256':sha(__file__),
       'minimum_target':2000,'stretch_target':5000,'largest_single_label_pool':max(counts.values()),
       'single_pool_minimum_shortfall':max(0,2000-max(counts.values())),'single_pool_stretch_shortfall':max(0,5000-max(counts.values())),
       'label_scope':'Distinct literal publisher annotation conventions and dominant-hand scopes stay separate; no English-as-gloss or isolated-sign relabeling.',
       'duplicate_methods':['stable dataset/collection/utterance identity','exact raw SHA256','full decoded BGR pixel SHA256 for distinct movies',
                            'same-source annotation-aware nonoverlapping utterance coordinates','selected-frame content hashes for timed slices'],
       'duplicate_limitations':'No exhaustive perceptual re-encoding or ASL-semantic duplicate detector; no alternate views counted. Different source recordings with the same gloss sentence remain distinct real performances.',
       'signer_independent':False,'original9dev_preserved':True,'reserved17test_untouched':True,'production_changed':False}
    write(folder/'quality.json',result);print(json.dumps({k:result[k] for k in ('unique_records','eligible_train_by_label_pool','rejection_counts','single_pool_minimum_shortfall')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');build(p.parse_args().watch)
