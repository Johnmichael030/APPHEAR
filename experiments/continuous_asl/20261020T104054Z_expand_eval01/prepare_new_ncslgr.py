"""Decode each new source once; slice only authentic publisher millisecond spans.

No time scaling, interpolation, timestamp repair, target editing or test access.
Frame/container fallback used by the existing decoder is explicitly rejected here.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from core import RUN, ROOT, FOUR, FIVE, SCHEMA, read, write, sha


def checked_slice(raw, row):
    a, b = row['start_ms'], row['end_ms']
    if a < 0 or b <= a:
        raise ValueError('INVALID_SOURCE_SPAN')
    if any(not (a <= s <= e <= b) for s, e in row['sign_times_ms']):
        raise ValueError('GLOSS_OUTSIDE_UTTERANCE')
    ts = raw.timestamp_ms
    mask = (ts >= a) & (ts < b)
    if mask.sum() < 2:
        raise ValueError('TOO_FEW_GENUINE_OBSERVATIONS')
    segment = replace(raw, metadata={**raw.metadata, **row, 'duration_scope':'publisher_utterance_span',
                      'timestamp_semantics':'original absolute media milliseconds, no scaling'},
                      **{k:getattr(raw,k)[mask] for k in raw.__dataclass_fields__ if k!='metadata'})
    segment.validate()
    return segment, mask


def main(watch=False):
    import cv2
    from prepare_features import make_raw
    from pipeline.perception_vnext.offline_candidate01 import Candidate01Offline, iter_video_frames
    from pipeline.perception_vnext.recognition_schema import save_raw_sequence, load_raw_sequence
    from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features
    os.environ['KERAS_HOME'] = str(RUN/'cache/keras')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    cv2.setNumThreads(2)
    plan = read(RUN/'depaul_catalog/plan.json')
    catalog = read(FOUR/'full_index_v2/preparation/manifest.json')
    protected = [r for r in catalog if r['split'] not in ('TRAIN_POOL_NOT_FIT','TRAIN_POOL_NEW_FIT')]
    protected_groups = {r['source_group'] for r in protected}
    protected_hashes = {r['raw_sha256'] for r in protected}
    previous_pixels = {r['decoded_pixel_sha256'] for r in catalog}
    groups = defaultdict(list)
    for row in plan['rows']:
        if row['role']=='NEW_RESEARCH_TRAIN_CANDIDATE' and ';' not in row['source_url']:
            groups[row['source_url']].append(row)
    output = RUN/'data/processed/timed'
    output.mkdir(parents=True, exist_ok=True)
    jobs = RUN/'data/processed/source_cache'
    jobs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    attempts = set()
    with Candidate01Offline(ROOT/'android/app/src/main/assets/perception_vnext/dwpose_t_singlepass.onnx',
                            ROOT/'android/app/src/main/assets/perception_vnext/hand_landmarks_detector.tflite') as runtime:
        while True:
            progressed = False
            for url, rows in sorted(groups.items(), key=lambda kv:-len(kv[1])):
                key = hashlib.sha256(url.encode()).hexdigest()[:20]
                done = jobs/(key+'.completed.json')
                failure = jobs/(key+'.failure.json')
                if done.exists() or failure.exists() or key in attempts:
                    continue
                video = RUN/'data/raw/ncslgr_media'/(key+'.mov')
                receipt_path = Path(str(video)+'.receipt.json')
                if not video.is_file() or not receipt_path.is_file():
                    continue
                attempts.add(key); progressed = True
                tick = time.perf_counter()
                try:
                    receipt = read(receipt_path)
                    if receipt['source_url'] != url or sha(video)!=receipt['sha256'] or video.stat().st_size!=receipt['bytes']:
                        raise ValueError('SOURCE_MEDIA_RECEIPT_HASH')
                    if receipt['sha256'] in protected_hashes or any(r['source_group'] in protected_groups for r in rows):
                        raise ValueError('PROTECTED_SOURCE_GROUP_OR_RAW_MEDIA')
                    capture = cv2.VideoCapture(str(video))
                    fps = capture.get(cv2.CAP_PROP_FPS); expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    capture.release()
                    if fps<=0 or expected_frames<2 or expected_frames>100000:
                        raise ValueError('MEDIA_LENGTH_BUDGET')
                    duration = expected_frames*1000/fps
                    if max(r['end_ms'] for r in rows)>duration+1000/fps:
                        raise ValueError('ANNOTATION_BEYOND_MEDIA_DURATION_NO_TIME_SCALING')
                    raw_path = jobs/(key+'.raw.npz')
                    cache_meta = jobs/(key+'.raw.json')
                    if cache_meta.exists():
                        meta = read(cache_meta)
                        if meta['video_sha256']!=receipt['sha256'] or sha(raw_path)!=meta['raw_sha256']:
                            raise ValueError('RAW_CACHE_HASH')
                        raw = load_raw_sequence(raw_path)
                    else:
                        records=[]; last=-1e9; pixel_hash=hashlib.sha256(); count=0; timings=[]; observation_hashes=[]
                        iterator, decoder_meta = iter_video_frames(video,None)
                        for i, ms, source, rgb in iterator:
                            count += 1
                            if source!='CONTAINER_POS_MSEC':
                                raise ValueError('NONAUTHENTIC_OR_REPAIRED_CONTAINER_TIMESTAMPS:'+source)
                            pixel_hash.update(rgb[...,::-1].tobytes())
                            if ms<last+100-1e-6:
                                continue
                            begin=time.perf_counter(); obs=runtime.process_rgb(rgb);timings.append((time.perf_counter()-begin)*1000)
                            records.append((i,ms,source,obs)); observation_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest()); last=ms
                        if count!=expected_frames:
                            raise ValueError('INCOMPLETE_FULL_DECODE')
                        if pixel_hash.hexdigest() in previous_pixels:
                            raise ValueError('DUPLICATE_PREVIOUS_DECODED_MEDIA')
                        raw=make_raw({'source_url':url,'source_video_sha256':receipt['sha256']}, records, decoder_meta)
                        raw.validate(); save_raw_sequence(raw_path,raw)
                        meta={'video_sha256':receipt['sha256'],'raw_sha256':sha(raw_path),'decoded_frames':count,
                              'decoded_pixel_sha256':pixel_hash.hexdigest(),'duration_ms':duration,'native_fps':fps,
                              'observation_pixel_hashes':observation_hashes,'observations':len(records),
                              'extraction_ms_sum':sum(timings),'extraction_p50_ms':float(np.percentile(timings,50)),
                              'extraction_p95_ms':float(np.percentile(timings,95)), 'seconds':time.perf_counter()-tick}
                        write(cache_meta,meta)
                    accepted=[]; rejected=[]
                    sorted_rows=sorted(rows,key=lambda r:(r['start_ms'],r['end_ms']))
                    for left,right in zip(sorted_rows,sorted_rows[1:]):
                        if left['end_ms']>right['start_ms']:
                            raise ValueError('OVERLAPPING_PUBLISHER_UTTERANCE_SPANS')
                    for row in rows:
                        row_path=output/(row['sample_id']+'.row.json')
                        if row_path.exists():
                            old=read(row_path)
                            if sha(old['feature_path'])!=old['feature_sha256']:raise ValueError('COMPLETED_FEATURE_HASH')
                            accepted.append(row['sample_id']);continue
                        try:
                            segment,mask=checked_slice(raw,row)
                            x=derive_causal_segmentation_features(segment)
                            repeats=sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:]))
                            if len(x)<len(row['glosses'])+repeats:raise ValueError('CTC_LENGTH_INFEASIBLE')
                            if x.dtype!=np.float32 or x.shape!=(len(segment.timestamp_ms),378) or not np.isfinite(x).all():raise ValueError('FEATURE_CONTRACT')
                            mid=max(1,len(x)//2)
                            prefix=replace(segment,**{k:getattr(segment,k)[:mid] for k in segment.__dataclass_fields__ if k!='metadata'})
                            if not np.array_equal(x[:mid],derive_causal_segmentation_features(prefix)):raise ValueError('CAUSAL_PREFIX_PARITY')
                            path=output/(row['sample_id']+'.features.npz')
                            if path.exists():raise ValueError('INCOMPLETE_OUTPUT_REQUIRES_REVIEW')
                            with path.open('xb') as f:
                                np.savez_compressed(f,X=x,timestamp_ms=segment.timestamp_ms,schema_version=np.asarray(SCHEMA),metadata_json=np.asarray(json.dumps(row)))
                            selected=np.asarray(meta['observation_pixel_hashes'])[mask]
                            segment_hash=hashlib.sha256(''.join(selected).encode()).hexdigest()
                            result={**row,'split':'train','video_path':str(video),'raw_sha256':receipt['sha256'],
                                    'feature_path':str(path),'feature_sha256':sha(path),'observations':len(x),
                                    'decoded_frames':meta['decoded_frames'],'source_decoded_pixel_sha256':meta['decoded_pixel_sha256'],
                                    'decoded_segment_observation_sha256':segment_hash,'source_cache':str(cache_meta),
                                    'preprocessing_status':'PASS','unique_valid':True,'cadence_max_hz':10,
                                    'timestamp_sources':['CONTAINER_POS_MSEC'],'alignment_scope':'Publisher EAF millisecond origin matches header; all spans fit fully decoded media; no speed scaling. Independent visual alignment not audited.'}
                            write(row_path,result);accepted.append(row['sample_id'])
                        except Exception as error:
                            rejected.append({'sample_id':row['sample_id'],'reason':str(error)})
                    write(done,{'source_url':url,'accepted':accepted,'rejected':rejected,'decoded_frames':meta['decoded_frames'],
                                'source_cache':str(cache_meta),'seconds':time.perf_counter()-tick})
                    print(json.dumps({'source':key,'accepted':len(accepted),'rejected':len(rejected),'seconds':time.perf_counter()-tick}),flush=True)
                except Exception as error:
                    write(failure,{'source_url':url,'reason':str(error),'candidate_utterances':len(rows)})
                    print(json.dumps({'source':key,'failure':str(error)}),flush=True)
            if not watch or (RUN/'acquisition_batch_complete.json').exists() or time.perf_counter()-started>2400:
                break
            if not progressed:time.sleep(2)
    results=[read(p) for p in sorted(output.glob('*.row.json'))]
    if len({r['sample_id'] for r in results})!=len(results) or len({r['decoded_segment_observation_sha256'] for r in results})!=len(results):
        raise ValueError('DUPLICATE_UTTERANCE_OR_OBSERVATION_CONTENT')
    folder=RUN/'timed_preparation'
    if (folder/'manifest.json').exists():raise ValueError('PREPARATION_ALREADY_FINALIZED')
    write(folder/'manifest.json',results)
    write(folder/'quality.json',{'status':'PASS_VALID_SUBSET','unique_valid':len(results),'raw_source_files':len({r['raw_sha256'] for r in results}),
          'eligible_train':len(results),'all_features_schema':SCHEMA,'pending_or_failed_source_files':len(groups)-len(list(jobs.glob('*.completed.json'))),
          'source_failures':[read(p) for p in jobs.glob('*.failure.json')],'signer_independent':False,'synthetic_samples':0,
          'script_sha256':sha(__file__),'seconds':time.perf_counter()-started,'no_test_access':True})
    print(json.dumps({'new_valid_utterances':len(results),'new_source_files':len({r['raw_sha256'] for r in results})}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');main(p.parse_args().watch)
