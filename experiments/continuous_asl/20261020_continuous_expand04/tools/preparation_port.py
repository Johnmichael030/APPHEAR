from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from core import RUN, ROOT, OLD, SCHEMA, read, write, sha
CAT=RUN/'full_index_v2/preparation'
OUT=RUN/'data/processed/index_v2'

def prepare():
    if (CAT/'manifest.json').exists():print('CATALOG_COMPLETE_NO_REPROCESSING');return
    import cv2
    from prepare_features import make_raw
    from pipeline.perception_vnext.offline_candidate01 import Candidate01Offline,iter_video_frames
    from pipeline.perception_vnext.recognition_schema import save_raw_sequence
    from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features
    os.environ['KERAS_HOME']=str(RUN/'cache/keras');os.environ['TF_CPP_MIN_LOG_LEVEL']='2';cv2.setNumThreads(2)
    plan=read(CAT/'plan.json');OUT.mkdir(parents=True,exist_ok=True)
    results=[];failures=[];timings=[];start=time.perf_counter()
    with Candidate01Offline(ROOT/'android/app/src/main/assets/perception_vnext/dwpose_t_singlepass.onnx',
                            ROOT/'android/app/src/main/assets/perception_vnext/hand_landmarks_detector.tflite') as runtime:
        for row in plan['rows']:
            sample=row['sample_id'];completed=OUT/f'{sample}.row.json'
            if completed.exists():
                previous=read(completed)
                if sha(previous['feature_path'])!=previous['feature_sha256']:raise ValueError('RESUME_FEATURE_HASH')
                results.append(previous);continue
            try:
                video=Path(row['video_path'])
                pending=time.monotonic()
                while not video.is_file() or not Path(str(video)+'.receipt.json').is_file():
                    failure=RUN/'logs'/('download_indexv2_'+sample+'_attempt1.json')
                    if failure.exists():raise ValueError('DOWNLOAD_FAILED_RETAINED_EVIDENCE:'+str(failure))
                    if time.monotonic()-pending>120:raise ValueError('DOWNLOAD_NOT_READY_WITHIN120SECONDS')
                    time.sleep(0.5)
                receipt=read(str(video)+'.receipt.json')
                if receipt['source_url']!=row['source_url'] or video.stat().st_size!=receipt['bytes']:raise ValueError('SOURCE_RECEIPT_JOIN')
                if row['new_in_catalog_v2']:
                    if sha(video)!=receipt['sha256']:raise ValueError('VIDEO_HASH')
                    frames,meta=iter_video_frames(video,None);records=[];last=-1e9;pixels=hashlib.sha256();count=0
                    for i,ms,source,rgb in frames:
                        count+=1
                        if count>10000:raise ValueError('DECODE_FRAME_BUDGET')
                        pixels.update(rgb[...,::-1].tobytes())
                        if ms<last+100-1e-6:continue
                        tick=time.perf_counter();obs=runtime.process_rgb(rgb);timings.append((time.perf_counter()-tick)*1000)
                        records.append((i,ms,source,obs));last=ms
                    if len(records)<2:raise ValueError('EMPTY_OR_TOO_SHORT')
                    raw=make_raw({**row,'source_video_sha256':receipt['sha256']},records,meta);raw.validate()
                    x=derive_causal_segmentation_features(raw);mid=max(1,len(x)//2)
                    prefix=replace(raw,**{k:getattr(raw,k)[:mid] for k in raw.__dataclass_fields__ if k!='metadata'})
                    if not np.allclose(x[:mid],derive_causal_segmentation_features(prefix),atol=1e-6,rtol=0):raise ValueError('NONCAUSAL_FEATURES')
                    ts=raw.timestamp_ms;feature_path=OUT/f'{sample}.features.npz'
                    if feature_path.exists():raise ValueError('INCOMPLETE_FEATURE_OUTPUT_RETAINED_NEEDS_REVIEW')
                    np.savez_compressed(feature_path,X=x,timestamp_ms=ts,schema_version=np.asarray(SCHEMA),
                                        metadata_json=np.asarray(json.dumps(row)))
                    save_raw_sequence(OUT/f'{sample}.raw.npz',raw)
                    decoded_sha=pixels.hexdigest();dimensions=np.asarray(raw.source_size[0]).tolist();fps=meta['native_fps']
                else:
                    cached=row['cached_record'];feature_path=Path(cached['feature_path'])
                    if cached['source_video_sha256']!=receipt['sha256'] or sha(feature_path)!=cached['feature_sha256']:raise ValueError('CACHED_SOURCE_FEATURE_JOIN')
                    with np.load(feature_path,allow_pickle=False) as f:
                        if str(f['schema_version'].item())!=SCHEMA:raise ValueError('CACHED_SCHEMA')
                        x=f['X'];ts=f['timestamp_ms']
                    count=cached['decoded_frames'];decoded_sha=cached['decoded_pixel_sha256'];dimensions=cached['dimensions'];fps=cached['native_fps']
                repeated=sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:]))
                if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=378 or not np.isfinite(x).all():raise ValueError('FEATURE_VALUES')
                if len(ts)!=len(x) or not np.isfinite(ts).all() or not np.all(np.diff(ts)>0):raise ValueError('TIMESTAMPS')
                if len(x)<len(row['glosses'])+repeated:raise ValueError('CTC_FEASIBILITY')
                result={k:v for k,v in row.items() if k!='cached_record'}
                result.update(preprocessing_status='PASS',feature_path=str(feature_path),feature_sha256=sha(feature_path),
                    raw_sha256=receipt['sha256'],bytes=receipt['bytes'],decoded_frames=count,observations=len(x),
                    decoded_pixel_sha256=decoded_sha,dimensions=dimensions,native_fps=fps,
                    preprocessing_evidence='NEW_FULL_DECODE_AND_FEATURES' if row['new_in_catalog_v2'] else 'REUSED_VERIFIED_SOURCE_FEATURE_DECODE_EVIDENCE',
                    cache_origin=None if row['new_in_catalog_v2'] else row['cached_record']['cache_origin'])
                write(completed,result);results.append(result)
                if row['new_in_catalog_v2']:print(json.dumps({'new_sample':sample,'role':row['split'],'observations':len(x)}),flush=True)
            except Exception as e:
                failures.append({'sample_id':sample,'reason':str(e),'new_in_catalog_v2':row['new_in_catalog_v2']})
                print(json.dumps({'sample_id':sample,'failure':str(e)}),flush=True)
    # Every acquired candidate stays in the catalog; duplicate samples get no unique-data credit.
    seen_raw={};seen_pixels={};duplicates=[]
    for row in sorted(results,key=lambda r:r['new_in_catalog_v2']):
        original=seen_raw.get(row['raw_sha256']) or seen_pixels.get(row['decoded_pixel_sha256'])
        row['unique_valid']=original is None
        if original:
            row['exclusion_reason']='DUPLICATE_OF_'+original;duplicates.append({'sample_id':row['sample_id'],'duplicate_of':original})
        else:
            seen_raw[row['raw_sha256']]=row['sample_id'];seen_pixels[row['decoded_pixel_sha256']]=row['sample_id'];row['exclusion_reason']=None
    valid=[r for r in results if r['unique_valid']]
    write(CAT/'manifest.json',results)
    report={'status':'PASS' if len(results)==len(plan['rows']) and not failures else 'PARTIAL',
        'requested':len(plan['rows']),'processed':len(results),'unique_valid':len(valid),'failures':failures,'duplicates':duplicates,
        'new_v2_raw_downloaded_and_valid':sum(r['new_in_catalog_v2'] and r['unique_valid'] for r in results),
        'new_v2_raw_bytes':sum(r['bytes'] for r in results if r['new_in_catalog_v2']),
        'reused_records':sum(not r['new_in_catalog_v2'] for r in results),'role_counts_unique':dict(Counter(r['split'] for r in valid)),
        'elapsed_seconds':time.perf_counter()-start,'new_observations_processed':len(timings),
        'extraction_ms_p50':float(np.percentile(timings,50)) if timings else None,
        'extraction_ms_p95':float(np.percentile(timings,95)) if timings else None,
        'signer_independence':False,'individual_sign_timing':'UNAVAILABLE_IN_INDEX_NO_INVENTED_BOUNDARIES',
        'time_semantics':'Preserved genuine source/container timestamps; no claim of independently verified real-world sign timing.',
        'v2_fitting_performed':False,'previous_frozen_training_counts':{'train':181,'dev':9}}
    write(CAT/'quality.json',report)
    with (CAT/'manifest.csv').open('x',encoding='utf-8',newline='') as f:
        fields=('sample_id','collection','utterance_id','source_group','signer_id','split','source_url','annotation_type','glosses',
                'raw_sha256','bytes','feature_path','feature_sha256','observations','new_in_catalog_v2','unique_valid','exclusion_reason')
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in results:w.writerow({k:json.dumps(row[k]) if k=='glosses' else row.get(k) for k in fields})
    print(json.dumps(report),flush=True)

if __name__=='__main__':prepare()
