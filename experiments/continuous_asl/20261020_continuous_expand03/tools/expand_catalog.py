"""Additive complete-index catalog; frozen comparison data/weights stay unchanged."""
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit, urlunsplit
import numpy as np
from core import RUN, ROOT, OLD, SECOND, SCHEMA, read, write, sha

CAT = RUN/'expanded_catalog_v2'
OUT = RUN/'data/processed/catalog_v2'

def role(collection):
    if collection in {'ncslgr10e.xml','ncslgr10m.xml'}: return 'PRIOR_DEV_QUARANTINE'
    if collection=='ncslgr10p.xml': return 'FROZEN_DEV_REFERENCE'
    if collection=='ncslgr10q.xml': return 'FROZEN_TEST_EXPOSED_LOCKED'
    return 'TRAIN_POOL_NOT_FIT'

def plan():
    if (CAT/'plan.json').exists():
        print('CATALOG_PLAN_EXISTS_NO_REWRITE'); return
    original=read(RUN/'dataset_plan.json'); index=Path(original['index_path'])
    if sha(index)!=original['index_sha256']: raise ValueError('SOURCE_INDEX_CHANGED')
    source=list(csv.DictReader(index.open(encoding='utf-8-sig',newline='')));uses=defaultdict(set)
    for row in source:uses[row['Compressed MOV file']].add((row['XML file'],row['Occurs in utterance id'],row['main gloss']))
    cached={};pixel={r['sample_id']:r['decoded_sha256'] for r in read(OLD/'logs/visual_duplicates.json')['records']}
    for row in read(OLD/'data/processed/full/manifest.json'):
        cached[row['sample_id']]={**row,'feature_path':str(OLD/row['feature_path']),
            'video_path':str(OLD/'data/raw/ncslgr/videos'/f"{row['sample_id']}.mov"),
            'decoded_pixel_sha256':pixel[row['sample_id']],'cache_origin':'PRIOR_37_RAW_RECORDINGS'}
    for partition in ('train_dev','test'):
        decoded={r['sample_id']:r['decoded_pixel_sha256'] for r in read(RUN/'data/processed'/partition/'quality.json')['decoded']}
        for row in read(RUN/'data/processed'/partition/'manifest.json'):
            cached[row['sample_id']]={**row,'decoded_pixel_sha256':decoded[row['sample_id']],
                'cache_origin':'PRIOR_37_RAW_RECORDINGS' if row['reuse'] else 'COMPARISON_V1_ACQUIRED'}
    lex=re.compile(r'(?:\(2h\))?[A-Za-z][A-Za-z0-9_:+().-]*\+*$');rows=[];seen=set();audit=[]
    for number,r in enumerate(source,1):
        url=r['Compressed MOV file'];u=urlsplit(url);words=r['main gloss'].split();key=(r['XML file'],r['Occurs in utterance id'])
        reason='ADMITTED'
        if r['Perspective/Camera id']!='0':reason='OTHER_CAMERA'
        elif len(uses[url])!=1:reason='SHARED_VIDEO_NEEDS_AUTHENTIC_UTTERANCE_TIMING'
        elif u.hostname!='csr.bu.edu' or not u.path.endswith('.mov'):reason='NOT_VERIFIED_PUBLIC_SMALL_MOV'
        elif len(words)<2:reason='NOT_MULTI_TOKEN_CONTINUOUS_TARGET'
        elif not all(lex.fullmatch(t) and not any(s in t.lower() for s in ('cl:','fs-','fingerspell')) for t in words):reason='COMPLEX_GLOSS_TOKEN_BOUNDARIES_NEED_STRUCTURED_ANNOTATIONS'
        elif key in seen:reason='DUPLICATE_INDEX_IDENTITY'
        audit.append({'source_row':number,'collection':key[0],'utterance_id':key[1],'camera':r['Perspective/Camera id'],
                      'source_url':url,'source_gloss':r['main gloss'],'decision':reason})
        if reason!='ADMITTED':continue
        seen.add(key);sample='ncslgr_'+hashlib.sha256('|'.join(key).encode()).hexdigest()[:16]
        row={'sample_id':sample,'dataset':'NCSLGR','collection':key[0],'utterance_id':key[1],'source_group':r['Video sequence id'],
             'signer_id':None,'original_split':'NOT_PROVIDED_IN_PUBLIC_INDEX','split':role(key[0]),
             'source_url_original':url,'source_url':urlunsplit(('https',u.netloc,u.path,u.query,'')),
             'gloss_text':r['main gloss'],'glosses':words,'annotation_type':'AUTHENTIC_INDEX_MAIN_GLOSS_TOKEN_SEQUENCE',
             'annotation_provenance':{'official_index':str(index),'index_sha256':original['index_sha256'],'source_row':number},
             'individual_sign_boundaries_available':False,'sign_times':None,'cadence_max_hz':10,
             'license_status':'PRIVATE_RESEARCH_TERMS_REVIEWED_NO_REDISTRIBUTION',
             'license_reference':'https://www.bu.edu/asllrp/dai-terms.html','new_in_catalog_v2':sample not in cached}
        if sample in cached:
            row.update(cached_record=cached[sample],video_path=cached[sample]['video_path'])
        else:row['video_path']=str(RUN/'data/raw/ncslgr/catalog_v2/videos'/f'{sample}.mov')
        rows.append(row)
    if len(rows)!=223:raise ValueError('FULL_INDEX_CANDIDATE_COUNT_CHANGED')
    how=SECOND/'data/raw/how2sign/val_rgb_front_clips.zip'
    old_bytes=sum(p.stat().st_size for p in (OLD/'data/raw/ncslgr/videos').glob('*.mov'))
    current_bytes=sum(p.stat().st_size for p in (RUN/'data/raw/ncslgr').rglob('*.mov'))
    write(CAT/'plan.json',{'schema':'apphear.continuous_expansion.catalog.v2','rows':rows,'index_rows':len(source),
        'index_sha256':original['index_sha256'],'eligible_candidates':len(rows),'new_raw_required':sum(r['new_in_catalog_v2'] for r in rows),
        'role_counts':dict(Counter(r['split'] for r in rows)),
        'exclusion_counts':dict(Counter(r['decision'] for r in audit if r['decision']!='ADMITTED')),
        'how2sign_bytes_reused_stat_only':how.stat().st_size,'prior_ncslgr_raw_bytes':old_bytes,
        'current_run_raw_bytes_before':current_bytes,'per_file_download_cap':8388608,
        'new_run_raw_total_cap':1073741824,'conservative_global_raw_cap_with_How2Sign':2147483648,
        'conservative_remaining_raw_budget':2147483648-how.stat().st_size-old_bytes-current_bytes,
        'frozen_v1_plan_sha256':sha(RUN/'dataset_plan.json'),'frozen_v1_selection_sha256':sha(RUN/'model_selection_freeze.json'),
        'future_training':'Catalog only. No v2 fitting or new selection on already-exposed TEST.',
        'signer_independent':False})
    with (CAT/'index_inventory.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    print(json.dumps({'eligible':len(rows),'new_raw_required':sum(r['new_in_catalog_v2'] for r in rows),
                      'role_counts':dict(Counter(r['split'] for r in rows)),'remaining_budget':2147483648-how.stat().st_size-old_bytes-current_bytes}))

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
                video=Path(row['video_path']);receipt=read(str(video)+'.receipt.json')
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
        'v2_fitting_performed':False,'frozen_v1_training_counts':{'train':100,'dev':9}}
    write(CAT/'quality.json',report)
    with (CAT/'manifest.csv').open('x',encoding='utf-8',newline='') as f:
        fields=('sample_id','collection','utterance_id','source_group','signer_id','split','source_url','annotation_type','glosses',
                'raw_sha256','bytes','feature_path','feature_sha256','observations','new_in_catalog_v2','unique_valid','exclusion_reason')
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in results:w.writerow({k:json.dumps(row[k]) if k=='glosses' else row.get(k) for k in fields})
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('plan','prepare'));a=p.parse_args()
    if a.phase=='plan':plan()
    else:prepare()
