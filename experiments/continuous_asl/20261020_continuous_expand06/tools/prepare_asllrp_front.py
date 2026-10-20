from dataclasses import replace
import hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
from core import RUN,ROOT,SCHEMA,read,write,sha
from prepare_asllrp import memory_available,check_media_contract

def main(shard=0,shards=1,pilot=False):
    import cv2
    from prepare_features import make_raw
    from pipeline.perception_vnext.offline_candidate01 import Candidate01Offline,iter_video_frames
    from pipeline.perception_vnext.recognition_schema import save_raw_sequence
    from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features
    if not 0<=shard<shards<=4:raise ValueError('SHARD_CONFIGURATION')
    rows=read(RUN/'asllrp2025_catalog/media_manifest.json')
    if pilot:
        # Three source collections, chosen deterministically before feature work.
        selected={}
        for r in rows:
            source=r['collection'].split('_')[0]
            if source not in selected:selected[source]=r
            if len(selected)==3:break
        rows=list(selected.values())
    else:
        gate=read(RUN/'asllrp2025_catalog/feature_pilot.json')
        if gate['status']!='PASS_MEDIA_FRAME_FEATURE_JOIN':raise ValueError('PILOT_GATE')
        rows=[r for i,r in enumerate(rows) if i<1300 and i%shards==shard]
    os.environ['KERAS_HOME']=str(RUN/'cache/keras');os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    cv2.setNumThreads(2)
    output=RUN/'data/processed/asllrp2025';output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();accepted=[];failures=[];resource_stop=None
    with Candidate01Offline(ROOT/'android/app/src/main/assets/perception_vnext/dwpose_t_singlepass.onnx',
                            ROOT/'android/app/src/main/assets/perception_vnext/hand_landmarks_detector.tflite') as runtime:
        for index,row in enumerate(rows):
            sample=row['sample_id'];done=output/(sample+'.row.json');failed=output/(sample+'.failure.json')
            if done.exists():
                prior=read(done)
                if prior['raw_sha256']!=row['raw_sha256'] or sha(prior['feature_path'])!=prior['feature_sha256']:raise ValueError('RESUME_HASH')
                accepted.append(sample);continue
            if failed.exists():failures.append(read(failed));continue
            if memory_available()<1073741824 or shutil.disk_usage(RUN).free<10737418240:
                resource_stop='RAM_OR_DISK_HEADROOM_RESUME_AFTER_RESOURCE_RELEASE';break
            try:
                video=Path(row['video_path']);tick=time.perf_counter()
                if sha(video)!=row['raw_sha256']:raise ValueError('MEDIA_HASH')
                capture=cv2.VideoCapture(str(video))
                fps=capture.get(cv2.CAP_PROP_FPS);nf=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release()
                check_media_contract(row,fps,nf)
                iterator,meta=iter_video_frames(video,None);records=[];last=-1e9;count=0;pixels=hashlib.sha256();cost=[];max_pts_error=0.
                for i,ms,source,rgb in iterator:
                    count+=1
                    if source!='CONTAINER_POS_MSEC':raise ValueError('REPAIRED_OR_FALLBACK_TIMESTAMPS_REJECTED')
                    max_pts_error=max(max_pts_error,abs(ms-i*1000/fps))
                    if max_pts_error>500/fps:raise ValueError('FRAME_INDEX_CONTAINER_PTS_DISAGREE')
                    pixels.update(rgb[...,::-1].tobytes())
                    if ms<last+100-1e-6:continue
                    begin=time.perf_counter();obs=runtime.process_rgb(rgb);cost.append((time.perf_counter()-begin)*1000)
                    records.append((i,ms,source,obs));last=ms
                if count!=nf:raise ValueError('INCOMPLETE_FULL_DECODE')
                if len(records)<2:raise ValueError('TOO_FEW_OBSERVATIONS')
                raw=make_raw(row,records,meta);raw.validate()
                if not raw.body_available.any() and not raw.hand_observed.any():raise ValueError('NO_BODY_OR_HAND_OBSERVATIONS')
                x=derive_causal_segmentation_features(raw)
                if x.dtype!=np.float32 or x.shape!=(len(records),378) or not np.isfinite(x).all():raise ValueError('FEATURE_CONTRACT')
                if not np.all(np.diff(raw.timestamp_ms)>0):raise ValueError('NONINCREASING_TIMESTAMPS')
                if len(x)<len(row['glosses'])+sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:])):raise ValueError('CTC_FEASIBILITY')
                mid=max(1,len(x)//2)
                prefix=replace(raw,**{k:getattr(raw,k)[:mid] for k in raw.__dataclass_fields__ if k!='metadata'})
                if not np.array_equal(x[:mid],derive_causal_segmentation_features(prefix)):raise ValueError('CAUSAL_PREFIX_PARITY')
                path=output/(sample+'.features.npz')
                if path.exists():raise ValueError('INCOMPLETE_OUTPUT_RETAINED_FOR_REVIEW')
                with path.open('xb') as f:np.savez_compressed(f,X=x,timestamp_ms=raw.timestamp_ms,schema_version=np.asarray(SCHEMA),metadata_json=np.asarray(json.dumps(row)))
                save_raw_sequence(output/(sample+'.raw.npz'),raw)
                result={**row,'feature_path':str(path),'feature_sha256':sha(path),'decoded_pixel_sha256':pixels.hexdigest(),
                        'observations':len(x),'decoded_frames':count,'native_fps':fps,'dimensions':raw.source_size[0].tolist(),
                        'timestamp_sources':raw.metadata['timestamp_sources'],'max_pts_frame_error_ms':max_pts_error,
                        'individual_sign_times_ms':[[1000*(e['start_frame_master']-row['utterance_start_frame_master'])/fps,
                                                   1000*(e['end_frame_master']-row['utterance_start_frame_master']+1)/fps] for e in row['events']],
                        'preprocessing_status':'PASS','unique_valid':True,'cadence_max_hz':10,
                        'extraction_ms_sum':sum(cost),'extraction_ms_p50':float(np.percentile(cost,50)),
                        'extraction_ms_p95':float(np.percentile(cost,95)),'preprocessing_seconds':time.perf_counter()-tick,
                        'hand_observed_fraction':raw.hand_observed.mean(0).tolist(),'body_available_fraction':float(raw.body_available.mean()),
                        'timing_status':'EXACT_INCLUSIVE_MASTER_FRAME_SPAN_AND_29.97FPS_MEDIA_COUNT_JOIN; NO_OFFSET_SCALING; VISUAL_ALIGNMENT_NOT_INDEPENDENTLY_AUDITED'}
                write(done,result);accepted.append(sample)
                if pilot or len(accepted)%25==0:print(json.dumps({'shard':shard,'valid':len(accepted),'failed':len(failures),'last_sample':sample}),flush=True)
            except Exception as e:
                failure={'sample_id':sample,'reason':str(e)};write(failed,failure);failures.append(failure)
                if pilot:print(json.dumps(failure),flush=True)
        report={'status':'PASS_MEDIA_FRAME_FEATURE_JOIN' if not failures and not resource_stop else 'PARTIAL_REJECTED_OR_RESOURCE_STOP',
                'requested':len(rows),'valid':len(accepted),'accepted':accepted,'failures':failures,'resource_stop':resource_stop,
                'shard':shard,'shards':shards,'seconds':time.perf_counter()-start,'script_sha256':sha(__file__),
                'synthetic':False,'training_performed':False}
    destination=RUN/'asllrp2025_catalog'/('feature_pilot.json' if pilot else f'front1300_{shard}_of_{shards}_{time.time_ns()}.json')
    write(destination,report);print(json.dumps({k:v for k,v in report.items() if k not in ('accepted','failures')}),flush=True)
    if pilot and report['status']!='PASS_MEDIA_FRAME_FEATURE_JOIN':raise SystemExit(1)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();main(a.shard,2,False)
