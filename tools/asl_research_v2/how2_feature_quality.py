"""Pose quality only on the How2Sign pilot; English is never a gloss target."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from common import ROOT,run_root,write,sha
from prepare_features import make_raw
from pipeline.perception_vnext.offline_candidate01 import Candidate01Offline,iter_video_frames
from pipeline.perception_vnext.recognition_schema import save_raw_sequence
from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
os.environ['KERAS_HOME']=str(run/'cache/keras');os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
rows=json.loads((run/'how2sign_inventory.json').read_text())['usable_pilot'];out=run/'data/derived/how2sign_pose_probe';out.mkdir(parents=True)
reports=[];timings=[];start=time.perf_counter()
with Candidate01Offline(ROOT/'android/app/src/main/assets/perception_vnext/dwpose_t_singlepass.onnx',ROOT/'android/app/src/main/assets/perception_vnext/hand_landmarks_detector.tflite') as runtime:
    for row in rows:
        path=Path(row['path'])
        if sha(path)!=row['sha256']:raise ValueError('HOW2_PILOT_HASH')
        iterator,meta=iter_video_frames(path,None);records=[];previous=-1e9
        for i,ms,source,rgb in iterator:
            if ms<previous+100-1e-6:continue
            tick=time.perf_counter();obs=runtime.process_rgb(rgb);timings.append((time.perf_counter()-tick)*1000);records.append((i,ms,source,obs));previous=ms
        raw=make_raw({**row,'annotation_type':'ENGLISH_TRANSLATION','source_video_sha256':row['sha256'],'gloss_targets':None},records,meta)
        raw.validate();features=derive_causal_segmentation_features(raw);save_raw_sequence(out/(row['sample_id']+'.raw.npz'),raw)
        np.savez_compressed(out/(row['sample_id']+'.features.npz'),X=features,timestamp_ms=raw.timestamp_ms,
            schema_version=np.asarray('apphear.continuous_asl.how2sign_pose_probe.v1'),label_type=np.asarray('ENGLISH_TRANSLATION_NOT_GLOSS'))
        reports.append({'sample_id':row['sample_id'],'observations':len(records),'hand_observed':raw.hand_observed.mean(0).tolist(),
            'hand_proposal_present':raw.hand_proposal_present.mean(0).tolist(),'joint_valid_fraction':raw.hand_joint_valid.mean(axis=(0,2)).tolist(),
            'body_available':float(raw.body_available.mean()),'mean_delta_time_ms':float(np.diff(raw.timestamp_ms).mean()),'gloss_training_eligible':False})
write(run/'how2sign_pose_quality.json',{'status':'PASS_POSE_PROBE_ONLY','clips':reports,'observations':len(timings),'elapsed_seconds':time.perf_counter()-start,
    'extraction_ms_p50':float(np.percentile(timings,50)),'extraction_ms_p95':float(np.percentile(timings,95)),
    'model_evidence_not_manual_occlusion_ground_truth':True,'training_executed_on_How2Sign':False,'original_split':'official_val'})
print(json.dumps({'clips':len(reports),'observations':len(timings),'seconds':time.perf_counter()-start}))
