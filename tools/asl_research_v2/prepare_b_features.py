"""Private research preprocessing; authentic sequence glosses, never English targets."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from pipeline.perception_vnext.offline_candidate01 import Candidate01Offline, iter_video_frames
from pipeline.perception_vnext.recognition_schema import RawSequence, RAW_SCHEMA, save_raw_sequence
from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features

SCHEMA = "apphear.continuous_asl.candidate01_causal378_native.v1"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2); f.write('\n')

def checked_run(path: Path) -> Path:
    run = path.resolve()
    run.relative_to(ROOT/'experiments'/'continuous_asl')
    if not (run/'preservation_baseline.json').is_file(): raise ValueError('RUN_IDENTITY')
    return run

def make_raw(row: dict, records: list, meta: dict) -> RawSequence:
    hand_fields = {'hand_xy':'xy', 'hand_proposal_support':'proposal_support',
        'hand_joint_valid':'valid', 'hand_presence':'presence', 'hand_observed':'observed',
        'hand_proposal_present':'proposal_present', 'hand_roi_center_side':'roi',
        'hand_handedness_diagnostic':'handedness'}
    arrays = {k: np.asarray([[h[v] for h in rec[3]['hands']] for rec in records])
              for k,v in hand_fields.items()}
    arrays.update({k:np.asarray([rec[3][k] for rec in records]) for k in
        ('source_size','body_xy','body_support','body_valid','body_available')})
    return RawSequence(metadata={**row, **meta, 'schema':RAW_SCHEMA,
        'image_semantics':'UPRIGHT_UNMIRRORED_SOURCE_RGB', 'duration_scope':'whole_continuous_utterance',
        'current_frame_only':True, 'synthetic_depth':False,'carry_forward':False,
        'timestamp_sources':sorted({r[2] for r in records})},
        frame_index=np.asarray([r[0] for r in records], np.int32),
        timestamp_ms=np.asarray([r[1] for r in records], np.float64), **arrays)

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--tag',required=True,choices=('sample','full')); p.add_argument('--ids',nargs='*')
    args=p.parse_args(); run=checked_run(args.run_root)
    os.environ['KERAS_HOME']=str(run/'cache'/'keras')
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import cv2
    cv2.setNumThreads(2)
    out=run/'experiment_B'/'data'/'processed'/args.tag
    if out.exists(): raise ValueError('OUTPUT_EXISTS: select a fresh run rather than overwrite')
    rows=json.loads((run/'ncslgr_eligible_index.json').read_text())
    if args.ids: rows=[r for r in rows if r['sample_id'] in args.ids]
    collections=sorted({r['collection'] for r in json.loads((run/'ncslgr_eligible_index.json').read_text())},
        key=lambda s: hashlib.sha256(('20261019:'+s).encode()).hexdigest())
    val_collections=set(collections[:max(1,round(len(collections)*.2))])
    onnx=ROOT/'android/app/src/main/assets/perception_vnext/dwpose_t_singlepass.onnx'
    refiner=ROOT/'android/app/src/main/assets/perception_vnext/hand_landmarks_detector.tflite'
    config={'schema':SCHEMA,'cadence_max_hz':None,'sampling':'every genuine native-rate decoded frame; no resampling',
        'feature_implementation_sha256':sha(ROOT/'pipeline/perception_vnext/segmentation/learned_features.py'),
        'extractor_sha256':sha(ROOT/'pipeline/perception_vnext/offline_candidate01.py'),
        'preprocessing_script_sha256':sha(Path(__file__)), 'onnx_sha256':sha(onnx),'refiner_sha256':sha(refiner),
        'normalization':'TRAIN mean/std only, fitted later; missingness remains in feature blocks',
        'split':'collection-disjoint diagnostic; signer identity unknown',
        'validation_collections':sorted(val_collections),'seed':20261019}
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True); write_new(out/'config.json',config)
    manifest=[]; timings=[]; failures=[]
    started=time.perf_counter()
    with Candidate01Offline(onnx,refiner) as runtime:
        for row in rows:
            sample=row['sample_id']; video=ROOT/'experiments/continuous_asl/20261019T155428Z_cslr01/data/raw/ncslgr/videos'/f'{sample}.mov'
            try:
                receipt=json.loads(video.with_suffix('.mov.receipt.json').read_text())
                if sha(video)!=receipt['sha256']: raise ValueError('VIDEO_HASH')
                iterator,meta=iter_video_frames(video,None); records=[]; count=0; last=-1e9; decode_start=time.perf_counter()
                for index,ms,source,image in iterator:
                    count+=1
                    if ms < last: raise ValueError("NONMONOTONIC_TIMESTAMP")
                    tick=time.perf_counter(); observation=runtime.process_rgb(image)
                    timings.append((time.perf_counter()-tick)*1000)
                    records.append((index,ms,source,observation)); last=ms
                if len(records)<2: raise ValueError('EMPTY_OR_TOO_SHORT')
                row={**row,'split':'val' if row['collection'] in val_collections else 'train',
                     'original_split':'NOT_PROVIDED_IN_PUBLIC_INDEX', 'source_video_sha256':receipt['sha256'],
                     'preprocessing_config_sha256':config_hash,'license_scope':'PRIVATE_NONCOMMERCIAL_RESEARCH_ONLY'}
                raw=make_raw(row,records,meta); raw.validate()
                features=derive_causal_segmentation_features(raw)
                # Genuine causal-prefix parity; no completed-duration information may leak back.
                for n in sorted({2,max(2,len(records)//2),len(records)}):
                    prefix=RawSequence(metadata=raw.metadata,**{k:getattr(raw,k)[:n] for k in RawSequence.__dataclass_fields__ if k!='metadata'})
                    if not np.array_equal(features[:n],derive_causal_segmentation_features(prefix)): raise ValueError('FEATURE_CAUSALITY')
                repeats=sum(a==b for a,b in zip(row['glosses'],row['glosses'][1:]))
                if len(features)<len(row['glosses'])+repeats: raise ValueError('CTC_LENGTH_INFEASIBLE')
                raw_path=out/f'{sample}.raw.npz'; feature_path=out/f'{sample}.features.npz'
                save_raw_sequence(raw_path,raw)
                metadata={**row,**meta,'schema_version':SCHEMA,'timestamps_ms':raw.timestamp_ms.tolist(),
                    'sequence_length':len(features),'decoded_frames':count,'dimensions':raw.source_size[0].tolist(),
                    'container_duration_seconds':count/meta['native_fps'],
                    'timestamp_sources':raw.metadata['timestamp_sources'],
                    'processed_seconds':time.perf_counter()-decode_start,
                    'hand_observed_fraction':np.mean(raw.hand_observed,axis=0).tolist(),
                    'body_available_fraction':float(np.mean(raw.body_available)),
                    'feature_path':str(feature_path.relative_to(run)), 'feature_sha256':None,
                    'quality_flags':['SIGNER_UNKNOWN','NO_INDIVIDUAL_SIGN_BOUNDARIES','NO_EXPLICIT_NONSIGN_LABELS',
                                     'CONTAINER_TIMING_NOT_INDEPENDENTLY_VERIFIED','DIAGNOSTIC_COLLECTION_SPLIT']}
                np.savez_compressed(feature_path,X=features,timestamp_ms=raw.timestamp_ms,
                    schema_version=np.asarray(SCHEMA),metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
                metadata['feature_sha256']=sha(feature_path); manifest.append(metadata)
                print(json.dumps({'sample_id':sample,'frames':count,'observations':len(features),'split':row['split'],
                    'hands':metadata['hand_observed_fraction']}),flush=True)
            except Exception as exc:
                failures.append({'sample_id':sample,'error':str(exc)})
                print(json.dumps({'sample_id':sample,'error':str(exc)}),flush=True)
                # Preserve already-processed files and stop on the first corrupt/missing input.
                break
    report={'schema':SCHEMA,'status':'PASS' if not failures else 'FAILED_PARTIAL','requested':len(rows),
        'processed':len(manifest),'failures':failures,'preprocessing_config_sha256':config_hash,
        'elapsed_seconds':time.perf_counter()-started,'selected_frame_count':len(timings),
        'extraction_ms_p50':float(np.percentile(timings,50)) if timings else None,
        'extraction_ms_p95':float(np.percentile(timings,95)) if timings else None,
        'extraction_ms_sum':sum(timings),'split_counts':dict(Counter(r['split'] for r in manifest)),
        'video_sha256_duplicates':len(manifest)-len({r['source_video_sha256'] for r in manifest}),
        'collection_overlap':sorted({r['collection'] for r in manifest if r['split']=='train'} &
                                    {r['collection'] for r in manifest if r['split']=='val'}),
        'signer_leakage':'UNVERIFIABLE: IDs unavailable','near_duplicate_method':'exact video SHA256 only; perceptual detection not performed',
        'hardware_scope':'Windows CPU ORT DWPose + CPU TFLite refiner, no Android FPS claim'}
    write_new(out/'manifest.json',manifest); write_new(out/'quality.json',report)
    print(json.dumps(report),flush=True)
    if failures: raise SystemExit(1)
if __name__=='__main__': main()

