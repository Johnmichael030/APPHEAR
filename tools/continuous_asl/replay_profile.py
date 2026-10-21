"""One real utterance, CPU RGB-to-gloss replay, including actual process memory."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time
import numpy as np
from prepare_features import ROOT,checked_run,sha,make_raw
from pipeline.perception_vnext.offline_candidate01 import Candidate01Offline,iter_video_frames
from pipeline.perception_vnext.segmentation.learned_features import derive_causal_segmentation_features

def memory() -> dict:
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    process=ctypes.WinDLL('kernel32',use_last_error=True); process.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(process.GetCurrentProcess(),ctypes.byref(counters),counters.cb): raise ctypes.WinError(ctypes.get_last_error())
    return {'actual_pid':os.getpid(),'working_set_bytes':counters.WorkingSetSize,
        'peak_working_set_bytes':counters.PeakWorkingSetSize,'private_commit_bytes':counters.PrivateUsage,
        'peak_pagefile_bytes':counters.PeakPagefileUsage}

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=checked_run(args.run_root)
    os.environ['KERAS_HOME']=str(run/'cache/keras');os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import cv2
    import onnxruntime as ort
    cv2.setNumThreads(2)
    row=next(r for r in json.loads((run/'data/processed/full/manifest.json').read_text()) if r['sample_id']=='ncslgr_189fd7d52377b3a8')
    model=run/'export/causal_gloss_ctc_fp32.onnx';contract=json.loads((run/'export/contract.json').read_text())
    if sha(model)!=contract['model_sha256']: raise ValueError('MODEL_HASH')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    before=memory();init=time.perf_counter();session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    with Candidate01Offline(ROOT/'android/app/src/main/assets/perception_vnext/dwpose_t_singlepass.onnx',ROOT/'android/app/src/main/assets/perception_vnext/hand_landmarks_detector.tflite') as runtime:
        init_seconds=time.perf_counter()-init;iterator,meta=iter_video_frames(run/'data/raw/ncslgr/videos'/f"{row['sample_id']}.mov",None)
        records=[];last=-1e9;cost=[];video_frames=0;last_logits=[];tick=time.perf_counter()
        for index,ms,source,image in iterator:
            video_frames+=1
            if ms<last+100-1e-6: continue
            t=time.perf_counter();observation=runtime.process_rgb(image);records.append((index,ms,source,observation))
            raw=make_raw(row,records,meta);features=derive_causal_segmentation_features(raw)
            logits=session.run(None,{'causal378':features[-61:][None]})[0]
            last_logits.append(logits[0,-1]);cost.append((time.perf_counter()-t)*1000);last=ms
        elapsed=time.perf_counter()-tick;after=memory()
    with np.load(run/row['feature_path'],allow_pickle=False) as saved: feature_error=float(np.abs(saved['X']-features).max())
    tokens=[];previous=-1
    for index in np.asarray(last_logits).argmax(-1).tolist():
        if index!=previous and index: tokens.append(contract['vocabulary'][index])
        previous=index
    result={'sample_id':row['sample_id'],'scope':'OFFLINE_WINDOWS_CPU_RGB_TO_EXPERIMENTAL_GLOSS; NOT_ANDROID_CAMERA_OR_PRODUCTION_PATH',
        'selected_observations':len(records),'decoded_video_frames':video_frames,'max_sampling_hz':10,
        'initialization_seconds':init_seconds,'replay_seconds_excluding_initialization':elapsed,
        'replay_observations_per_second':len(records)/elapsed,'selected_frame_ms_p50':float(np.percentile(cost,50)),
        'selected_frame_ms_p95':float(np.percentile(cost,95)),'selected_frame_scope':'pose + hand + raw packing + causal features + ONNX gloss model',
        'memory_before':before,'memory_after':after,'feature_replay_max_abs_error':feature_error,
        'greedy_gloss_output':tokens,'ground_truth':row['glosses'],'production_15FPS_gate':'NOT_PASSED_CADENCE_IS_10HZ_AND_NO_DEVICE_MEASUREMENT',
        'training_peak_RAM':'NOT_CAPTURED: PowerShell monitor observed venv launcher rather than worker; do not use that RSS as training memory'}
    with (run/'logs/replay_profile.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
    if feature_error>1e-6: raise SystemExit(1)
if __name__=='__main__': main()
