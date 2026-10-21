"""Packaged S2 replay/parity and timing; no sign/boundary labels are fabricated."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
from prepare_features import ROOT,checked_run,sha,SCHEMA

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args()
    run=checked_run(args.run_root);os.environ['KERAS_HOME']=str(run/'cache/keras');os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
    assets=ROOT/'android/app/src/main/assets/phone_causal_recognition_01'
    contract=json.loads((assets/'contract.json').read_text());norm=json.loads((assets/'cadence_aligned_normalization_stats.json').read_text())
    model=assets/'s2_phase7_fp32.tflite'
    if sha(model)!=contract['tflite_sha256'] or sha(assets/'cadence_aligned_normalization_stats.json')!=contract['normalization_asset_sha256']: raise ValueError('S2_ASSET_IDENTITY')
    if norm['feature_schema']!=contract['feature_schema'] or contract['feature_width']!=378: raise ValueError('S2_SCHEMA')
    mean=np.asarray(norm['mean'],np.float32);std=np.asarray(norm['std'],np.float32)
    start=time.perf_counter();interpreter=tf.lite.Interpreter(model_path=str(model),num_threads=4); cold=(time.perf_counter()-start)*1000
    timings=[];states=Counter();allocated=-1
    def invoke(x: np.ndarray) -> np.ndarray:
        nonlocal allocated
        if len(x)!=allocated:
            interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'],[1,len(x),378],strict=False)
            interpreter.allocate_tensors();allocated=len(x)
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'],x[None])
        tick=time.perf_counter();interpreter.invoke();timings.append((time.perf_counter()-tick)*1000)
        logits=interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
        exp=np.exp(logits-logits.max());return exp/exp.sum()
    fixtures=ROOT/'android/app/src/debug/assets/phone_causal_recognition_01'
    feature_file=fixtures/contract['fixture']['features_asset'];expected_file=fixtures/contract['fixture']['probabilities_asset']
    if sha(feature_file)!=contract['fixture']['features_sha256'] or sha(expected_file)!=contract['fixture']['probabilities_sha256']: raise ValueError('FIXTURE_IDENTITY')
    fixture=np.fromfile(feature_file,np.float32).reshape(-1,378);expected=np.fromfile(expected_file,np.float32).reshape(-1,3)
    predictions=np.asarray([invoke(fixture[max(0,i-60):i+1]) for i in range(len(fixture))])
    error=np.abs(predictions-expected)
    fixture_timing=list(timings);timings.clear()
    rows=json.loads((run/'data/processed/full/manifest.json').read_text())
    for row in rows:
        path=run/row['feature_path']
        if sha(path)!=row['feature_sha256']: raise ValueError('FEATURE_HASH')
        with np.load(path,allow_pickle=False) as f:
            if str(f['schema_version'].item())!=SCHEMA: raise ValueError('FEATURE_SCHEMA')
            x=(f['X']-mean)/std
        for i in range(len(x)):
            probs=invoke(x[max(0,i-60):i+1]);states[str(int(probs.argmax()))]+=1
    result={'status':'PASS' if error.max()<1e-4 else 'PARITY_FAILED','model_sha256':sha(model),
        'scope':'Windows S2-only replay, not full Android recognition baseline','threads':4,'provider':'TFLite XNNPACK CPU',
        'cold_interpreter_construction_ms':cold,'fixture_observations':len(fixture),'fixture_max_abs_probability_error':float(error.max()),
        'fixture_mean_abs_probability_error':float(error.mean()),'fixture_argmax_mismatches':int(np.count_nonzero(predictions.argmax(-1)!=expected.argmax(-1))),
        'fixture_timing_ms_p50':float(np.percentile(fixture_timing,50)),'fixture_timing_ms_p95':float(np.percentile(fixture_timing,95)),
        'research_clip_observations':len(timings),'research_s2_ms_p50':float(np.percentile(timings,50)),
        'research_s2_ms_p95':float(np.percentile(timings,95)),'research_s2_ms_sum':sum(timings),
        'predicted_state_counts_not_ground_truth':dict(states),'word_accuracy':'BLOCKED_NO_COMPATIBLE_HELD_OUT_BENCHMARK',
        'boundary_metrics':'BLOCKED_NO_TIMED_BOUNDARY_GROUND_TRUTH','nonsign_metrics':'BLOCKED_NO_LABELED_NONSIGN_FOOTAGE'}
    with (run/'logs/baseline_s2.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
    if result['status']!='PASS': raise SystemExit(1)
if __name__=='__main__': main()
