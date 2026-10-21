"""Frozen AppHear legacy258 classifier on the identical A validation rows."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from common import run_root,write,sha
p=argparse.ArgumentParser();p.add_argument('--run-root',required=True,type=Path);args=p.parse_args();run=run_root(args.run_root)
os.environ['KERAS_HOME']=str(run/'cache/keras');os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(1)
source=Path('D:/derived/derived/apphear3/runs/tcn_streaming_conversational_fps_v30/streaming.tflite')
interpreter=tf.lite.Interpreter(model_path=str(source),num_threads=2)
x=np.load(run/'experiment_A/data/val/X.npy',mmap_mode='r');metadata=np.load(run/'experiment_A/data/val/metadata.npz',allow_pickle=False)
pred=[];probabilities=[];timings=[]
interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'],[1,96,258],strict=False);interpreter.allocate_tensors()
for i in range(len(x)):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'],x[i:i+1]);tick=time.perf_counter();interpreter.invoke();timings.append((time.perf_counter()-tick)*1000)
    pf=interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
    if pf.shape!=(96,236):raise ValueError('LEGACY_MODEL_SHAPE')
    probs=(pf*metadata['mask'][i,:,None]).sum(0)/metadata['mask'][i].sum();probabilities.append(probs);pred.append(int(probs.argmax()))
from train_a import metrics
result=metrics(metadata['y'],np.asarray(pred),metadata['participant_ids']);result.update(model_path=str(source),model_sha256=sha(source),
    model_scope='unchanged legacy AppHear 236-way normalized258 model; CURRENT RecognitionV2 is incompatible',
    latency_p50_ms=float(np.percentile(timings,50)),latency_p95_ms=float(np.percentile(timings,95)))
write(run/'experiment_A/baseline.json',result);np.save(run/'experiment_A/baseline_probabilities.npy',np.asarray(probabilities))
print(json.dumps({k:result[k] for k in ('top1','macro_f1','rows','latency_p50_ms','latency_p95_ms')}))
