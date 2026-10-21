"""Read framework providers and measure CPU fallback, in the isolated process."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import importlib.metadata
p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True); args=p.parse_args()
run=args.run_root.resolve(); os.environ['KERAS_HOME']=str(run/'cache/keras');os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import torch
torch.set_num_threads(2);torch.manual_seed(20261019)
x=torch.randn((512,512),requires_grad=True);y=torch.randn((512,512)); times=[]
for i in range(8):
    x.grad=None;t=time.perf_counter();loss=(x@y).square().mean();loss.backward()
    if i>=2: times.append((time.perf_counter()-t)*1000)
import onnxruntime as ort
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2); tf.config.threading.set_inter_op_parallelism_threads(1)
result={'python':sys.version,'executable':sys.executable,
    'packages':{p:importlib.metadata.version(p) for p in ('torch','tensorflow','onnxruntime','numpy','opencv-python','pytest')},
    'tensorflow_physical_gpus':[str(d) for d in tf.config.list_physical_devices('GPU')],
    'tensorflow_built_with_cuda':tf.test.is_built_with_cuda(),'onnxruntime_available_providers':ort.get_available_providers(),
    'CPU_fallback':{'operation':'float32 512x512 matmul + backward, 2 threads','median_ms':float(np.median(times)),
                    'p95_ms':float(np.percentile(times,95)),'placement':str(loss.device),'finite_gradient':bool(torch.isfinite(x.grad).all())},
    'training_framework':'PyTorch CUDA; TF and ORT used only for current CPU landmark extraction'}
(run/'logs/framework_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
